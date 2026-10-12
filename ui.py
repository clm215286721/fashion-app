"""Gradio demo: 上传服装照片 -> 展示 Top-K 相似商品 (含商家/批发价).

Tab 1 以图搜款: 公开数据集 + 商家库双路检索.
Tab 2 商家入库 (操作员): 拖拽商家商品图 -> 落盘 -> 重建商家索引 -> 热重载.
"""
import csv
import os
import re
import subprocess
import sys

import gradio as gr
from PIL import Image

from searcher import FashionSearcher
import intake_lib

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE, "data", "images")
MERCHANTS_DIR = os.path.join(BASE, "data", "merchants")
_MID_RE = re.compile(r"^[\w\-]+$")
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

searcher = FashionSearcher(
    model_path=os.path.join(BASE, "models", "vision_fp16.onnx"),
    index_path=os.path.join(BASE, "index", "fashion.faiss"),
    ids_path=os.path.join(BASE, "index", "ids.npy"),
    db_path=os.path.join(BASE, "index", "meta.db"),
    merchant_index_path=os.path.join(BASE, "index", "merchant.faiss"),
    merchant_ids_path=os.path.join(BASE, "index", "merchant_ids.npy"),
)


def _local_path(r):
    if r.get("source") == "merchant":
        return os.path.join(MERCHANTS_DIR, r["merchant_id"],
                            os.path.basename(r["filename"]))
    return os.path.join(IMAGES_DIR, r["filename"])


def _caption(r):
    if r.get("source") == "merchant":
        return (f"【商家】{r.get('product_name','')} | {r.get('merchant_name','')} | "
                f"批发价 {r.get('price','')} | 起批量 {r.get('moq','')} | "
                f"{r.get('merchant_contact','')} (相似度 {r['score']:.3f})")
    return (f"{r.get('productDisplayName','')} | {r.get('articleType','')} "
            f"{r.get('baseColour','')} (相似度 {r['score']:.3f})")


def run_search(img, top_k):
    if img is None:
        return []
    results = searcher.search(Image.fromarray(img), top_k=int(top_k))
    return [(_local_path(r), _caption(r)) for r in results]


def _upath(f):
    """兼容 gr.File 返回 str 路径或文件对象的两种情况."""
    return f if isinstance(f, str) else f.name


def intake(mid, mname, contact, pname, price, moq, files):
    """操作员入库: 微信收到的图拖进来 -> 去黑边/切拼图/去重 -> 落盘
    -> 追加 products.csv -> 重建商家索引 -> 热重载."""
    log = []
    mid = (mid or "").strip()
    if not _MID_RE.match(mid):
        return "商家 id 非法 (只允许字母/数字/下划线/横杠)"
    if not files:
        return "请先上传商品图"
    mdir = os.path.join(MERCHANTS_DIR, mid)
    os.makedirs(mdir, exist_ok=True)
    if os.path.commonpath([os.path.realpath(mdir),
                           os.path.realpath(MERCHANTS_DIR)]) != os.path.realpath(MERCHANTS_DIR):
        return "商家目录异常 (疑似 symlink), 拒绝写入"

    seen = intake_lib.load_seen_hashes(MERCHANTS_DIR)
    k = intake_lib.next_index(mdir, mid)
    rows, t_new, t_dup, t_skip = [], 0, 0, 0
    n_files = len(files)
    for i, f in enumerate(files, 1):
        up = _upath(f)
        try:
            panels = intake_lib.extract_panels(up)
        except Exception:
            t_skip += 1
            log.append(f"跳过打不开的图片: {os.path.basename(up)}")
            continue
        keep, n_dup = intake_lib.dedup_panels(panels, seen)
        t_dup += n_dup
        base = (pname or "").strip() or mid
        pname_i = base if n_files == 1 else f"{base}-{i}"
        for panel in keep:
            while True:  # O_EXCL 原子创建; 并发上传同一商家 k 碰撞则递增重试
                k += 1
                dst = os.path.join(mdir, f"{mid}_p{k}.jpg")
                try:
                    intake_lib.write_panel_atomic(panel, dst)
                    break
                except FileExistsError:
                    continue
            rows.append([os.path.basename(dst), pname_i, price or "", moq or "",
                         contact or "", mname or mid])
            t_new += 1
    if not rows:
        return "\n".join(log) + "\n没有可用图片 (全部重复或打不开)"
    log.append(f"处理 {n_files} 个文件: 切出 {t_new + t_dup} 张面板, "
               f"新增 {t_new} 张, 去重 {t_dup} 张, 跳过 {t_skip} 个文件")

    # 追加 products.csv
    csv_path = os.path.join(mdir, "products.csv")
    need_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as cf:
        w = csv.writer(cf)
        if need_header:
            w.writerow(["image", "product_name", "price", "moq",
                        "contact", "merchant_name"])
        w.writerows(rows)

    # 子进程重建商家索引 (与 UI 进程隔离, 不额外占模型内存)
    p = subprocess.run(
        [sys.executable, os.path.join(BASE, "build_merchant_index.py"),
         "--merchants", MERCHANTS_DIR,
         "--model", os.path.join(BASE, "models", "vision_fp16.onnx"),
         "--out", os.path.join(BASE, "index")],
        capture_output=True, text=True, timeout=600)
    log.append(p.stdout.strip())
    if p.returncode != 0:
        log.append("建库失败:\n" + p.stderr.strip()[-2000:])
        return "\n".join(log)
    searcher.reload_merchant()
    log.append(f"商家索引已热重载, 当前共 {searcher.merchant_ntotal} 件商家商品")
    return "\n".join(log)


with gr.Blocks(title="服装拍照找款 MVP") as demo:
    gr.Markdown("# 服装拍照找款 MVP\nFashionSigLIP + FAISS 以图搜衣 (公开数据集 + 商家库双路检索)")
    with gr.Tab("以图搜款"):
        with gr.Row():
            img_in = gr.Image(type="numpy", label="上传服装照片")
            with gr.Column():
                topk = gr.Slider(1, 20, value=8, step=1, label="返回数量")
                btn = gr.Button("搜索", variant="primary")
        gallery = gr.Gallery(label="相似商品 Top-K (【商家】= 商家库, 含批发价/联系方式)",
                             columns=4)
        btn.click(run_search, inputs=[img_in, topk], outputs=gallery)
    with gr.Tab("商家入库 (操作员)"):
        gr.Markdown("微信收到档口商品图 -> 填表 -> 上传 -> 自动入库. "
                    "同一商家重复发的图会自动去重.")
        with gr.Row():
            mid = gr.Textbox(label="商家 id (字母/数字/下划线)", value="m002")
            mname = gr.Textbox(label="商家名称")
            contact = gr.Textbox(label="联系方式 (微信/电话)")
        with gr.Row():
            pname = gr.Textbox(label="商品名")
            price = gr.Textbox(label="批发价")
            moq = gr.Textbox(label="起批量")
        files = gr.File(label="商品图 (可多选, jpg/png/webp)",
                        file_count="multiple")
        intake_btn = gr.Button("入库并重建索引", variant="primary")
        intake_log = gr.Textbox(label="入库日志", lines=12)
        intake_btn.click(intake,
                         inputs=[mid, mname, contact, pname, price, moq, files],
                         outputs=intake_log)

if __name__ == "__main__":
    # 对外暴露时建议设 GRADIO_USER / GRADIO_PASSWORD 环境变量加认证
    auth = None
    if os.environ.get("GRADIO_USER"):
        auth = (os.environ["GRADIO_USER"], os.environ.get("GRADIO_PASSWORD", ""))
    demo.launch(server_name="0.0.0.0", server_port=7860, auth=auth)
