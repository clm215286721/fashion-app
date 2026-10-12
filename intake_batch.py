#!/usr/bin/env python3
"""压缩包一键入库.

用法:
    intake_batch.py <zip路径> [--name 商家名] [--contact 联系方式]
                    [--price 批发价] [--product 品名] [--into m003]

约定:
  - 包里一家商家放一个文件夹(文件夹名作备注); 没有文件夹则整包视为一家.
  - 图片混放即可: 多宫格拼图自动切分, 手机截图自动去黑边, 单张直接入库.
  - 与已入库图片重复(dHash+文件大小)自动跳过并报告.
  - --into m003: 把图追加到已有商家(补货), 否则新建 m00N 档案.
  - 跑完自动重建商家索引并做召回自检.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intake_lib

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_batches(tmp):
    """包内顶层文件夹=一家; 根目录直放的图片自成一家."""
    batches = {}
    for entry in sorted(os.listdir(tmp)):
        if entry.startswith(".") or entry == "__MACOSX":
            continue
        full = os.path.join(tmp, entry)
        if os.path.isdir(full):
            has_img = any(
                os.path.splitext(f)[1].lower() in IMG_EXTS and not f.startswith(".")
                for _, _, fs in os.walk(full) for f in fs)
            if has_img:
                batches[entry] = full
    root_imgs = [f for f in os.listdir(tmp)
                 if os.path.isfile(os.path.join(tmp, f))
                 and os.path.splitext(f)[1].lower() in IMG_EXTS]
    if root_imgs:
        batches["."] = tmp
    return batches


def load_existing(mdir):
    rows, k = [], 0
    p = os.path.join(mdir, "products.csv")
    if os.path.exists(p):
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        for r in rows:
            tail = (r.get("image") or "").rsplit("_p", 1)[-1].split(".")[0]
            if tail.isdigit():
                k = max(k, int(tail))
    return rows, k


def intake_one_image(fp, mdir, prefix, seen, pname, args, state):
    """单张源图: 去黑边/切分 -> 去重 -> 续编号落盘. 返回 (新增数, 跳过数)."""
    panels = intake_lib.extract_panels(fp)  # 打不开抛异常, 调用方捕获跳过
    keep, n_dup = intake_lib.dedup_panels(panels, seen)
    n_new = 0
    for panel in keep:
        state["k"] += 1
        dst = os.path.join(mdir, f"{prefix}_p{state['k']}.jpg")
        intake_lib.write_panel_atomic(panel, dst)
        state["rows"].append({"image": os.path.basename(dst),
                              "product_name": pname,
                              "price": args.price, "moq": "",
                              "contact": args.contact,
                              "merchant_name": args.name})
        n_new += 1
    return n_new, n_dup


def self_check(mvp, mids):
    """召回自检: 每个新品名拿首张去搜, 同品名面板应被召回."""
    import sqlite3
    from searcher import FashionSearcher
    s = FashionSearcher(
        os.path.join(mvp, "models/vision_fp16.onnx"),
        os.path.join(mvp, "index/fashion.faiss"),
        os.path.join(mvp, "index/ids.npy"),
        os.path.join(mvp, "index/meta.db"),
        os.path.join(mvp, "index/merchant.faiss"),
        os.path.join(mvp, "index/merchant_ids.npy"))
    con = sqlite3.connect(os.path.join(mvp, "index/meta.db"))
    ok = True
    for mid in mids:
        prods = {}
        for rel, pname in con.execute(
                "SELECT filename, product_name FROM products "
                "WHERE merchant_id=? AND source='merchant'", (mid,)):
            prods.setdefault(pname or rel, []).append(rel)
        for pname, rels in sorted(prods.items()):
            with Image.open(os.path.join(mvp, "data/merchants", rels[0])) as q:
                res = s.search(q, top_k=len(rels) + 3)
            hit = sum(1 for x in res if (x.get("product_name") or "") == pname)
            flag = "OK" if hit >= (len(rels) + 1) // 2 else "WARN"
            if flag == "WARN":
                ok = False
            print(f"  自检[{flag}] {mid} {pname}: {hit}/{len(rels)}")
    con.close()
    return ok


def main():
    ap = argparse.ArgumentParser(description="压缩包一键入库")
    ap.add_argument("zip", help="压缩包路径")
    ap.add_argument("--name", default="待补充", help="商家名")
    ap.add_argument("--contact", default="", help="联系方式/地址")
    ap.add_argument("--price", default="", help="批发价")
    ap.add_argument("--product", default="新品", help="品名(多图自动加 -1/-2 后缀)")
    ap.add_argument("--into", default="", help="追加到已有商家目录, 如 m003")
    args = ap.parse_args()

    mvp = os.path.dirname(os.path.abspath(__file__))
    merch_root = os.path.join(mvp, "data", "merchants")
    seen = intake_lib.load_seen_hashes(merch_root)

    tmp = tempfile.mkdtemp(prefix="intake_")
    try:
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(tmp)
        batches = collect_batches(tmp)
        if not batches:
            sys.exit("包里没有图片, 退出")

        if args.into:
            mids = [args.into]
            batch_list = [(args.into, bdir) for _, bdir in batches.items()]
        else:
            nums = [int(d[1:]) for d in os.listdir(merch_root)
                    if d.startswith("m") and d[1:].isdigit()]
            nxt = max(nums, default=0) + 1
            mids, batch_list = [], []
            for label, bdir in batches.items():
                mid = f"m{nxt:03d}"
                nxt += 1
                mids.append(mid)
                batch_list.append((mid, bdir))
        print(f"处理 {len(batch_list)} 家: {[m for m, _ in batch_list]}")

        for mid, bdir in batch_list:
            mdir = os.path.join(merch_root, mid)
            os.makedirs(mdir, exist_ok=True)
            rows, k = load_existing(mdir)
            state = {"rows": rows, "k": k}
            files = []
            for dp, _, fns in os.walk(bdir):
                for fn in sorted(fns):
                    if fn.startswith(".") or os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                        continue
                    files.append(os.path.join(dp, fn))
            t_new, t_dup = 0, 0
            for i, fp in enumerate(files, 1):
                pname = args.product if len(files) == 1 else f"{args.product}-{i}"
                try:
                    a, b = intake_one_image(fp, mdir, mid, seen, pname, args, state)
                    t_new += a
                    t_dup += b
                except Exception as e:
                    print(f"  跳过 {os.path.basename(fp)}: {e}")
            if state["rows"]:
                with open(os.path.join(mdir, "products.csv"), "w",
                          newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=state["rows"][0].keys())
                    w.writeheader()
                    w.writerows(state["rows"])
            print(f"[{mid}] 新增 {t_new} 张, 跳过重复 {t_dup} 张")

        print("重建索引...")
        r = subprocess.run(
            [sys.executable, os.path.join(mvp, "build_merchant_index.py"),
             "--merchants", merch_root,
             "--model", os.path.join(mvp, "models/vision_fp16.onnx"),
             "--out", os.path.join(mvp, "index")],
            capture_output=True, text=True)
        tail = r.stdout.strip().splitlines()
        print(tail[-1] if tail else r.stderr[-500:])
        if r.returncode != 0:
            return 2
        print("召回自检...")
        self_check(mvp, mids)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
