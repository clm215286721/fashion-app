#!/usr/bin/env python3
"""商家商品代录入建库: data/merchants/{merchant_id}/ -> index/merchant.faiss.

每次全量重建商家索引 (商家图少, embedding 快, 无增量一致性问题):
  1. 扫描 data/merchants/*/products.csv (无 CSV 则兜底扫描图片)
  2. dHash 去重: 同一商家重复发的图跳过 (跨商家不去重, 同款多家要保留)
  3. FashionSigLIP embedding -> FAISS 独立索引 (与公开数据集双路检索)
  4. SQLite: merchants 表重建; products 表 source='merchant' 的行删后重插

products.csv 列: image,product_name,price,moq,contact,merchant_name
图片格式: jpg/jpeg/png/webp (微信转发的都行)

用法:
  python build_merchant_index.py --merchants data/merchants \
      --model models/vision_fp16.onnx --out index
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import sqlite3
import sys
import time

import faiss
import numpy as np
import onnxruntime as ort
from PIL import Image

from searcher import ensure_schema, l2norm, preprocess_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_MID_RE = re.compile(r"^[\w\-]+$")


def dhash(img) -> int:
    """64 位差异哈希, 快速去重用 (微信重复转发的图能抓住)."""
    g = img.convert("L").resize((9, 8), Image.BICUBIC)
    px = np.asarray(g, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def scan_merchants(root):
    """扫商家目录, 返回 [(merchant_id, abs_img_path, rel_filename, meta)].

    meta 来自 products.csv, 无 CSV 或行缺失时字段为空 (不丢图).
    """
    items = []
    if not os.path.isdir(root):
        return items
    for mid in sorted(os.listdir(root)):
        mdir = os.path.join(root, mid)
        if not os.path.isdir(mdir) or not _MID_RE.match(mid):
            print(f"跳过非法商家目录: {mid}")
            continue
        meta_by_img = {}
        csv_path = os.path.join(mdir, "products.csv")
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    img = (row.get("image") or "").strip()
                    if img:
                        meta_by_img[img] = row
        seen_hash = set()
        for fn in sorted(os.listdir(mdir)):
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            ap = os.path.join(mdir, fn)
            try:
                with Image.open(ap) as im:
                    im.load()
                    h = dhash(im)
                    key = (h, os.path.getsize(ap))  # 哈希+大小联合去重
            except Exception as e:
                print(f"跳过坏图 {mid}/{fn}: {e}")
                continue
            if key in seen_hash:  # 同一商家重复发的图
                print(f"去重跳过 {mid}/{fn}")
                continue
            seen_hash.add(key)
            row = meta_by_img.get(fn, {})
            items.append((mid, ap, f"{mid}/{fn}", {
                "product_name": row.get("product_name") or fn,
                "price": row.get("price") or "",
                "moq": row.get("moq") or "",
                "contact": row.get("contact") or "",
                "merchant_name": row.get("merchant_name") or mid,
                # dhash 可能顶位置 1 (超 SQLite 有符号 64 位), 转有符号存
                "phash": h if h < 2**63 else h - 2**64,
            }))
    return items


def build(args):
    os.makedirs(args.out, exist_ok=True)
    db_path = os.path.join(args.out, "meta.db")
    ensure_schema(db_path)

    items = scan_merchants(args.merchants)
    if not items:
        sys.exit("商家目录为空, 先把微信收到的图放进 data/merchants/{商家id}/")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
    sess = ort.InferenceSession(args.model, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    dim = int(sess.get_outputs()[0].shape[1])

    # 先 embedding (失败在这里暴露, 不污染 DB)
    vecs, ok_items = [], []
    t0 = time.time()
    for mid, ap, rel, meta in items:
        try:
            with Image.open(ap) as im:
                x = preprocess_image(im)
            feat = sess.run([out_name], {in_name: x})[0].astype(np.float32)
            vecs.append(l2norm(feat))
            ok_items.append((mid, rel, meta))
        except Exception as e:
            print(f"embedding 失败 {rel}: {e}")
    if not ok_items:
        sys.exit("没有成功 embedding 的商家图")
    print(f"embedding {len(ok_items)} 张, {time.time()-t0:.1f}s")

    # DB: 商家部分全量重建 (products 表 source='merchant' 的行 + merchants 表)
    con = sqlite3.connect(db_path)
    con.execute("DELETE FROM products WHERE source='merchant'")
    con.execute("DELETE FROM merchants")
    merchants = {}
    rowids = []
    for mid, rel, meta in ok_items:
        merchants[mid] = (meta["merchant_name"], meta["contact"])
        cur = con.execute(
            "INSERT INTO products(source, filename, merchant_id, product_name,"
            " price, moq, phash) VALUES ('merchant',?,?,?,?,?,?)",
            (rel, mid, meta["product_name"], meta["price"],
             meta["moq"], meta["phash"]))
        rowids.append(cur.lastrowid)
    con.executemany(
        "INSERT OR REPLACE INTO merchants(merchant_id, name, contact)"
        " VALUES (?,?,?)",
        [(mid, name, contact) for mid, (name, contact) in merchants.items()])
    con.commit()
    con.close()

    # FAISS: 原子替换 (先写 tmp 再 rename, 崩溃重跑即可恢复)
    index = faiss.IndexFlatIP(dim)
    index.add(np.concatenate(vecs, axis=0))
    faiss_path = os.path.join(args.out, "merchant.faiss")
    ids_path = os.path.join(args.out, "merchant_ids.npy")
    faiss.write_index(index, faiss_path + ".tmp")
    with open(ids_path + ".tmp", "wb") as f:  # 传文件句柄, np.save 不再补 .npy 后缀
        np.save(f, np.array(rowids, dtype=np.int64))
    os.replace(faiss_path + ".tmp", faiss_path)
    os.replace(ids_path + ".tmp", ids_path)
    # manifest 最后写: reader 校验 sha256+ntotal, 半成品会被降级禁用而非误读
    with open(ids_path, "rb") as f:
        ids_sha = hashlib.sha256(f.read()).hexdigest()
    meta_path = os.path.join(args.out, "merchant.meta.json")
    with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"ntotal": index.ntotal, "ids_sha256": ids_sha,
                   "built_at": datetime.datetime.now().isoformat(timespec="seconds")}, f)
    os.replace(meta_path + ".tmp", meta_path)
    print(f"完成: {len(merchants)} 家商家, {index.ntotal} 件商品 -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merchants", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    build(ap.parse_args())
