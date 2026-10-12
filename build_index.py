#!/usr/bin/env python3
"""离线建库: 商品图 -> FashionSigLIP embedding -> FAISS 索引 -> SQLite 元数据.

用法:
    python build_index.py --images data/images --meta data/styles.csv \
        --model models/vision_fp16.onnx --out index --limit 5000 --batch 32

产出 (out 目录):
    fashion.faiss  FAISS 内积索引 (向量已 L2 归一化, 即余弦检索)
    ids.npy        int64 商品 id 数组, 与索引位置一一对齐
    meta.db        SQLite 商品元数据 (id/类目/颜色/名称/文件名)
"""
import argparse
import csv
import os
import sqlite3
import sys
import time

import faiss
import numpy as np
import onnxruntime as ort
from PIL import Image

from searcher import preprocess_image, l2norm


def load_products(images_dir, meta_csv, limit):
    """读 styles.csv, 按 masterCategory 轮询取样保证类目覆盖, 只保留图片存在的商品."""
    groups = {}
    with open(meta_csv, newline="", encoding="utf-8-sig") as f:  # utf-8-sig 防 BOM
        for row in csv.DictReader(f):
            try:
                pid = int(row["id"])
            except (ValueError, KeyError):
                continue
            fname = f"{pid}.jpg"
            if not os.path.exists(os.path.join(images_dir, fname)):
                continue
            row["filename"] = fname
            groups.setdefault(row.get("masterCategory") or "Unknown", []).append(row)
    cats = sorted(groups)
    picked = []
    while len(picked) < limit and any(groups[c] for c in cats):
        for c in cats:
            if groups[c] and len(picked) < limit:
                picked.append(groups[c].pop(0))
    print(f"取样 {len(picked)} 件, 覆盖 {len(cats)} 个大类: {cats}")
    return picked


def build(args):
    os.makedirs(args.out, exist_ok=True)
    products = load_products(args.images, args.meta, args.limit)
    if not products:
        sys.exit("没有可用商品图, 检查 --images 与 --meta 路径")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
    sess = ort.InferenceSession(args.model, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    dim = int(sess.get_outputs()[0].shape[1])  # 从模型推导, 不硬编码

    index = faiss.IndexFlatIP(dim)
    ids, batch_imgs, batch_ids = [], [], []
    t0 = time.time()
    done = 0

    def flush():
        nonlocal done
        if not batch_imgs:
            return
        x = np.concatenate(batch_imgs, axis=0)
        feat = sess.run([out_name], {in_name: x})[0].astype(np.float32)
        index.add(l2norm(feat))
        ids.extend(batch_ids)
        done += len(batch_ids)
        batch_imgs.clear()
        batch_ids.clear()
        if done % 500 == 0 or done == len(products):
            dt = time.time() - t0
            print(f"  {done}/{len(products)}  {dt:.0f}s  {done/max(dt,1):.1f} 张/秒", flush=True)

    for p in products:
        try:
            with Image.open(os.path.join(args.images, p["filename"])) as img:
                batch_imgs.append(preprocess_image(img))
            batch_ids.append(int(p["id"]))
        except Exception as e:  # 坏图跳过, 不中断整批
            print(f"跳过坏图 {p['filename']}: {e}")
            continue
        if len(batch_imgs) >= args.batch:
            flush()
    flush()

    faiss.write_index(index, os.path.join(args.out, "fashion.faiss"))
    np.save(os.path.join(args.out, "ids.npy"), np.array(ids, dtype=np.int64))

    db_path = os.path.join(args.out, "meta.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE products(id INTEGER PRIMARY KEY, filename TEXT, gender TEXT,"
        " masterCategory TEXT, subCategory TEXT, articleType TEXT,"
        " baseColour TEXT, productDisplayName TEXT)"
    )
    idset = set(ids)
    con.executemany(
        "INSERT OR REPLACE INTO products VALUES (?,?,?,?,?,?,?,?)",  # 防 CSV 重复 id 崩溃
        [(int(p["id"]), p["filename"], p.get("gender"), p.get("masterCategory"),
          p.get("subCategory"), p.get("articleType"), p.get("baseColour"),
          p.get("productDisplayName"))
         for p in products if int(p["id"]) in idset],
    )
    con.commit()
    con.close()
    print(f"完成: 索引 {index.ntotal} 条向量 -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    build(ap.parse_args())
