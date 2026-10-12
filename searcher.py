"""服装拍照找款 MVP 核心检索模块.

FashionSigLIP (Google SigLIP 微调) vision tower ONNX + FAISS, 供 FastAPI 与 Gradio 共用.
纯 onnxruntime, 不依赖 torch, 内存友好 (vision fp16 仅 177MB).
"""
import os
import sqlite3
import threading
import json
import hashlib

import faiss
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

IMG_SIZE = 224
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """按 SigLIP 预处理: squash resize 到 224, mean=0.5/std=0.5 归一化.

    返回 NCHW float32, shape (1, 3, 224, 224).
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...]


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def ensure_schema(db_path: str):
    """幂等: 给 meta.db 补商家相关列与 merchants 表, 老库自动升级."""
    with sqlite3.connect(db_path) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(products)")}
        for ddl in ("source TEXT DEFAULT 'kaggle'", "merchant_id TEXT",
                    "product_name TEXT", "price TEXT", "moq TEXT",
                    "phash INTEGER"):
            if ddl.split()[0] not in cols:
                con.execute(f"ALTER TABLE products ADD COLUMN {ddl}")
        con.execute("CREATE TABLE IF NOT EXISTS merchants("
                    "merchant_id TEXT PRIMARY KEY, name TEXT, contact TEXT)")
        con.commit()


class FashionSearcher:
    """加载 ONNX 模型 + FAISS 索引, 对外提供 embed / search."""

    def __init__(self, model_path: str, index_path: str, ids_path: str, db_path: str,
                 merchant_index_path: str = None, merchant_ids_path: str = None):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        self.index = faiss.read_index(index_path)
        self.ids = np.load(ids_path, allow_pickle=False)  # int64, 与 index 位置一一对齐
        self.db_path = db_path
        ensure_schema(self.db_path)  # 老库自动升级: 加商家相关列与 merchants 表
        # 串行化 ONNX 推理: intra_op 线程数已接近核数, 并发请求若各跑各的会线程颠簸;
        # 加锁后多请求排队推理, 单用户延迟不变, 高并发尾延迟反而更稳
        self._infer_lock = threading.Lock()
        if self.index.ntotal != len(self.ids):
            raise ValueError("index 与 ids 数量不一致")
        # 商家索引可选: 代劳入库的商家商品走独立 FAISS, 查询时双路合并
        self._merchant_index_path = merchant_index_path
        self._merchant_ids_path = merchant_ids_path
        self._merchant_meta_path = (
            os.path.join(os.path.dirname(merchant_index_path), "merchant.meta.json")
            if merchant_index_path else None)
        # (faiss_index, ids) 元组整体原子替换: search 线程要么看到旧的要么看到新的,
        # 与 reload_merchant 配对, 消除 mindex/mids 分步赋值的中途态
        self._merchant = None
        self.reload_merchant()

    @property
    def merchant_ntotal(self):
        return self._merchant[0].ntotal if self._merchant else 0

    def reload_merchant(self):
        """重载商家索引. 校验 manifest; 任何异常 -> 商家检索降级关闭, 主索引不受影响."""
        state = None
        try:
            if not (self._merchant_index_path
                    and os.path.exists(self._merchant_index_path)):
                return
            with open(self._merchant_meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            h = hashlib.sha256()
            with open(self._merchant_ids_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != meta.get("ids_sha256"):
                raise ValueError("merchant_ids.npy 与 manifest sha256 不一致 (疑似半成品)")
            ids = np.load(self._merchant_ids_path, allow_pickle=False)
            index = faiss.read_index(self._merchant_index_path)
            if not (index.ntotal == len(ids) == meta.get("ntotal")):
                raise ValueError("商家 index/ids/manifest 数量不一致")
            state = (index, ids)
        except Exception as e:
            # 降级: 商家检索关闭, 主流程继续 (崩溃重建后下次 reload 自动恢复)
            print(f"商家索引加载失败, 商家检索已降级关闭: {e}")
        self._merchant = state

    def embed(self, img: Image.Image) -> np.ndarray:
        x = preprocess_image(img)
        with self._infer_lock:
            feat = self.session.run([self.out_name], {self.in_name: x})[0]
        return l2norm(feat.astype(np.float32))

    def _search_single(self, index, ids, q, k):
        """单路 FAISS 检索, 返回 [(score, product_rowid)], 过滤 -1 填充."""
        if index is None or index.ntotal == 0:
            return []
        k = min(k, index.ntotal)
        scores, pos = index.search(q, k)
        # FAISS 在 top_k > ntotal 时用 idx=-1 填充, 必须过滤, 否则负索引取到错误商品
        return [(float(s), int(ids[p])) for s, p in zip(scores[0], pos[0]) if p >= 0]

    def search(self, img: Image.Image, top_k: int = 8):
        top_k = max(1, min(top_k, 20))  # 核心层设防: 防超大 top_k 打爆 FAISS/SQL
        q = self.embed(img)
        # 双路检索: 公开数据集 + 商家库, 按相似度合并取 Top-K
        cand = self._search_single(self.index, self.ids, q, top_k)
        m = self._merchant  # 一次性读元组, 与 reload 的原子替换配对
        if m is not None:
            cand += self._search_single(m[0], m[1], q, top_k)
        cand.sort(key=lambda x: -x[0])
        cand = cand[:top_k]
        if not cand:
            return []
        pids = [pid for _, pid in cand]
        metas = self._lookup(pids)
        return [
            {"id": pid, "score": s, **m}
            for (s, pid), m in zip(cand, metas)
            if m.get("filename")  # DB 缺行时跳过, 避免下游拿到空路径
        ]

    def _lookup(self, pids):
        pids = [int(p) for p in pids]
        if not pids:
            return []
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT p.id, p.filename, p.source, p.merchant_id,"
                " p.product_name, p.price, p.moq,"
                " p.gender, p.masterCategory, p.subCategory,"
                " p.articleType, p.baseColour, p.productDisplayName,"
                " m.name AS merchant_name, m.contact AS merchant_contact"
                " FROM products p LEFT JOIN merchants m"
                " ON p.merchant_id = m.merchant_id"
                f" WHERE p.id IN ({','.join('?' * len(pids))})",
                pids,
            ).fetchall()
        by_id = {r["id"]: dict(r) for r in rows}
        return [by_id.get(pid, {"filename": ""}) for pid in pids]
