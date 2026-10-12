"""FastAPI 检索服务: POST /search 上传服装图, 返回 Top-K 相似商品.

会员鉴权 (hhyms.com WordPress):
- 请求头 Authorization: Basic base64(wp用户名:应用密码)
- 按 VIP/等级扣每日配额, 额度见 wp_auth.QUOTA_MAP
- 无鉴权头时按最低档 (Lv1-Lv3, 10次/天, 按 IP 记次)
"""
import base64
import binascii
import io
import os
import re

from fastapi import FastAPI, File, UploadFile, Header
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

from searcher import FashionSearcher
import wp_auth

BASE = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE, "index")
IMAGES_DIR = os.path.join(BASE, "data", "images")
MERCHANTS_DIR = os.path.join(BASE, "data", "merchants")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB, 防大文件打爆内存
_FILENAME_RE = re.compile(r"^[\w.\-]+\.(jpg|jpeg|png|webp)$")  # 文件名白名单, 防穿越(含 Windows 反斜杠)
_MID_RE = re.compile(r"^[\w\-]+$")  # 商家 id 白名单
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")  # 操作员重载索引用, 不设则禁用

searcher = FashionSearcher(
    model_path=os.path.join(BASE, "models", "vision_fp16.onnx"),
    index_path=os.path.join(INDEX_DIR, "fashion.faiss"),
    ids_path=os.path.join(INDEX_DIR, "ids.npy"),
    db_path=os.path.join(INDEX_DIR, "meta.db"),
    merchant_index_path=os.path.join(INDEX_DIR, "merchant.faiss"),
    merchant_ids_path=os.path.join(INDEX_DIR, "merchant_ids.npy"),
)

app = FastAPI(title="服装拍照找款 MVP")


def _image_url(r):
    if r.get("source") == "merchant":
        return f"/mimages/{r['merchant_id']}/{os.path.basename(r['filename'])}"
    return f"/images/{r['filename']}"


@app.get("/health")
def health():
    return {"ok": True, "dim": 768, "ntotal": searcher.index.ntotal,
            "merchant_ntotal": searcher.merchant_ntotal}


@app.post("/search")
def search(file: UploadFile = File(...), top_k: int = 8,
           authorization: str = Header(default="")):
    # 会员鉴权 + 配额
    wp_id, tier = None, "lv1_3"
    if authorization.startswith("Basic "):
        try:
            cred = base64.b64decode(authorization[6:]).decode()
            username, app_pw = cred.split(":", 1)
            wp_id, tier, _desc = wp_auth.get_user_tier(username, app_pw)
        except Exception:
            return JSONResponse({"error": "WP 认证失败, 请检查用户名/应用密码"},
                                status_code=401)
    else:
        # 匿名: 按 IP 记次 (wp_id 用 hash(IP) 负数区分)
        wp_id = -abs(hash("anon")) % (10 ** 9)
    ok, remaining, msg = wp_auth.check_and_consume(wp_id, tier)
    if not ok:
        return JSONResponse({"error": msg}, status_code=429)
    # 用同步 def: FastAPI 自动放入线程池, 避免 ONNX 推理阻塞事件循环
    try:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return JSONResponse({"error": "文件过大（限 10MB）"}, status_code=413)
        img = Image.open(io.BytesIO(data))
        img.load()  # Image.open 是惰性的, 这里强制解码, 截断图在此抛异常
        results = searcher.search(img, top_k=max(1, min(top_k, 20)))
    except Exception:
        return JSONResponse({"error": "图片解析失败"}, status_code=400)
    finally:
        file.file.close()
    for r in results:
        r["image_url"] = _image_url(r)
    resp = {"results": results, "quota_msg": msg}
    if remaining >= 0:
        resp["quota_remaining"] = remaining
    return resp


@app.get("/mimages/{merchant_id}/{filename}")
def merchant_image(merchant_id: str, filename: str):
    """商家商品图: 双白名单 + basename, 防穿越."""
    if not _MID_RE.match(merchant_id) or not _FILENAME_RE.match(filename):
        return JSONResponse({"error": "非法参数"}, status_code=400)
    path = os.path.join(MERCHANTS_DIR, merchant_id, os.path.basename(filename))
    # 防 symlink 越狱: 商家目录若被链到别处, 拒绝服务
    if os.path.commonpath([os.path.realpath(path),
                           os.path.realpath(MERCHANTS_DIR)]) != os.path.realpath(MERCHANTS_DIR):
        return JSONResponse({"error": "非法参数"}, status_code=400)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


@app.get("/merchants")
def merchants():
    """商家列表 (操作员页用): 商家名/联系方式/商品数."""
    import sqlite3
    with sqlite3.connect(os.path.join(INDEX_DIR, "meta.db")) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT m.merchant_id, m.name, m.contact,"
            " COUNT(p.id) AS n_products FROM merchants m"
            " LEFT JOIN products p ON p.merchant_id=m.merchant_id"
            " AND p.source='merchant' GROUP BY m.merchant_id").fetchall()
    return {"merchants": [dict(r) for r in rows]}


@app.post("/admin/reload")
def admin_reload(token: str = ""):
    """操作员入库后重载商家索引. 需 ADMIN_TOKEN, 未设置则禁用."""
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    searcher.reload_merchant()
    return {"ok": True, "merchant_ntotal": searcher.merchant_ntotal}


@app.get("/images/{filename}")
def image(filename: str):
    if not _FILENAME_RE.match(filename):
        return JSONResponse({"error": "非法文件名"}, status_code=400)
    path = os.path.join(IMAGES_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)
