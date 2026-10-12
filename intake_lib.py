"""商家图片入库公共库.

网页后台 (ui.py) 与命令行 (intake_batch.py) 共用同一套处理逻辑,
避免两套代码漂移.

流水线: 打开 -> 去黑边(手机截图) -> 拼图切分 -> dhash 去重 -> 原子落盘.

安全分工:
  - 调用方保证: mid 合法性、商家目录 symlink 收容检查、CSV 写入.
  - 本模块保证: 落盘文件名限定为 {prefix}_p{k}.jpg、
    O_EXCL 原子创建 (并发上传同一商家不互相覆盖).
"""
import os
import re
import shutil

import numpy as np
from PIL import Image

from build_merchant_index import dhash

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def crop_black_borders(im, thresh=12):
    """手机截图去上下黑边, 只留产品照片."""
    a = np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=2)
    rows, cols = a.mean(axis=1), a.mean(axis=0)
    ys = np.where(rows > thresh)[0]
    xs = np.where(cols > thresh)[0]
    if len(ys) == 0 or len(xs) == 0:
        return im
    return im.crop((int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1))


def grid_panels(im, min_size=150, white_thresh=235, frac_thresh=0.75):
    """白底拼图切分: 某整行/整列白色像素占比>frac_thresh 视为分隔带.

    只认内部 gutter (不切边缘白边, 单张产品图原样保留, 去重才稳定).
    行优先 / 列优先两种切分, 取面板数多的那个 (兼容 1+2 这类异形拼图).
    """
    a = np.asarray(im.convert("RGB"), dtype=np.float32).mean(axis=2)
    white = a > white_thresh

    def split_1d(frac):
        n = len(frac)
        gutters, s = [], None
        for i, f in enumerate(frac):
            if f > frac_thresh and s is None:
                s = i
            elif f <= frac_thresh and s is not None:
                if i - s >= 3:
                    gutters.append((s, i))
                s = None
        if s is not None and n - s >= 3:
            gutters.append((s, n))
        # 只要内部 gutter
        gutters = [(x0, x1) for x0, x1 in gutters if x0 > 0 and x1 < n]
        cuts, prev = [], 0
        for x0, x1 in gutters:
            cuts.append((prev, x0))
            prev = x1
        cuts.append((prev, n))
        return [(x0, x1) for x0, x1 in cuts if x1 - x0 > min_size]

    rows_first, cols_first = [], []
    for y0, y1 in split_1d(white.mean(axis=1)):
        for x0, x1 in split_1d(white[y0:y1, :].mean(axis=0)):
            rows_first.append((x0, y0, x1, y1))
    for x0, x1 in split_1d(white.mean(axis=0)):
        for y0, y1 in split_1d(white[:, x0:x1].mean(axis=1)):
            cols_first.append((x0, y0, x1, y1))
    return rows_first if len(rows_first) >= len(cols_first) else cols_first


def load_seen_hashes(merchants_root):
    """已入库图片的 dhash 集合, 用于跨批次去重 (只比哈希, 不比文件大小)."""
    seen = set()
    for dp, _, fns in os.walk(merchants_root):
        for fn in fns:
            if fn.startswith(".") or os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            try:
                with Image.open(os.path.join(dp, fn)) as e:
                    seen.add(dhash(e))
            except Exception:
                pass
    return seen


def next_index(mdir, prefix):
    """目录下 {prefix}_p{k}.jpg 的最大 k (续编号用)."""
    pat = re.compile(rf"{re.escape(prefix)}_p(\d+)\.jpg$")
    k = 0
    if os.path.isdir(mdir):
        for fn in os.listdir(mdir):
            m = pat.fullmatch(fn)
            if m:
                k = max(k, int(m.group(1)))
    return k


def extract_panels(fp):
    """打开源文件, 去黑边, 切拼图.

    返回 panel 列表, 每个元素为 dict:
      {"kind": "copy", "src": fp}          整图未动过, 可字节拷贝落盘
      {"kind": "jpeg", "image": PIL.Image}  裁切块, 需重编码落盘
    打不开的图片抛异常, 由调用方捕获并跳过.
    """
    with Image.open(fp) as im:
        raw = im.convert("RGB")
        im = crop_black_borders(raw)
        w, h = im.size
        verbatim = (w, h) == raw.size
        panels = grid_panels(im) or [(0, 0, w, h)]
        out = []
        for box in panels:
            if box == (0, 0, w, h) and verbatim:
                out.append({"kind": "copy", "src": fp})
            else:
                out.append({"kind": "jpeg", "image": im.crop(box)})
        return out


def panel_dhash(panel):
    if panel["kind"] == "copy":
        with Image.open(panel["src"]) as im:
            return dhash(im.convert("RGB"))
    return dhash(panel["image"])


def dedup_panels(panels, seen):
    """dhash 去重. 返回 (keep, n_dup), 新哈希记入 seen."""
    keep, n_dup = [], 0
    for p in panels:
        h = panel_dhash(p)
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        keep.append(p)
    return keep, n_dup


def write_panel_atomic(panel, dst):
    """O_EXCL 原子创建并写入; 目标已存在时抛 FileExistsError.

    copy 种直接拷贝源文件字节 (去重最稳), jpeg 种编码写入.
    写入失败时清理半成品文件, 不污染目录.
    """
    fd = os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "wb") as out:
            if panel["kind"] == "copy":
                with open(panel["src"], "rb") as src:
                    shutil.copyfileobj(src, out)
            else:
                panel["image"].save(out, format="JPEG", quality=92)
    except BaseException:
        try:
            os.unlink(dst)
        except OSError:
            pass
        raise
    return dst
