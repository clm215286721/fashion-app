"""WordPress 会员鉴权模块.

对接 hhyms.com (WordPress + 子比主题) 的用户体系:
- 用户名/密码验真: 调 WP REST API /wp/v2/users/me (Application Password)
- VIP/等级: 读 usermeta (键名可配, 子比默认见下)
- 配额: 按等级/VIP 映射, 每日重置, SQLite 本地记次

子比 usermeta 键 (如与实际不符, 改环境变量):
- ZIBLL_VIP_KEY: vip 等级, 0=无 1=黄金 2=钻石 (默认 'zibll_vip_level')
- ZIBLL_VIP_EXPIRE_KEY: vip 到期时间戳 (默认 'zibll_vip_expire')
- ZIBLL_LEVEL_KEY: 用户等级 Lv1-10 (默认 'zibll_user_level')
- ZIBLL_MERCHANT_KEY: 商家标记 (默认 'fashion_merchant', 手动在后台给商家用户打标记)
"""
import base64
import contextlib
import json
import os
import sqlite3
import time
import urllib.request
import urllib.error

WP_BASE = os.environ.get("WP_BASE", "https://www.hhyms.com")
VIP_KEY = os.environ.get("ZIBLL_VIP_KEY", "zibll_vip_level")
VIP_EXPIRE_KEY = os.environ.get("ZIBLL_VIP_EXPIRE_KEY", "zibll_vip_expire")
LEVEL_KEY = os.environ.get("ZIBLL_LEVEL_KEY", "zibll_user_level")
MERCHANT_KEY = os.environ.get("ZIBLL_MERCHANT_KEY", "fashion_merchant")

BASE = os.path.dirname(os.path.abspath(__file__))
QUOTA_DB = os.path.join(BASE, "index", "quota.db")

# 配额映射: (日限次, 描述). -1 = 不限
QUOTA_MAP = {
    "lv1_3":   (10,  "Lv1-Lv3"),
    "lv4_6":   (50,  "Lv4-Lv6"),
    "lv7_10":  (200, "Lv7-Lv10"),
    "gold":    (500, "黄金会员"),
    "diamond": (-1,  "钻石会员"),
    "merchant":(-1,  "商家"),
}


def _wp_get(path, username, app_password, timeout=10):
    """调 WP REST API, 返回 dict. 失败抛 RuntimeError."""
    url = WP_BASE.rstrip("/") + path
    cred = base64.b64encode(f"{username}:{app_password}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {cred}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError("WP 账号或应用密码错误")
        raise RuntimeError(f"WP API 错误: HTTP {e.code}")
    except Exception as e:
        raise RuntimeError(f"WP 连接失败: {e}")


def get_wp_user(username, app_password):
    """验真并取用户信息. 返回 (wp_id, meta_dict).

    用 context=edit 确保能读到 usermeta (VIP/等级/商家标记).
    Application Password 认证的用户对自己的数据有 edit 权限.
    """
    me = _wp_get("/wp-json/wp/v2/users/me?context=edit", username, app_password)
    if not isinstance(me, dict):
        raise RuntimeError("WP API 返回格式异常")
    meta = me.get("meta", {}) or {}
    return me.get("id"), meta


def resolve_tier(meta):
    """按 usermeta 判定档位. 返回 tier key (见 QUOTA_MAP)."""
    now = int(time.time())
    # 商家标记优先
    if str(meta.get(MERCHANT_KEY, "")) in ("1", "true"):
        vip = str(meta.get(VIP_KEY, "0"))
        # 商家须是钻石 (按需求), 否则按普通档
        if vip == "2":
            return "merchant"
    vip = str(meta.get(VIP_KEY, "0"))
    expire = int(meta.get(VIP_EXPIRE_KEY, 0) or 0)
    if vip in ("1", "2") and expire > now:
        return "diamond" if vip == "2" else "gold"
    # 按等级
    try:
        lv = int(meta.get(LEVEL_KEY, 1) or 1)
    except (ValueError, TypeError):
        lv = 1
    if lv >= 7:
        return "lv7_10"
    if lv >= 4:
        return "lv4_6"
    return "lv1_3"


def _quota_conn():
    os.makedirs(os.path.dirname(QUOTA_DB), exist_ok=True)
    con = sqlite3.connect(QUOTA_DB, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS usage (
        wp_id INTEGER, day TEXT, count INTEGER DEFAULT 0,
        PRIMARY KEY (wp_id, day))""")
    con.commit()
    return contextlib.closing(con)


def check_and_consume(wp_id, tier):
    """查配额并扣一次. 返回 (ok, remaining, msg).

    原子操作: 单条 INSERT...ON CONFLICT, 避免 SELECT+UPDATE 竞态.
    """
    limit, desc = QUOTA_MAP[tier]
    if limit < 0:
        return True, -1, f"{desc}不限次"
    day = time.strftime("%Y-%m-%d")
    with _quota_conn() as con:
        c = con.cursor()
        # 原子递增, 返回更新后的值
        c.execute(
            "INSERT INTO usage (wp_id, day, count) VALUES (?,?,1) "
            "ON CONFLICT (wp_id, day) DO UPDATE SET count=count+1 "
            "RETURNING count",
            (wp_id, day))
        used = c.fetchone()[0]
        con.commit()
    if used > limit:
        # 超了, 回滚这一次 (减回去)
        with _quota_conn() as con:
            con.execute(
                "UPDATE usage SET count=count-1 WHERE wp_id=? AND day=?",
                (wp_id, day))
            con.commit()
        return False, 0, f"今日搜图次数已用完 ({desc}: {limit}/天), 明日重置或升级会员"
    return True, limit - used, f"{desc}今日剩余 {limit - used} 次"


def gradio_auth(username, password):
    """给 gr.launch(auth=...) 用的认证函数. 返回 True/False.

    注意: 这里 password 是 WP 应用密码 (不是 WP 登录密码),
    用户在 WP 后台 用户→个人资料→应用密码 生成.
    """
    try:
        wp_id, meta = get_wp_user(username, password)
    except RuntimeError:
        return False
    # 认证通过即可进; 配额在搜图时扣
    return True


def get_user_tier(username, app_password):
    """取用户档位 (搜图时用). 返回 (wp_id, tier, desc)."""
    wp_id, meta = get_wp_user(username, app_password)
    tier = resolve_tier(meta)
    return wp_id, tier, QUOTA_MAP[tier][1]
