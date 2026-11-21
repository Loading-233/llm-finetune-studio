"""8765 入口网关：用口令控制「数据集 / 训练 / 适配器与评测」三个模块。

用法：
- 首次访问 http://<本机IP>:8765/?key=<口令>，服务器写入 Cookie 后自动跳回正常页面
- 之后这台设备一直算管理员；其他设备不给口令就只看到「对话 / 文本工坊 / 沙盒」
- 直接调用受保护接口而没有 Cookie 的，返回 403
- 口令存放在 scripts/admin_key.txt，改完重启服务生效
"""
from __future__ import annotations

import logging
import secrets
import sys
from pathlib import Path
from urllib.parse import urlencode

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

KEY_FILE = ROOT / "scripts" / "admin_key.txt"
COOKIE = "cast_key"
COOKIE_AGE = 60 * 60 * 24 * 365

# 受保护的接口前缀：数据集、训练任务、评测、模型下载与释放
GUARDED = (
    "/api/datasets",
    "/api/train",
    "/api/jobs",
    "/api/eval",
    "/api/evals",
    "/api/models/download",
    "/api/runtime/unload",
)

# 适配器列表是对话页要用的只读信息：允许读，禁止删/改
GUARDED_WRITE = ("/api/adapters",)

log = logging.getLogger("cast.gateway")
cache: tuple[float, str] = (-1.0, "")


def admin_key() -> str:
    """读取口令；文件改动后自动生效。"""
    global cache
    try:
        mtime = KEY_FILE.stat().st_mtime
    except OSError:
        return ""
    if mtime != cache[0]:
        try:
            cache = (mtime, KEY_FILE.read_text(encoding="utf-8").strip())
        except OSError as exc:
            log.warning("读取口令失败（%s）", exc)
            return cache[1]
    return cache[1]


def is_admin(request: Request) -> bool:
    key = admin_key()
    if not key:
        # 没配置口令时不设限制，避免把主人锁在门外
        return True
    token = request.cookies.get(COOKIE, "")
    return bool(token) and secrets.compare_digest(token, key)


class AccessGuard(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        key = admin_key()
        given = request.query_params.get("key")
        if key and given and secrets.compare_digest(given, key):
            # 口令正确：写入 Cookie，去掉地址里的 key 再跳回去
            params = [(k, v) for k, v in request.query_params.multi_items() if k != "key"]
            target = request.url.path + (f"?{urlencode(params)}" if params else "")
            response = RedirectResponse(target or "/", status_code=303)
            response.set_cookie(COOKIE, key, max_age=COOKIE_AGE, samesite="lax")
            return response

        if not is_admin(request):
            path = request.url.path
            blocked = any(path.startswith(prefix) for prefix in GUARDED)
            if not blocked and request.method != "GET":
                blocked = any(path.startswith(prefix) for prefix in GUARDED_WRITE)
            if blocked:
                client = request.client.host if request.client else "?"
                log.warning("拒绝 %s %s（来自 %s）", request.method, path, client)
                return JSONResponse({"detail": "该模块仅限本人设备使用"}, status_code=403)
        return await call_next(request)


app.add_middleware(AccessGuard)


@app.get("/api/access")
def access(request: Request):
    client = request.client.host if request.client else ""
    return {"admin": is_admin(request), "ip": client}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
