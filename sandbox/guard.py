"""沙盒：限定工作区内的命令执行与文件操作，越界需要人工授权。"""

from __future__ import annotations

import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from app import config

SANDBOX = config.SANDBOX
WORKSPACE = Path(SANDBOX.get("workspace", "D:/AI")).resolve()
TIMEOUT = int(SANDBOX.get("timeout", 60))

# 危险命令：即使授权也直接拒绝
DENY_PATTERNS = [
    r"\bformat\s+[a-z]:",
    r"\bdiskpart\b",
    r"\bbcdedit\b",
    r"\bshutdown\b",
    r"\breg\s+(add|delete)\b",
    r"\bnet\s+user\b",
    r"\btakeown\b",
    r"\bcipher\s+/w\b",
    r"\bdel\s+/[fsq].*[a-z]:\\",
    r"rd\s+/s\s+/q\s+[a-z]:\\?$",
    r"rm\s+-rf\s+/(\s|$)",
    r"\bmkfs\b",
    r":\(\)\s*\{.*\};:",
]

# 需要审批的情况：写到工作区外、访问系统目录、联网下载
OUTSIDE_HINTS = [
    (re.compile(r"\b(curl|wget|Invoke-WebRequest|iwr)\b", re.I), "疑似联网下载"),
    (re.compile(r"\b(git\s+push|pip\s+install|npm\s+install|winget\s+install)\b", re.I), "会改动外部环境"),
    (re.compile(r"\b(python|powershell|cmd|bash)\b.*\s(-c|/c)\b", re.I), "内嵌脚本执行"),
]


@dataclass
class PendingRequest:
    id: str
    command: str
    cwd: str
    reason: str
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


PENDING: dict[str, PendingRequest] = {}


def ensure_workspace() -> Path:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    return WORKSPACE


def resolve_in_workspace(relative: str) -> Path:
    """把相对路径解析到工作区内，越界抛 ValueError。"""
    ensure_workspace()
    candidate = (WORKSPACE / relative).resolve() if not Path(relative).is_absolute() else Path(relative).resolve()
    if candidate != WORKSPACE and WORKSPACE not in candidate.parents:
        raise ValueError(f"路径不在工作区内：{relative}")
    return candidate


def _outside_paths(command: str) -> list[str]:
    """找出命令里指向工作区之外的绝对路径。"""
    found = []
    for match in re.finditer(r"[A-Za-z]:[\\/][^\s\"']*", command):
        raw = match.group(0).rstrip("\\/")
        try:
            path = Path(raw).resolve()
        except OSError:
            continue
        if path != WORKSPACE and WORKSPACE not in path.parents:
            found.append(raw)
    return found


def classify(command: str, cwd: str | None = None) -> tuple[str, str]:
    """返回 (allow | approval | deny, 原因)。"""
    text = command.strip()
    if not text:
        return "deny", "命令为空"

    for pattern in DENY_PATTERNS:
        if re.search(pattern, text, re.I):
            return "deny", "命中危险命令规则"

    outside = _outside_paths(text)
    if outside:
        return "approval", f"访问工作区外路径：{outside[0]}"

    for pattern, reason in OUTSIDE_HINTS:
        if pattern.search(text):
            return "approval", reason

    if cwd:
        try:
            resolve_in_workspace(cwd)
        except ValueError:
            return "approval", f"工作目录在工作区外：{cwd}"

    return "allow", "工作区内操作"


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _run(command: str, cwd: Path, timeout: int) -> dict:
    started = time.time()
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=str(cwd),
            capture_output=True,
            timeout=timeout,
        )
        return {
            "ok": completed.returncode == 0,
            "code": completed.returncode,
            "stdout": _decode(completed.stdout or b"")[-8000:],
            "stderr": _decode(completed.stderr or b"")[-4000:],
            "elapsed": round(time.time() - started, 2),
            "command": command,
            "cwd": str(cwd),
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "code": -1,
            "stdout": "",
            "stderr": f"执行超时（{timeout}s）",
            "elapsed": round(time.time() - started, 2),
            "command": command,
            "cwd": str(cwd),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "code": -1,
            "stdout": "",
            "stderr": str(exc),
            "elapsed": round(time.time() - started, 2),
            "command": command,
            "cwd": str(cwd),
        }


def execute(command: str, cwd: str | None = None, timeout: int | None = None, approved: bool = False) -> dict:
    """执行命令。未授权且被判为越界时返回待审批请求。"""
    verdict, reason = classify(command, cwd)
    if verdict == "deny":
        return {"status": "denied", "reason": reason, "command": command}

    if verdict == "approval" and not approved:
        request = PendingRequest(id=uuid.uuid4().hex[:8], command=command, cwd=cwd or ".", reason=reason)
        PENDING[request.id] = request
        return {
            "status": "needs_approval",
            "request": {
                "id": request.id,
                "command": request.command,
                "cwd": request.cwd,
                "reason": request.reason,
                "created_at": request.created_at,
            },
        }

    workdir = WORKSPACE if not cwd else resolve_in_workspace(cwd)
    result = _run(command, workdir, timeout or TIMEOUT)
    result["status"] = "executed"
    result["approved"] = approved
    return result


def approve(request_id: str) -> dict:
    request = PENDING.pop(request_id, None)
    if request is None:
        return {"status": "missing", "reason": "请求不存在或已处理"}
    return execute(request.command, request.cwd, approved=True)


def deny(request_id: str) -> dict:
    request = PENDING.pop(request_id, None)
    if request is None:
        return {"status": "missing", "reason": "请求不存在或已处理"}
    return {"status": "denied", "request_id": request_id, "command": request.command}


def pending_list() -> list[dict]:
    return [
        {"id": item.id, "command": item.command, "cwd": item.cwd, "reason": item.reason, "created_at": item.created_at}
        for item in PENDING.values()
    ]


def info() -> dict:
    return {
        "workspace": str(WORKSPACE),
        "timeout": TIMEOUT,
        "deny_rules": len(DENY_PATTERNS),
        "note": "策略级沙盒：命令在工作区内执行，越界或敏感操作需人工批准；危险命令直接拒绝。",
    }
