"""对话里的工具操作：文件读写（含越界审批）+ PowerShell 命令执行。

规则：
- 工作区 D:\\AI 内的文件操作直接执行；
- 越界文件操作、联网/装包/内嵌脚本等命令，走 sandbox.guard 的审批队列；
- 用户带 auto_approve=True（前端勾选「无需请求」）时直接执行，但危险命令仍拒绝。
"""

from __future__ import annotations

import ntpath
import re

from sandbox import files as sandbox_io
from sandbox import guard

FILE_WORD = re.compile(
    r"(文件|文档|文本|\.txt|\.md|\.json|\.py|\.csv|\.log|\.ps1|\.bat|\.ini|\.yaml|\.yml|\.toml|\.xml|\.html)",
    re.I,
)
CREATE_WORD = re.compile(r"(创建|新建|建立|写入|写一个|生成|保存|存成|往.{0,20}里写|写到|写进|写[:：])")
READ_WORD = re.compile(r"(读取|打开|查看|看看|读一下|显示|内容是什么)")
LIST_WORD = re.compile(r"(列出|有哪些|都有什么|目录)")
PATH_RE = re.compile(
    r"[A-Za-z]:\\[^\s，。；、\"'「」]+"
    r"|[\w\u4e00-\u9fa5.\-]+(?:[/\\][\w\u4e00-\u9fa5.\-]+)*\.(?:txt|md|json|py|csv|log|ps1|bat|ini|yaml|yml|toml|xml|html)",
    re.I,
)
CONTENT_RE = re.compile(r"(?:内容|写入|写上|里面写|写|保存为)[:：为]?\s*(.*)$", re.S)
COMMAND_RE = re.compile(
    r"^(?:请)?(?:帮我)?(?:执行|运行|跑一下|用\s*(?:powershell|cmd))[:：]?\s*(.+)$", re.I | re.S
)


def _absolute(raw: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", raw))


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _command_reply(result: dict) -> dict:
    status = result.get("status")
    if status == "denied":
        return {"thinking": "命令被安全策略拒绝", "reply": f"这条命令被拒绝：{result.get('reason')}。"}
    if status == "needs_approval":
        request = result["request"]
        return {
            "thinking": "命令需要授权",
            "reply": (
                f"需要你授权：`{request['command']}`\n\n"
                f"原因：{request['reason']}\n"
                f"请在「沙盒 → 待审批」里点批准（编号 `{request['id']}`），批准后命令会自动执行，"
                "输出显示在沙盒面板里。"
            ),
        }
    out = (result.get("stdout") or "").strip()
    err = (result.get("stderr") or "").strip()
    body = out if out else "（无输出）"
    reply = f"已执行（{result.get('elapsed')}s，退出码 {result.get('code')}）：\n\n```\n{body[-2000:]}\n```"
    if err:
        reply += f"\n\n错误输出：\n```\n{err[-800:]}\n```"
    return {"thinking": "执行命令", "reply": reply}


def handle(text: str, auto_approve: bool = False) -> dict | None:
    """命中工具操作就执行并返回回复；否则返回 None，走正常对话。"""
    if not text:
        return None

    # 1. PowerShell / 命令执行（以「执行/运行」开头时优先按命令处理）
    command_match = COMMAND_RE.search(text.strip())
    if command_match:
        return _command_reply(guard.execute(command_match.group(1).strip(), approved=auto_approve))

    # 2. 文件操作
    if not FILE_WORD.search(text):
        return None

    # 列表不需要路径，必须在 PATH_RE 之前判断
    if LIST_WORD.search(text) and not PATH_RE.search(text):
        items = sandbox_io.listing("")
        if not items:
            return {"thinking": "列出工作区", "reply": "工作区 D:\\AI 目前是空的。"}
        names = "\n".join(f"- {'📁' if item['dir'] else '📄'} {item['path']}" for item in items[:50])
        return {"thinking": "列出工作区", "reply": f"工作区 D:\\AI 共 {len(items)} 项：\n{names}"}

    path_match = PATH_RE.search(text)
    if not path_match:
        return None
    raw = path_match.group(0).strip().strip("「」\"'`")

    # 越界：转成命令行走审批，批准后由沙盒执行
    if _absolute(raw) and not raw.lower().startswith(str(guard.WORKSPACE).lower()):
        if CREATE_WORD.search(text):
            content_match = CONTENT_RE.search(text[path_match.end():])
            content = (content_match.group(1).strip().strip("「」\"'`") if content_match else "")
            folder = ntpath.dirname(raw)
            command = (
                f"New-Item -ItemType Directory -Force -Path {_quote(folder)} | Out-Null; "
                f"Set-Content -LiteralPath {_quote(raw)} -Value {_quote(content)} -Encoding UTF8"
            )
        elif READ_WORD.search(text):
            command = f"Get-Content -LiteralPath {_quote(raw)} -Encoding UTF8"
        else:
            return None
        return _command_reply(guard.execute(command, cwd=".", approved=auto_approve))

    # 工作区内
    relative = raw.replace("D:\\AI\\", "").replace("D:\\AI", "").lstrip("\\/")
    try:
        if READ_WORD.search(text) and not CREATE_WORD.search(text):
            data = sandbox_io.read_file(relative)
            body = data["content"].strip() or "（空文件）"
            return {
                "thinking": f"读取工作区文件 {relative}",
                "reply": f"已读取 `{relative}`，共 {len(data['content'])} 个字符：\n\n```\n{body[:2000]}\n```",
            }
        if CREATE_WORD.search(text):
            tail = text[path_match.end():]
            content_match = CONTENT_RE.search(tail)
            content = content_match.group(1).strip().strip("「」\"'`") if content_match else ""
            result = sandbox_io.write_file(relative, content)
            size = result["size"]
            if content:
                preview = content if len(content) <= 200 else content[:200] + "…"
                return {
                    "thinking": f"在工作区写入 {relative}（{size} 字节）",
                    "reply": f"已写入 `{relative}`（{size} 字节）：\n\n```\n{preview}\n```",
                }
            return {
                "thinking": f"在工作区创建空文件 {relative}",
                "reply": f"已创建空文件 `{relative}`。要写内容就说「往 {relative} 里写：……」",
            }
    except ValueError as exc:
        return {"thinking": "文件操作被拒绝", "reply": f"操作失败：{exc}"}
    except OSError as exc:
        return {"thinking": "文件操作出错", "reply": f"文件操作出错：{exc}"}
    return None
