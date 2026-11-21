"""沙盒文件操作：仅限工作区 D:\\AI 内。"""

from __future__ import annotations

from pathlib import Path

from sandbox.guard import WORKSPACE, ensure_workspace, resolve_in_workspace

MAX_READ = 200_000


def listing(relative: str = "") -> list[dict]:
    base = resolve_in_workspace(relative) if relative else ensure_workspace()
    if not base.exists():
        return []
    items = []
    try:
        children = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as exc:
        raise ValueError(f"无法读取目录：{exc}") from exc
    for child in children[:500]:
        try:
            stat = child.stat()
        except OSError:
            continue
        items.append(
            {
                "name": child.name,
                "path": str(child.relative_to(WORKSPACE)).replace("\\", "/"),
                "dir": child.is_dir(),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        )
    return items


def read_file(relative: str) -> dict:
    path = resolve_in_workspace(relative)
    if not path.is_file():
        raise ValueError("文件不存在")
    if path.stat().st_size > MAX_READ:
        raise ValueError("文件过大，暂不支持在线编辑")
    return {"path": relative, "content": path.read_text(encoding="utf-8", errors="replace")}


def write_file(relative: str, content: str) -> dict:
    path = resolve_in_workspace(relative)
    if path.is_dir():
        raise ValueError("目标是一个目录")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": relative, "size": len(content.encode("utf-8"))}


def delete_path(relative: str) -> dict:
    path = resolve_in_workspace(relative)
    if path == WORKSPACE:
        raise ValueError("不能删除工作区根目录")
    if path.is_dir():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
        path.rmdir()
    elif path.exists():
        path.unlink()
    return {"path": relative, "deleted": True}


def search(keyword: str, limit: int = 50) -> list[dict]:
    hits = []
    for path in ensure_workspace().rglob("*"):
        if not path.is_file() or path.stat().st_size > MAX_READ:
            continue
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if keyword in line:
                    hits.append(
                        {
                            "path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
                            "line": number,
                            "text": line.strip()[:200],
                        }
                    )
                    if len(hits) >= limit:
                        return hits
        except OSError:
            continue
    return hits
