"""命令行对话探测：向本地服务发一次对话请求，打印最终回答。"""

from __future__ import annotations

import json
import sys
import urllib.request

URL = "http://127.0.0.1:8765/api/chat"


def ask(question: str, adapter_id: str = "", timeout: int = 300, verbosity: str = "detail") -> str:
    payload = json.dumps(
        {
            "adapter_id": adapter_id,
            "verbosity": verbosity,
            "max_new_tokens": 640,
            "messages": [{"role": "user", "content": question}],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    answer = []
    thoughts = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if event.get("type") == "delta":
                answer.append(event.get("text", ""))
            elif event.get("type") == "thought":
                thoughts.append(event.get("text", ""))
            elif event.get("type") == "thinking":
                print(f"  · {event.get('text')}")
                for hit in event.get("hits") or []:
                    print(f"    ↳ {hit.get('question')}")
            elif event.get("type") == "error":
                answer.append(f"[错误] {event.get('text') or event.get('message')}")
    if thoughts:
        print("【思考过程】")
        print("".join(thoughts).strip())
        print("【最终回答】")
    return "".join(answer).strip() or "（无输出）"


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "你好"
    adapter = sys.argv[2] if len(sys.argv) > 2 else ""
    print(ask(question, adapter))
