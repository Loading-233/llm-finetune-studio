"""任务定义与模板渲染。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

TASKS_PATH = Path(__file__).resolve().parent / "tasks.yaml"


@dataclass
class Variable:
    name: str
    label: str
    default: str = ""


@dataclass
class ForgeTask:
    id: str
    name: str
    description: str
    system: str
    template: str
    output_format: str = "text"
    temperature: float = 0.5
    max_tokens: int = 600
    input_hint: str = ""
    retrieval: bool = False
    variables: list[Variable] = field(default_factory=list)

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "output_format": self.output_format,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "input_hint": self.input_hint,
            "retrieval": self.retrieval,
            "variables": [{"name": v.name, "label": v.label, "default": v.default} for v in self.variables],
        }


def load_tasks(path: Path | None = None) -> dict[str, ForgeTask]:
    payload = yaml.safe_load((path or TASKS_PATH).read_text(encoding="utf-8")) or {}
    tasks: dict[str, ForgeTask] = {}
    for item in payload.get("tasks", []):
        task = ForgeTask(
            id=item["id"],
            name=item.get("name", item["id"]),
            description=item.get("description", ""),
            system=item.get("system", "").strip(),
            template=item.get("template", "{{input}}"),
            output_format=item.get("output_format", "text"),
            temperature=float(item.get("temperature", 0.5)),
            max_tokens=int(item.get("max_tokens", 600)),
            input_hint=item.get("input_hint", ""),
            retrieval=bool(item.get("retrieval", False)),
            variables=[
                Variable(name=v["name"], label=v.get("label", v["name"]), default=str(v.get("default", "")))
                for v in (item.get("variables") or [])
            ],
        )
        tasks[task.id] = task
    return tasks


def render(task: ForgeTask, text: str, variables: dict[str, str] | None = None, context: str = "") -> str:
    values = {variable.name: variable.default for variable in task.variables}
    values["context"] = context or "（无）"
    for key, value in (variables or {}).items():
        if key in values and str(value).strip():
            values[key] = str(value).strip()
    prompt = task.template.replace("{{input}}", text.strip())
    for key, value in values.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", value)
    return prompt.strip()
