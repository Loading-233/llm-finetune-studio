"""生成 Markdown 输出风格训练集：用户问法保持自然（不出现"用 Markdown/代码块"字样），
回答里覆盖标题、列表、表格、带语言标识的代码块、行内代码、粗体、链接、引用、分隔线、任务列表。

输出到 stdout 指定路径，默认 data/eval 之外的临时目录；导入用 /api/datasets/import。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASES: list[dict] = []


def add(user: str, assistant: str) -> None:
    BASES.append({"user": user.strip(), "assistant": assistant.strip("\n")})


# ---------------------------------------------------------------- 教程类（标题 + 有序步骤 + 引用）
TUTORIALS = [
    ("用 pandas 统计 CSV 里每列的空值数量", "pandas 空值统计"),
    ("怎么用 Git 把一个提交回退掉", "回退 Git 提交"),
    ("怎么在 Python 里读写 JSON 文件", "Python 读写 JSON"),
    ("本地跑一个 FastAPI 服务", "启动 FastAPI"),
    ("怎么给 SQL 查询加索引", "SQL 索引优化"),
    ("怎么用 PowerShell 批量重命名文件", "批量重命名文件"),
    ("怎么用 Docker 跑一个 Python 应用", "Docker 跑 Python"),
    ("怎么排查接口 500 错误", "排查接口 500"),
]
for question, title in TUTORIALS:
    add(
        question + "？",
        f"""## {title}

先给结论：按下面的顺序做，绝大多数情况能直接解决。**不要跳过第 2 步的验证**。

### 操作步骤

1. **确认环境和版本**：先跑 `--version`，排除版本不匹配。
2. **最小复现**：用一个最小输入跑通，再回到真实数据。
3. **改一处、验一处**：每次只改一个变量，避免问题叠加。

### 常见坑

- 路径里带空格时没有加引号
- 缓存没清理，改了代码却还是旧结果
- 权限不足，但报错信息看起来像"文件不存在"

> 提示：先把错误信息完整复制出来再搜索，前 20 行通常就有关键词。

---

如果按上面做完还是不行，把报错原文发我，我帮你继续定位。""",
    )


# ---------------------------------------------------------------- 代码类（带语言标识的代码块）
CODE_SAMPLES = [
    (
        "用 Python 写个读取 CSV 并统计每列空值数量的脚本",
        "统计 CSV 空值",
        "python",
        """import pandas as pd

def count_missing(path: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.isna().sum().sort_values(ascending=False)

if __name__ == "__main__":
    print(count_missing("data.csv"))""",
        ["`isna().sum()` 按列统计", "排序后先处理空值最多的列"],
    ),
    (
        "Python 怎么把列表里的重复元素去掉",
        "列表去重",
        "python",
        """def unique(items):
    # 保持原顺序：dict.fromkeys 比 set 更稳妥
    return list(dict.fromkeys(items))

print(unique([3, 1, 3, 2, 1]))  # [3, 1, 2]""",
        ["`set()` 不保序", "需要保序时用 `dict.fromkeys()`"],
    ),
    (
        "用 Python 写个重试装饰器",
        "重试装饰器",
        "python",
        """import time
import functools

def retry(times=3, delay=1.0):
    def wrapper(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            last = None
            for i in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last = exc
                    time.sleep(delay * (i + 1))
            raise last
        return inner
    return wrapper

@retry(times=3)
def fetch(url):
    ...""",
        ["指数退避能减少雪崩", "只捕获预期异常，别吞掉所有错误"],
    ),
    (
        "SQL 查出每个部门工资最高的员工",
        "部门最高工资",
        "sql",
        """SELECT d.name, e.name, e.salary
FROM employees e
JOIN departments d ON d.id = e.dept_id
WHERE (e.dept_id, e.salary) IN (
    SELECT dept_id, MAX(salary)
    FROM employees
    GROUP BY dept_id
);""",
        ["同薪并列会返回多人", "MySQL 8 也可以用窗口函数 `ROW_NUMBER()`"],
    ),
    (
        "SQL 怎么统计每天的活跃用户数",
        "每日活跃用户",
        "sql",
        """SELECT DATE(login_at) AS day,
       COUNT(DISTINCT user_id) AS dau
FROM user_events
WHERE login_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY day
ORDER BY day;""",
        ["`COUNT(DISTINCT ...)` 才是不重复人数", "按天分组前先把时间戳转日期"],
    ),
    (
        "JS 怎么防抖一个输入框搜索",
        "输入防抖",
        "javascript",
        """function debounce(fn, wait = 300) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), wait);
  };
}

const search = debounce((keyword) => {
  console.log('请求:', keyword);
}, 300);""",
        ["防抖是等安静下来再发", "节流是固定间隔发，别混用"],
    ),
    (
        "用 jQuery 不太好，原生 JS 怎么发 POST 请求",
        "原生 POST 请求",
        "javascript",
        """const res = await fetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question: '你好' }),
});

if (!res.ok) throw new Error(`HTTP ${res.status}`);
const data = await res.json();""",
        ["记得判断 `res.ok`", "异步函数里用 `await` 要包在 `try/catch` 中"],
    ),
    (
        "用 PowerShell 查最近三天修改过的文件",
        "筛选近期文件",
        "powershell",
        """$since = (Get-Date).AddDays(-3)
Get-ChildItem -Path D:\\AI -Recurse -File |
    Where-Object { $_.LastWriteTime -gt $since } |
    Sort-Object LastWriteTime -Descending |
    Select-Object FullName, LastWriteTime""",
        ["`-File` 可以排除目录", "路径含空格要加引号"],
    ),
    (
        "bash 怎么统计日志里出现次数最多的 IP",
        "统计高频 IP",
        "bash",
        """awk '{print $1}' access.log \\
  | sort \\
  | uniq -c \\
  | sort -rn \\
  | head -10""",
        ["`uniq` 前必须先 `sort`", "`head` 控制输出条数"],
    ),
    (
        "给我一个商品接口的 JSON 返回示例",
        "商品接口 JSON",
        "json",
        """{
  "code": 0,
  "data": {
    "id": 1024,
    "name": "星尘笔记会员",
    "price": 199,
    "tags": ["订阅", "年付"]
  },
  "message": "ok"
}""",
        ["字段命名统一用下划线或小驼峰", "`code` 与 HTTP 状态码分开表达"],
    ),
]
for question, title, lang, code, notes in CODE_SAMPLES:
    bullets = "\n".join(f"- {note}" for note in notes)
    add(
        question + "？" if not question.endswith("？") else question,
        f"""## {title}

直接可用的版本如下：

```{lang}
{code}
```

### 要点

{bullets}

> 这段代码可以原样跑，改输入路径即可。""",
    )


# ---------------------------------------------------------------- 对比类（表格）
COMPARES = [
    ("MySQL 和 PostgreSQL", "MySQL", "PostgreSQL", "并发读多选 MySQL，复杂查询和扩展选 PostgreSQL"),
    ("Git merge 和 rebase", "merge", "rebase", "公共分支用 merge，个人分支整理用 rebase"),
    ("列表和元组", "list", "tuple", "需要修改用 list，固定结构用 tuple"),
    ("同步和异步接口", "同步", "异步", "简单场景同步够用，高并发 IO 密集用异步"),
    ("SQLite 和 PostgreSQL", "SQLite", "PostgreSQL", "本地单机用 SQLite，多人并发用 PostgreSQL"),
    ("REST 和 GraphQL", "REST", "GraphQL", "资源清晰用 REST，字段按需组装用 GraphQL"),
]
for subject, left, right, advice in COMPARES:
    add(
        f"{subject}有什么区别？",
        f"""## {subject} 对比

结论：没有绝对更好，取决于你的场景。

| 维度 | {left} | {right} |
|---|---|---|
| 上手成本 | 低 | 中 |
| 灵活性 | 一般 | 高 |
| 性能 | 视场景 | 视场景 |
| 生态 | 成熟 | 成熟 |

### 怎么选

1. **先看团队熟悉度**：能用对的方案才是好方案。
2. **再看瓶颈位置**：瓶颈在读写比还是查询复杂度。
3. **最后看维护成本**：长期维护比一次性上线更重要。

> 我的建议：{advice}。""",
    )


# ---------------------------------------------------------------- 清单类（任务列表）
CHECKLISTS = [
    ("上线前需要检查什么", "上线检查清单", ["接口回归通过", "数据库备份完成", "监控告警已配置", "回滚脚本可用", "值班人确认"]),
    ("新项目第一天要准备什么", "新项目启动清单", ["建仓库和分支规范", "配置 CI", "拉开发环境", "约定提交信息格式", "建需求看板"]),
    ("写周报前要整理哪些内容", "周报准备清单", ["本周完成项", "未完成原因", "下周计划", "风险与求助"]),
]
for question, title, items in CHECKLISTS:
    lines = "\n".join(f"- [ ] {item}" for item in items)
    add(
        question + "？",
        f"""## {title}

按顺序过一遍，全部打勾再往下走。

{lines}

### 说明

- 第 1 项是**硬门槛**，没过不要继续。
- 任何一项有风险，先记录到风险清单，不要口头约定。

> 建议把这份清单放进仓库的 `docs/checklist.md`，每次上线复用。""",
    )


# ---------------------------------------------------------------- 排错类
TROUBLES = [
    ("Python 报 ModuleNotFoundError: No module named 'requests'", "模块找不到"),
    ("端口 8000 被占用怎么办", "端口被占用"),
    ("Git push 报 rejected，提示 non-fast-forward", "推送被拒绝"),
    ("pandas 读 CSV 中文乱码", "CSV 中文乱码"),
]
for question, title in TROUBLES:
    add(
        question if question.endswith("？") else question + "，怎么解决？",
        f"""## {title}

**原因**：环境或状态和预期不一致，通常是版本、路径或占用问题。

### 先验证

```bash
python -c "import sys; print(sys.executable)"
```

### 再修复

1. 确认当前解释器是不是你以为的那个。
2. 重新安装依赖，观察是否仍有报错。
3. 清理缓存后重试。

> 如果三步都没解决，把完整报错贴出来。""",
    )


# ---------------------------------------------------------------- 周报 / 文档类（多级标题 + 分隔线 + 链接）
add(
    "帮我写一份这周的开发周报",
    """## 本周开发周报

### 一、完成情况

- 接口联调完成，修复 **12 个缺陷**
- 上线灰度环境，核心链路通过 `pytest` 回归
- 补齐监控指标，见 [监控面板](https://example.com/dashboard)

### 二、未完成

1. 批量导入性能未达标，当前 3 万条/分钟
2. 移动端适配延后到下周

### 三、下周计划

- 优化导入批大小与索引
- 完成移动端断点适配

---

### 风险

> 导入优化依赖数据库扩容，需要运维同学提前准备。""",
)
add(
    "把这段会议记录整理成纪要",
    """## 会议纪要

**时间**：周三 14:00　**参与**：产品、后端、前端

### 结论

1. 本期范围锁定为**导入优化**和**权限修复**。
2. 移动端适配顺延，不作为本周验收项。

### 待办

- [ ] 后端：拆分导入任务，周四前给方案
- [ ] 前端：权限按钮置灰逻辑
- [ ] 产品：补充验收用例

> 下次同步：周五 10:00。""",
)


# ---------------------------------------------------------------- 邮件 / 文案类
add(
    "写一封项目延期的道歉邮件",
    """## 关于项目延期的说明

尊敬的客户：

您好。由于**接口联调**阶段发现兼容性问题，原定本周五的交付需要延期三个工作日。

### 我们的处理

1. 已定位问题并完成修复方案评审
2. 增派一名后端同学并行处理
3. 每日同步一次进度

> 给您带来的不便我们深表歉意，新的交付时间初步定在下周三。

如需提前了解细节，可随时联系我。

此致
敬礼""",
)


# ---------------------------------------------------------------- 数据结论类
add(
    "这组留存数据能得出什么结论：次日 42%，7 日 21%，30 日 9%",
    """## 留存分析结论

| 周期 | 留存率 | 判断 |
|---|---|---|
| 次日 | 42% | 尚可 |
| 7 日 | 21% | 偏低 |
| 30 日 | 9% | 明显偏低 |

### 结论

1. **首日流失严重**：超过一半用户第二天不再回来。
2. **7 日到 30 日衰减趋缓**，说明留下的用户粘性还可以。
3. 优先优化**首次使用路径**，而不是继续拉新。

> 建议先做新手引导的 A/B 实验，观察次日留存能否提升 3 个百分点。""",
)

add(
    "帮我算一下 AB 实验需要多大样本量",
    """## 样本量估算

结论：按基线转化率 **10%**、最小可检测提升 **1 个百分点**、显著性 `α=0.05`、功效 `80%` 计算，每组约需 **1.6 万** 用户。

### 关键参数

- 基线转化率：`p1 = 0.10`
- 目标转化率：`p2 = 0.11`
- 显著性水平：`α = 0.05`
- 统计功效：`1 - β = 0.80`

### 注意

- 样本量按**每组**计算，总量要乘 2。
- 实验周期至少覆盖一个完整周，避免星期效应。

> 如果流量不够，就降低提升预期，别缩短周期。""",
)


def build(out: Path, merge: str = "") -> list[dict]:
    """每个基础样本配多种自然问法变体；可选合并一个已有数据集。"""
    prefixes = ["", "请问", "我想知道，", "麻烦问下，", "帮我看看，"]
    records: list[dict] = []
    seen: set[str] = set()
    for base in BASES:
        for prefix in prefixes:
            user = f"{prefix}{base['user']}" if prefix else base["user"]
            if user in seen:
                continue
            seen.add(user)
            records.append(
                {
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": base["assistant"]},
                    ]
                }
            )
    if merge:
        merge_path = Path(merge)
        if not merge_path.is_absolute():
            merge_path = ROOT / merge_path
        for line in merge_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    random.Random(42).shuffle(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "outputs" / "md_style_v1.jsonl"))
    parser.add_argument("--merge", default="", help="追加合并一个已有数据集文件（相对项目根）")
    args = parser.parse_args()
    out = Path(args.out)
    records = build(out, args.merge)
    print(f"基础样本 {len(BASES)} 条，合并后 {len(records)} 条 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
