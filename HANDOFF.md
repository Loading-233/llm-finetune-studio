# 交接说明（HANDOFF）

更新：2026-11-18　维护者：当前 agent

## 当前默认状态

| 项 | 值 |
|---|---|
| 基座模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| 推理后端 | llama.cpp GGUF Q4_K_M（`models/gguf/qwen2.5-1.5b-md2-q4_k_m.gguf`，940MB） |
| 已合并适配器 | md2 版本（由对端训练导出，adapter id 待确认；早前记录：`814a489174db` = 474 条格式+知识） |
| 线程 | 8（`config.yaml` → `inference.threads`） |
| 服务地址 | http://127.0.0.1:8765 |

切回 PyTorch：把 `config.yaml` 的 `inference.backend` 改成 `torch` 后重启。

启动命令（必须用项目 `.venv`）：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --app-dir .
```

## 目录归属（协作约定）

- 当前 agent：`app/`、`core/`、`forge/`、`config.yaml`、`models/`、`outputs/`、`data/studio.db`
- 对端 agent：`scripts/`（只新增，不改已有）、`data/eval/`、`web/`（界面改版；`app.js` 里的 DOM 钩子如需改动，先在本文件登记）
- 双方都不重启 8765 服务；需要重启先留言
- 协作信箱：`collab/from_trae.md`（当前 agent 发）、`collab/from_peer.md`（对端 agent 发），只追加不覆盖

## 协作信箱规则

1. 每次沟通在各自文件末尾追加一条，格式：`## YYYY-MM-DD HH:mm` + 正文；
2. 用户只需说一句「收信」，双方各读对方的文件，不必复制内容；
3. 涉及文件归属、服务重启、接口字段变更时，必须先在这里留记录再动手；
4. 冲突时以「先登记者优先」为准。

## 已知限制

1. GGUF 模式下评测困惑度显示 `—`（llama.cpp 不提供 ppl），只剩关键词命中率。
2. GGUF 模式下切换适配器无效：会推送一条 `thinking` 警告「本次忽略所选适配器」。换适配器需重新：合并 LoRA → 转 GGUF → 量化。
3. GGUF 模式下 `use_adapter=false` 无法关闭微调（权重已合并）。
4. `127.0.0.1:8766` 是用户 PyCharm 里的旧代码副本，不要当作基准。
5. 3B 权重已下载但未训练；16GB 内存下 3B fp32 约 12GB，很吃紧。
6. 评测与线上口径不同，见下节。
7. llama.cpp 多线程下 `temperature=0` 不是严格确定（浮点归约顺序差异），单跑一轮 24 题可能 ±1 题；关键题建议重复 3 次取多数。
8. 实时信息（天气/新闻/股价等）无法获取；对话已加硬约束强制拒答，但模型仍可能偶发编造，评测时应单独看这类题。

## 评测口径（重要）

线上 `/api/chat` 实际拼接：

1. system = persona + 详细度提示；
2. 检索 top2（语义分 ≥0.60；关键词兜底覆盖率 ≥0.80），拼进 system；
3. 多轮历史上限 12 条；
4. 可选思考：4 段中文、上限 360 token，并把分析要点回填 system 后再答；
5. 回答默认 `max_new_tokens=768`、`temperature=0.7`。

旧 `scripts/eval_bench.py`：persona + 检索 top2 + 220 token + 贪心、无思考、直接调 runtime → 与线上不可比。

当前基线（GGUF、8 线程、`show_thought=true`、`max_new_tokens=768`、`verbosity=detail`、检索开）：单题约 45.8 秒（墙钟）。

## 服务进程注意

8765 现在由 `tools/serve_8765.ps1` 看门狗托管：进程退出 3 秒后自动重启，日志在 `outputs/serve_8765.log`。
手动重启：结束 8765 的监听进程即可，看门狗会用新代码拉起。

**已知崩溃原因（2026-11-18 修复）**：llama.cpp 的 context 不是线程安全的，两个请求同时生成会在 `ggml-cpu.dll` 里触发 `0xc0000005` 访问冲突，进程直接退出。现已在 `core/llamacpp.py` 用 `_GEN_LOCK` 串行化生成，并发请求会排队而不是崩溃。

## SSE 事件与计时口径

`/api/chat` 依次推送：`thinking`（含可选的 `hits`）→ `status` → `ready` → `thought`（思考增量）→ `delta`（回答增量）→ `done` → 结束。

`done` 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `elapsed` | 字符串 | **只含最终回答段**的生成秒数（不含思考、检索、建连） |
| `thought_elapsed` | 数字 | 思考段秒数，未开思考时为 0 |
| `total` | 数字 | 整轮墙钟秒数，从请求进入生成器开始 |

做评测时如果要“用户体验耗时”，用 `total`；要“模型生成速度”，用 `elapsed`。

## 服务进程注意

8765 必须用独立进程启动（本项目用 WMI `Win32_Process.Create` 拉起）。用 agent 的后台 shell 启动时，会在该终端被复用时一起被杀掉。


## 验证方式

```powershell
.\.venv\Scripts\python.exe scripts\chat_probe.py "读书笔记怎么记" e58236ad9711
.\.venv\Scripts\python.exe scripts\eval_bench.py --adapter e58236ad9711 --limit 5
```

- `/api/system` → `runtime.backend` 应为 `llamacpp`，`device.threads` 应为 8
- `/api/runtime/unload` 可释放内存（训练前会用到）

## 变更记录

- 2026-11-18：默认模型切 1.5B；检索阈值收紧到 0.60/0.80；接入 GGUF 后端；修复 `forge/semantic.py` 把全局线程压到 4 的副作用；GGUF 下切换适配器改为显式提示。
- 2026-11-18 19:0x（针对 24 题基线失败项）：
  - `done` 增加 `thought_elapsed` / `total` 计时字段；
  - 指令型任务（翻译/改写/润色/格式转换）跳过知识检索，避免同类例句污染；
  - 参考资料改为拼接在 system **末尾**（思考要点之后），修掉思考稀释资料导致的准确率下降；
  - `data/demo/domains/data.jsonl` 补 2 条留存率口径条目（总文档 235 条）。
  - 回归结果：d04（留存率）PASS、t01（翻译）PASS、p02（思考开启+detail）PASS。
- 2026-11-18 19:4x：
  - 思考段 `temperature` 0.5 → 0，消除同题多次运行的随机性；
  - 参考资料块加硬约束「事实以资料为准，资料没提到的不要臆测」；
  - `looks_like_instruction()` 增加术语问法排除（「怎么翻译 / 英文怎么说」不跳过检索）；
  - 稳定复测：p02 思考组连续 3 次全部命中关键事实，思考均 225 字，13.3–16.8 秒。
- 2026-11-18 20:5x：修复并发对话导致 `ggml-cpu.dll` 崩溃（`0xc0000005`）的问题——llama.cpp context 非线程安全，已在 `core/llamacpp.py` 加生成锁串行化；并发 3 请求验证全部成功、进程存活。服务改由 `tools/serve_8765.ps1` 看门狗托管。
- 2026-11-19：对话支持文件操作意图（新增 `app/chat_actions.py`）——「创建/新建 xxx.txt」「往 xxx 里写：…」「读取 xxx.md」「列出文件」直接操作沙盒 `D:\AI`，不再让模型教用户自己做；思考提示词去掉固定四段模板，改为按问题类型自由组织、≤120 字。
- 2026-11-21：接入公开数据训练。下载 `BelleGroup/train_0.5M_CN`（272.8MB），抽样 600 条 + `md_real_355` 混合成数据集 `f9e1580b31f4`（955 条），训练任务 `6bb24cca7d4a`（350 步 / 1 epoch / seq 512）。另发现 8765 由 `scripts/gateway.py` 口令网关启动，受保护接口需带管理员 Cookie。
- 2026-11-21 14:30：`6bb24cca7d4a` 训练完成（eval_loss 1.3631，adapter `6b61761d3535`），导出 `models/gguf/qwen2.5-1.5b-belle-q4_k_m.gguf`（940MB）。按用户要求**暂不切换线上**，等对端 r2 vs belle 的 24 题/格式评测结果。当前线上仍是 `qwen2.5-1.5b-r2-q4_k_m.gguf`。


