# 铸语 · 本地大模型微调工作台

一个可以在普通 Windows 电脑（CPU、无显卡）上跑通「数据集 → LoRA 微调 → 评测 → 对话/工具调用」全流程的本地工作台。默认基座是 **Qwen2.5-1.5B-Instruct**，推理使用 **llama.cpp + GGUF 量化**，7×24 常驻本机，数据不出网。

## 功能

- **数据集管理**：导入 JSONL / Alpaca / ShareGPT，预览样本，查看统计
- **LoRA 训练**：PEFT + Transformers Trainer，CPU 可训；支持断点续训、按参数指纹自动匹配历史 checkpoint
- **评测体系**：24 题能力基线 + 8 题 Markdown 格式 + 5 题工具调用（action），可重复跑、可导出报告
- **对话**：SSE 流式输出、可折叠思考过程、Markdown 渲染（标题/表格/代码块卡片）
- **RAG 检索**：BM25 + bge-small-zh-v1.5 向量混合检索，命中才注入，避免无关资料污染回答
- **文本工坊**：摘要、改写、翻译、提纲、邮件等 10 类任务模板
- **沙盒**：工作区固定在 `D:\AI`，区内文件直接读写；越界文件操作与 PowerShell 命令进入审批队列，用户可勾选「无需请求」放行，危险命令始终拒绝
- **网关口令**：`scripts/gateway.py` 控制数据集/训练/评测等管理接口，访客只能使用对话、工坊、沙盒

## 目录结构

```
app/         FastAPI 服务：路由、数据库、对话与流式输出、工具意图
core/        训练、推理运行时、分词与样本构造、适配器评测、llama.cpp 后端
forge/       文本工坊任务、检索（BM25 + 向量）、RAG 流程
sandbox/     沙盒文件操作与命令审批
scripts/     训练/导出/评测/网关/数据构建脚本
tools/       启动与日志看板（*.ps1）；llama.cpp 二进制不入库，需自行放置
web/         原生 JS 前端（对话、数据集、训练、评测、工坊、沙盒）
data/demo    示例知识库（43 + 139 条铸语知识）
data/eval    评测题库（能力 / 格式 / 工具调用）
data/train   工具能力训练样本
models/      基座模型与 GGUF（不入库）
outputs/     训练任务、适配器、评测报告（不入库）
```

## 快速开始

```powershell
# 1. 虚拟环境
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. 准备基座模型（任选其一）
#    放到 models/Qwen__Qwen2.5-1.5B-Instruct
#    或设置 HF_ENDPOINT=https://hf-mirror.com 后用 huggingface_hub 下载

# 3. 生成管理口令并启动
"你的口令" | Out-File -Encoding ascii scripts\admin_key.txt
.\.venv\Scripts\python.exe scripts\gateway.py
```

访问 `http://127.0.0.1:8765/?key=你的口令`，第一次登录后浏览器会记住身份。

> 如果不需要口令门禁，直接运行 `app/main.py` 或 `uvicorn app.main:app --port 8765` 即可。

## 训练与导出

- 界面：数据集页选数据 → 训练页设参数 → 启动；日志实时写入 `outputs/jobs/<id>/train.log`
- 命令行看板：

```powershell
powershell -ExecutionPolicy Bypass -File tools\watch_job.ps1      # 自动跟最新任务
```

- 合并适配器并导出 GGUF（需要自行准备 llama.cpp 的 `convert_hf_to_gguf.py` 与 `llama-quantize.exe`）：

```powershell
.\.venv\Scripts\python.exe scripts\export_merged.py <adapter_id>
.\.venv\Scripts\python.exe <llama.cpp>\convert_hf_to_gguf.py models\gguf\merged-1.5b --outfile models\gguf\model-f16.gguf --outtype f16
<llama.cpp>\llama-quantize.exe models\gguf\model-f16.gguf models\gguf\model-q4_k_m.gguf Q4_K_M
```

- 切换推理后端与模型：编辑 `config.yaml` 的 `inference.backend`（`llamacpp` / `torch`）与 `inference.gguf`，重启服务生效。

## 评测

```powershell
.\.venv\Scripts\python.exe scripts\eval_live.py                                   # 24 题能力基线
.\.venv\Scripts\python.exe scripts\eval_live.py --bench data\eval\benchmark_markdown.jsonl
.\.venv\Scripts\python.exe scripts\eval_repeat.py --id p02 --times 3              # 单题稳定性
```

报告写入 `outputs/evals/`。

## 已知限制

- GGUF 后端不提供困惑度（ppl），评测只保留关键词命中率
- GGUF 权重已合并适配器，对话页切换其他适配器不生效；换适配器需重新合并导出
- llama.cpp 多线程下 `temperature=0` 非严格确定，单轮评测可能有 ±1 题波动
- 无法获取实时信息（天气/新闻/股价），相关提问会被强制拒答
- 3B 模型在 16GB 内存机器上训练非常吃紧，建议只做推理

## 技术栈

Python 3.12 · FastAPI · SQLite · Transformers · PEFT · llama-cpp-python · llama.cpp · bge-small-zh-v1.5 · 原生 JavaScript
