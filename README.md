# 🔍 Research Buddy

基于 LangGraph + Langfuse 的深度研究 Agent。输入一个问题，自动拆解、搜索、验证、生成结构化研究报告。

![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ 功能特点

- **智能拆解** — LLM 自动将研究问题拆解为 3-5 个子问题
- **并行搜索** — Tavily API 并行搜索各子问题，中文搜索质量高
- **交叉验证** — 规则引擎检查搜索结果充足性，不足时自动补充搜索
- **结构化报告** — LLM 综合生成带来源引用、置信度评估的研究报告
- **自我反思** — LLM 评估报告质量，不达标时自动修正并补充搜索
- **流式输出** — SSE 实时推送研究进度和报告内容
- **Web UI** — 深色主题界面，Markdown 报告渲染
- **可观测性** — Langfuse 全链路 Trace、评分、Prompt 管理
- **人机交互** — 支持规划后确认子问题、报告后补充要求

## 🏗️ 架构

```
用户输入研究问题
    ↓
[规划] LLM 拆解为 3-5 个子问题
    ↓
[搜索] Tavily API 并行搜索各子问题
    ↓
[验证] 规则引擎检查搜索结果充足性 ──不足──→ 回到搜索
    ↓ 通过
[综合] LLM 生成结构化研究报告（流式输出）
    ↓
[反思] LLM 自评报告质量 ──不达标──→ 补充搜索 → 重新综合
    ↓ 通过
输出：结构化报告 + 来源引用 + 置信度
```

## 📁 项目结构

```
research-buddy/
├── src/research_buddy/
│   ├── __init__.py              # 包入口
│   ├── state.py                 # State 定义（TypedDict + operator.add 追加语义）
│   ├── config.py                # 环境变量配置
│   ├── graph.py                 # LangGraph 工作流（节点、边、条件分支、HITL）
│   ├── api.py                   # FastAPI 应用（HTTP + SSE + 静态文件）
│   ├── nodes/                   # LangGraph 节点实现
│   │   ├── planner.py           #   规划节点 — 拆解问题，生成自适应语言搜索词
│   │   ├── searcher.py          #   搜索节点 — 并行搜索，支持补充搜索
│   │   ├── validator.py         #   验证节点 — 检查结果充足性
│   │   ├── synthesizer.py       #   综合节点 — LLM 流式生成报告
│   │   └── reflector.py         #   反思节点 — LLM 评估报告质量
│   ├── tools/
│   │   └── search.py            # Tavily API 搜索（专为 AI Agent 优化）
│   ├── eval/                    # Langfuse 评估体系
│   │   ├── dataset.py           #   测试数据集（8 个研究问题）
│   │   ├── judge.py             #   LLM-as-Judge 自动评分
│   │   └── prompts.py           #   Prompt 版本管理（Langfuse 远程 + 本地 fallback）
│   └── static/
│       └── index.html           # Web UI（深色主题，SSE 流式，Markdown 渲染）
├── scripts/
│   ├── dev.sh                   # 启动/停止/重启脚本（自动杀端口占用）
│   ├── run_api.py               # 启动 API 服务
│   ├── run_eval.py              # 运行评估
│   ├── test_phase1.py           # 测试线性图
│   ├── test_phase2.py           # 测试条件分支 + 循环修正
│   └── test_phase3.py           # 测试人机交互
├── tests/
├── docs/
│   └── learning-notes.md        # 学习笔记
├── pyproject.toml               # 依赖管理（uv）
├── Dockerfile                   # Docker 部署
├── .env.example                 # 环境变量模板
└── CLAUDE.md                    # 项目说明（AI 辅助开发）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 需要 Python 3.11+ 和 uv
pip install uv

# 克隆项目
git clone <repo-url>
cd research-buddy

# 安装依赖
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```env
# LLM — 支持 OpenAI 兼容的任何 API
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 可选：生产反思使用独立模型；留空时与 OPENAI_MODEL 相同
# REFLECTOR_MODEL=gpt-4o

# 可选：离线文章质量回归使用固定独立模型
# ARTICLE_EVAL_JUDGE_MODEL=gpt-4o

# 搜索 — Tavily（https://tavily.com 注册，免费 1000 次/月）
TAVILY_API_KEY=tvly-xxx

# Langfuse 可观测性（可选，https://cloud.langfuse.com 注册免费版）
LANGFUSE_PUBLIC_KEY=pk-xxx
LANGFUSE_SECRET_KEY=sk-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

> **提示**：LLM 支持任何 OpenAI 兼容 API，如讯飞 MaaS、硅基流动、DeepSeek 等。

> **`TAVILY_API_KEY` 是必填的**。缺失或搜索层连续失败时，研究会直接中止并报明确错误，而不是让模型凭内部知识编一份零来源的报告。

每次文章生成会自动进入独立素材库，响应中的 `article_id` 可用于查询全文、阶段版本和评价。
接口与独立 Judge 配置见 [文章素材库说明](docs/article-archive.md)。

### 2.1 可选：中文向量检索

知识层的语义检索默认用 ChromaDB 内置的 `all-MiniLM-L6-v2`，它以英文语料训练，中文召回偏低。需要中文检索质量时选一个后端：

```env
# 本地多语言模型（质量好，但会拉入 torch，安装体积 +2~3GB）
EMBEDDING_BACKEND=sentence-transformers

# 或走 OPENAI_API_BASE 的 /embeddings（零安装体积，但中转站不一定支持）
EMBEDDING_BACKEND=openai
```

用本地模型前先装可选依赖：

```bash
uv sync --extra multilingual
```

后端不可用时会打 WARNING 并降级到默认模型，不会静默切换。**切换后端后已有向量库不能混用**（两个 MiniLM 都是 384 维，混用不报错但结果失去意义）：要么改回原模型，要么删掉 `data/chroma_db` 重建。

### 3. 启动服务

```bash
# 启动（自动杀旧进程，不报 Address already in use）
./scripts/dev.sh

# 其他命令
./scripts/dev.sh stop      # 停止
./scripts/dev.sh restart   # 重启
./scripts/dev.sh status    # 查看状态
```

浏览器打开 http://localhost:8000 ，输入问题即可使用。

## 🧪 测试

```bash
# 单元测试（全离线，不需要 API Key）
uv run pytest -q

# 文章质量 A/B 回归（会实际调用 LLM，先用一题一次估算费用）
uv run python scripts/run_article_eval.py \
  --candidate-rules eval/prompts/candidate-v1.md \
  --limit 1 --samples 1

# 测试线性工作流（规划→搜索→综合）
uv run python scripts/test_phase1.py

# 测试条件分支 + 循环修正（验证→反思→补充搜索）
uv run python scripts/test_phase2.py

# 测试人机交互（规划后确认、报告后补充）
uv run python scripts/test_phase3.py

# 运行评估（LLM-as-Judge 评分，需要 Langfuse 密钥才会写入分数）
uv run python -m research_buddy.eval.dataset   # 首次：创建/更新测试集
uv run python scripts/run_eval.py
```

程序测试与文章质量评测用途不同。后者会冻结搜索证据，对 baseline/candidate 多次采样，生成本地 JSON 和 HTML 对比报告；完整说明见 [文章质量回归评测](docs/article-quality-evaluation.md)。

## 🐳 Docker 部署

```bash
# 构建镜像
docker build -t research-buddy .

# 运行（需要传入环境变量）
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  research-buddy
```

## 🔌 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/research` | 同步研究（等待完整报告） |
| GET | `/research/stream?question=xxx` | SSE 流式研究（实时进度） |
| POST | `/research/stream` | SSE 流式研究（POST 请求体） |

### SSE 事件类型

| 事件 | 说明 |
|------|------|
| `progress` | 节点进度（当前执行到哪个节点） |
| `message` | 详细进度消息 |
| `report` | 研究报告（完整 Markdown） |
| `done` | 研究完成 |
| `error` | 发生错误 |

### 示例：curl 流式研究

```bash
curl -N "http://localhost:8000/research/stream?question=LangGraph%20和%20LangChain%20的区别"
```

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 工作流编排 | LangGraph（StateGraph、条件边、循环、HITL） |
| LLM | langchain-openai（OpenAI 兼容 API） |
| 搜索 | Tavily API（专为 AI Agent 优化） |
| 可观测性 | Langfuse（Trace、Span、评分、Prompt 管理） |
| Web 框架 | FastAPI + SSE |
| 包管理 | uv |
| 部署 | Docker |

## 📖 学习路线

本项目按 5 个阶段递进式构建，每阶段聚焦 LangGraph / Langfuse 的核心特性：

| 阶段 | 内容 | 学到 |
|------|------|------|
| Phase 1 | 线性图 + Langfuse 接入 | StateGraph、Node、State、Trace/Span |
| Phase 2 | 条件分支 + 循环修正 | Conditional Edge、Loop、验证/反思 |
| Phase 3 | Human-in-the-loop | interrupt_before、Checkpoint、Command(resume=) |
| Phase 4 | Langfuse 评估体系 | Dataset、LLM-as-Judge、Prompt Management |
| Phase 5 | 生产化 | FastAPI、SSE 流式、Docker 部署 |

## 📝 License

MIT
