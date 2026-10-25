# 架构全景

![系统架构](./images/03-system-architecture.png)

## 系统组件

系统是一个**单进程单体应用**，由 Uvicorn/FastAPI 托管，LangGraph 在后台线程编排研究流程，数据本地持久化。

| 层 | 主要组件 | 职责 |
|---|---|---|
| 展示层 | `static/`（原生 JS SPA） | 研究操作、SSE 展示、报告阅读、素材库、版本比较、导出 |
| API 层 | `api.py` | HTTP/SSE、请求校验、后台运行恢复、HITL、静态资源 |
| 编排层 | `graph.py`、`state.py` | LangGraph 节点、条件边、循环、共享状态、检查点 |
| 研究节点 | `planner`、`searcher`、`validator` | 问题拆解、并行搜索、证据充分性判断 |
| 编辑节点 | `editorial_planner`、`synthesizer`、`language_editor`、`article_editor` | 编辑简报、写作、续写、语言与事实审校、图片插入 |
| 质量节点 | `reflector` | 独立评分、确定性规则、补搜/重写决策、最佳稿恢复 |
| 知识与追踪 | `knowledge/*`、`tracking/*` | 历史召回、增量研究、变化检测、定时追踪、通知 |
| 文章档案 | `archive.py`、`knowledge/db.py` | 每次生成、阶段稿、评审和策展状态永久保存 |
| 工具层 | `tools/search.py`、`tools/images.py`、`utils.py` | Tavily、图片下载与视觉筛选、LLM 和流式公共能力 |
| 评估层 | `eval/*`、`scripts/run_article_eval.py` | Langfuse 数据集、LLM-as-Judge、冻结证据 A/B 回归 |

## 模块划分

```
src/research_buddy/
├── graph.py              # LangGraph 工作流（普通/知识/HITL/追踪四类图）
├── state.py              # ResearchState（TypedDict + 追加/覆盖两种 reducer 语义）
├── config.py             # 环境变量配置（.env 为事实来源）
├── api.py                # FastAPI 应用（HTTP + SSE + 静态文件）
├── archive.py            # 文章素材库归档
├── nodes/                # LangGraph 节点实现
│   ├── planner.py            # 规划节点
│   ├── searcher.py           # 搜索节点
│   ├── validator.py          # 验证节点
│   ├── editorial_planner.py  # 编辑简报节点
│   ├── synthesizer.py        # 写作节点（流式）
│   ├── language_editor.py    # 语言审校节点
│   ├── article_editor.py     # 事实审校节点
│   ├── reflector.py          # 反思节点
│   ├── knowledge_lookup.py   # 知识召回节点
│   ├── knowledge_store.py    # 知识入库节点
│   ├── diff_analyzer.py      # 变化分析节点
│   └── change_notifier.py    # 变化通知节点
├── tools/
│   ├── search.py         # Tavily 搜索封装
│   └── images.py         # 图片下载与视觉筛选
├── knowledge/            # 知识库（db.py / store.py / vector.py）
├── tracking/             # 追踪（diff.py / notifier.py / scheduler.py）
└── eval/                 # 评估（dataset.py / judge.py / prompts.py / article_quality.py）
```

## 数据流

![数据流](./images/04-data-flow.png)

核心研究链的数据流（两个闭环）：

```text
问题输入
  → planner 拆解为子问题
  → searcher 并行搜索（Tavily）
  → validator 证据充分性判断
      ├─ 有缺口且可搜 → 回到 searcher（证据闭环）
      └─ 通过 → editorial_planner 生成证据账本 + 编辑简报
  → synthesizer 流式写作
  → language_editor 语言审校
  → article_editor 事实审校
  → reflector 独立评分
      ├─ 有缺口且预算可用 → 回到 searcher 补搜
      ├─ 不达标 → 回到 synthesizer 基于现有证据重写（成稿闭环）
      └─ 通过 → 输出成稿 + 来源引用 + 置信度
```

知识研究图在核心链前后各加一个节点：`knowledge_lookup → 核心研究链 → knowledge_store`；追踪图继续执行 `knowledge_store → diff_analyzer → change_notifier`。

## 外部集成

| 外部服务 | 协议/端点 | 认证方式 | 说明 |
|---|---|---|---|
| LLM | OpenAI 兼容 HTTP API（`OPENAI_API_BASE`） | `OPENAI_API_KEY`（Bearer） | 主模型 + 可选 `REFLECTOR_MODEL`/`ARTICLE_EVAL_JUDGE_MODEL` |
| Tavily | Tavily Search API | `TAVILY_API_KEY` | 必填，面向 AI Agent 的搜索 |
| Langfuse | `LANGFUSE_HOST`（默认 cloud.langfuse.com） | Public/Secret Key | Trace、评分、Prompt 管理，超时放宽至 20s |
| 视觉模型 | 可选独立 `VISION_API_BASE` | `VISION_API_KEY` | 图片相关性/水印筛选，未配置则整条图片管线关闭 |
| 通知 Webhook | 企业微信/钉钉/Telegram 机器人 URL | URL 内嵌 key | 追踪变化推送，可选 |

## 部署拓扑

```text
浏览器
  → http://localhost:8000
  → Uvicorn / FastAPI 单进程（Docker 容器内）
     → LangGraph 后台线程 + SSE
     → SQLite / ChromaDB / 图片缓存（data/ 目录）
     → 外部 LLM / Tavily / Langfuse / Webhook
```

- 通过 `Dockerfile`（`python:3.11-slim` + uv）构建，`EXPOSE 8000`，`uv run uvicorn` 启动。
- 运行时数据目录 `data/`：`research_buddy.db`（业务/运行/素材库）、`hitl_checkpoints.db`（HITL 检查点）、`chroma_db/`（向量索引）、`research-images/`（本地图片缓存）。
- SQLite 与 APScheduler 本地调度器适合**单机单进程**部署；多实例需将运行状态、检查点、调度锁和数据库迁移到共享基础设施。

---
*最后更新: 2026-09-03 — 初始化生成*
