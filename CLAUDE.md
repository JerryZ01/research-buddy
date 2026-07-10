# Research Buddy

基于 LangGraph + Langfuse 的深度研究 Agent，输入一个问题，自动拆解、搜索、验证、生成结构化研究报告。

## 为什么做这个

- 解决真实需求：做考编/技术调研时，手动搜索+整理太耗时
- 学习 LangGraph 全部核心特性：多步骤编排、条件分支、循环修正、人机交互、Checkpoint
- 学习 Langfuse 全链路可观测性：Trace、Span、评分、Prompt 管理、Dataset 评估

## 技术栈

- Python 3.11+
- langgraph >= 0.2
- langchain-openai（接中转站 API）
- langfuse >= 2.0
- tavily-python（搜索 API）
- fastapi（API 层，后期加）
- uv（包管理）

## 工作流

```
用户输入研究问题
    ↓
[规划] 拆解为 3-5 个子问题，制定搜索策略
    ↓
[搜索] 并行搜索各子问题（Tavily API）
    ↓
[提取] 从搜索结果中提取关键信息
    ↓
[验证] 交叉验证信息一致性，标记矛盾/不确定
    ↓
[综合] 生成结构化研究报告
    ↓
[反思] 自我评估报告质量 ──不足──→ 回到搜索（补充搜索）
    ↓ 通过
输出：结构化报告 + 来源引用 + 置信度
```

## 分阶段路线

### Phase 1：最小可用（线性图 + Langfuse 基础）
- [x] 项目初始化：uv、依赖、目录结构
- [x] 最简 StateGraph：问题→拆解→搜索→综合→输出
- [x] Langfuse CallbackHandler 接入，看到 Trace
- [x] 用一个真实问题跑通全流程

**学到**：StateGraph、Node、State、Langfuse Trace/Span/Generation

### Phase 2：条件分支 + 循环修正
- [x] 加入验证节点：交叉验证搜索结果
- [x] 加入反思节点：LLM 自评报告质量
- [x] 条件边：质量不足 → 补充搜索 → 重新综合
- [x] 最大循环次数限制（防无限循环）

**学到**：Conditional Edge、Loop、State 更新策略

### Phase 3：Human-in-the-loop
- [x] 规划后暂停，用户确认/调整子问题
- [x] 综合后暂停，用户补充要求
- [x] Checkpoint 保存进度，可中断后继续

**学到**：interrupt_before/interrupt_after、Checkpoint、Command(resume=)

### Phase 4：Langfuse 评估体系
- [x] 构建测试 Dataset（5-10 个研究问题 + 预期要点）
- [x] LLM-as-Judge 自动评分（相关性、完整性、准确性）
- [x] 人工评分 UI（Langfuse 自带）
- [x] Prompt 版本管理（各节点 prompt 在 Langfuse 中管理）

**学到**：Dataset、Evaluation、LLM-as-Judge、Prompt Management

### Phase 5：生产化
- [x] FastAPI 包装为 HTTP 服务
- [x] 流式输出（Server-Sent Events）
- [x] 简单 Web UI（可选）
- [x] Docker 部署

**学到**：Streaming、生产部署、API 设计

### Phase 6：知识层（增量研究 + 向量检索）
- [x] SQLite 存储报告元数据和关键事实
- [x] ChromaDB 向量存储和检索
- [x] knowledge_lookup 节点：增量研究时查询历史知识
- [x] knowledge_store 节点：保存研究报告到知识库
- [x] 增量模式：只搜索新信息，复用历史知识

**学到**：持久化存储、向量检索、增量研究模式

### Phase 7：定时追踪 + 变化检测 + 智能通知
- [x] APScheduler 定时追踪任务
- [x] DiffAnalyzer 变化分析（LLM + difflib 双模式）
- [x] Notifier 多平台通知（企业微信/钉钉/Telegram/Generic）
- [x] 变化检测节点和通知节点集成到追踪工作流

**学到**：定时任务、文本差异分析、Webhook 通知

## 参考项目

- [cmbagent_lg](https://github.com/borisbolliet/cmbagent_lg) — LangGraph + Langfuse 的深度研究实现
- [langgraph-template-travel-planner](https://github.com/datarootsio/langgraph-template-travel-planner) — LangGraph + Langfuse + HITL 模板
- [fastapi-mcp-langgraph-template](https://github.com/NicholasGoh/fastapi-mcp-langgraph-template) — FastAPI + LangGraph 生产模板

## 目录结构

```
research-buddy/
├── CLAUDE.md              # 项目说明（本文件）
├── pyproject.toml          # 依赖管理
├── .env.example            # 环境变量模板
├── src/
│   └── research_buddy/
│       ├── __init__.py
│       ├── graph.py        # LangGraph 工作流（4 种图 + 运行函数）
│       ├── state.py        # State 定义（TypedDict + operator.add 追加语义）
│       ├── config.py       # 环境变量配置
│       ├── api.py          # FastAPI 应用（HTTP + SSE + 静态文件）
│       ├── nodes/          # LangGraph 节点实现
│       │   ├── __init__.py
│       │   ├── planner.py      # 规划节点
│       │   ├── searcher.py     # 搜索节点（含提取逻辑）
│       │   ├── validator.py    # 验证节点
│       │   ├── synthesizer.py  # 综合节点
│       │   ├── reflector.py    # 反思节点
│       │   ├── knowledge_lookup.py  # 知识查询节点
│       │   ├── knowledge_store.py   # 知识存储节点
│       │   ├── diff_analyzer.py     # 变化分析节点
│       │   └── change_notifier.py   # 变化通知节点
│       ├── tools/
│       │   ├── __init__.py
│       │   └── search.py    # Tavily API 搜索
│       ├── knowledge/       # 知识层（SQLite + ChromaDB）
│       │   ├── __init__.py
│       │   ├── db.py        # SQLite 数据库
│       │   ├── store.py     # 知识存储门面
│       │   └── vector.py    # ChromaDB 向量存储
│       ├── tracking/        # 追踪层（定时 + 通知）
│       │   ├── __init__.py
│       │   ├── scheduler.py # APScheduler 定时任务
│       │   ├── notifier.py  # 多平台 Webhook 通知
│       │   └── diff.py      # 文本差异分析
│       ├── eval/            # Langfuse 评估体系
│       │   ├── __init__.py
│       │   ├── dataset.py   # 测试数据集
│       │   ├── judge.py     # LLM-as-Judge 评分
│       │   └── prompts.py   # Prompt 版本管理
│       └── static/
│           └── index.html   # Web UI（深色主题，SSE 流式）
├── tests/
│   ├── test_routing.py     # 路由逻辑测试
│   ├── test_state.py       # State 定义测试
│   ├── test_graph.py       # 图构建测试
│   └── test_diff.py        # DiffAnalyzer 测试
├── scripts/
│   ├── dev.sh              # 启动/停止/重启脚本
│   ├── run_api.py          # 启动 API 服务
│   ├── run_eval.py         # 运行评估
│   ├── test_phase1.py      # 测试线性图
│   ├── test_phase2.py      # 测试条件分支 + 循环修正
│   └── test_phase3.py      # 测试人机交互
├── docs/
│   └── learning-notes.md   # 学习笔记
└── Dockerfile              # Docker 部署
```
