# Research Buddy 架构全解

> 基于 LangGraph + Langfuse 的深度研究 Agent，输入一个问题，自动拆解、搜索、验证、生成结构化研究报告。

---

## 一、项目概览

### 1.1 核心能力

| 能力 | 说明 |
|------|------|
| 深度研究 | 输入一个问题 → 自动拆解为子问题 → 并行搜索 → 交叉验证 → 综合报告 |
| 循环修正 | LLM 自评报告质量 → 不足则补充搜索 → 重新综合 → 最多 N 轮 |
| 人机交互 | 规划后暂停让用户调整子问题，综合后暂停让用户补充要求 |
| 增量研究 | 基于历史知识只搜索新信息，复用已有知识，避免重复搜索 |
| 定时追踪 | 按 cron 表达式定期重新研究，检测变化，推送通知 |
| 可观测性 | Langfuse 全链路 Trace/Span/Generation，LLM-as-Judge 自动评分 |

### 1.2 技术栈

| 层 | 技术 | 作用 |
|----|------|------|
| 工作流编排 | LangGraph ≥ 0.2 | StateGraph + Node + Conditional Edge + Loop |
| LLM | langchain-openai | ChatOpenAI（通过中转站 API） |
| 搜索 | tavily-python | Tavily 搜索 API（专为 AI Agent 优化） |
| 可观测性 | langfuse ≥ 2.0 | Trace、Span、评分、Prompt 管理、Dataset 评估 |
| 向量检索 | chromadb ≥ 1.5 | 报告分块 + 关键事实的语义检索 |
| 持久化 | SQLite | 报告元数据、主题、追踪记录 |
| 定时调度 | APScheduler ≥ 3.11 | AsyncIOScheduler + CronTrigger |
| API 层 | FastAPI + uvicorn | HTTP + SSE 流式 + 静态文件 |
| 包管理 | uv | 依赖管理 + 虚拟环境 |

### 1.3 目录结构

```
research-buddy/
├── CLAUDE.md                    # 项目说明 + 路线图
├── pyproject.toml               # 依赖管理（uv + hatchling）
├── .env.example                 # 环境变量模板
├── Dockerfile                   # Docker 部署
├── src/
│   └── research_buddy/
│       ├── __init__.py          # 包入口 + 日志配置 + CLI main()
│       ├── config.py            # 环境变量配置（load_dotenv）
│       ├── state.py             # ResearchState TypedDict 定义
│       ├── utils.py             # 共享工具（parse_llm_json, create_llm, ...）
│       ├── graph.py             # LangGraph 工作流（4 种图 + 运行函数）
│       ├── api.py               # FastAPI 应用（HTTP + SSE + 静态文件）
│       ├── nodes/               # LangGraph 节点实现
│       │   ├── planner.py       # 规划节点
│       │   ├── searcher.py      # 搜索节点（并行 + 增量去重）
│       │   ├── validator.py     # 验证节点（充足性检查 + URL 去重）
│       │   ├── synthesizer.py   # 综合节点（流式输出）
│       │   ├── reflector.py     # 反思节点（LLM 自评 + 条件路由）
│       │   ├── knowledge_lookup.py  # 知识查询节点
│       │   ├── knowledge_store.py   # 知识存储节点
│       │   ├── diff_analyzer.py     # 变化分析节点
│       │   └── change_notifier.py   # 变化通知节点
│       ├── tools/
│       │   └── search.py        # Tavily API 搜索（懒初始化单例）
│       ├── knowledge/           # 知识层
│       │   ├── db.py            # SQLite 数据库（线程本地连接）
│       │   ├── store.py         # KnowledgeStore 统一门面
│       │   └── vector.py        # ChromaDB 向量存储
│       ├── tracking/            # 追踪层
│       │   ├── scheduler.py     # APScheduler 定时调度器
│       │   ├── notifier.py      # 多平台 Webhook 通知
│       │   └── diff.py          # 文本差异分析（difflib + LLM）
│       ├── eval/                # 评估层
│       │   ├── dataset.py       # Langfuse 测试数据集
│       │   ├── judge.py         # LLM-as-Judge 评分
│       │   └── prompts.py       # Prompt 版本管理
│       └── static/
│           └── index.html       # Web UI（深色主题，SSE 流式）
├── tests/
│   ├── test_routing.py          # 路由逻辑测试
│   ├── test_state.py            # State 定义测试
│   ├── test_graph.py            # 图构建测试
│   └── test_diff.py             # DiffAnalyzer 测试
├── scripts/
│   ├── dev.sh                   # 启动/停止/重启脚本
│   ├── run_api.py               # 启动 API 服务
│   ├── run_eval.py              # 运行评估
│   ├── test_phase1.py           # 线性图端到端测试
│   ├── test_phase2.py           # 条件分支 + 循环修正测试
│   └── test_phase3.py           # 人机交互测试
└── docs/
    └── learning-notes.md        # 学习笔记
```

---

## 二、核心架构：LangGraph 工作流

### 2.1 状态定义（state.py）

`ResearchState` 是 LangGraph 的全局共享状态，所有节点读写同一份状态：

```python
class ResearchState(TypedDict):
    # 输入
    question: str                                    # 原始研究问题

    # 知识层
    topic_id: str                                    # 关联主题 ID
    knowledge_context: str                           # 历史知识上下文
    has_knowledge: bool                              # 是否有历史知识
    is_incremental: bool                             # 增量模式标志
    known_source_urls: list[str]                     # 已有来源 URL（增量去重）
    key_facts: Annotated[list[str], operator.add]     # 关键事实（追加语义）
    saved_report_id: str                             # 保存后的报告 ID

    # 规划阶段
    sub_questions: Annotated[list[SubQuestion], operator.add]  # 子问题（追加）

    # 搜索阶段
    search_results: Annotated[list[SearchResult], operator.add]  # 搜索结果（追加）

    # 验证阶段
    validation_gaps: Annotated[list[ValidationGap], operator.add]  # 信息缺口（追加）

    # 综合阶段
    report: str                                      # 最终报告

    # 反思阶段
    reflection_pass: bool                            # 反思是否通过
    reflection_feedback: str                         # 反馈/改进建议
    reflection_round: int                            # 当前反思轮次

    # Human-in-the-loop
    user_feedback: str                               # 用户反馈

    # 追踪层
    detected_changes: Annotated[list[dict], operator.add]  # 检测到的变化（追加）
    similarity: float                                # 新旧报告相似度
    tracking_log_id: str                             # 追踪记录 ID
    notification_sent: bool                          # 是否已发送通知

    # 进度消息
    messages: Annotated[list[str], operator.add]     # 节点进度消息（追加）
```

**关键设计**：`Annotated[list, operator.add]` 让列表字段实现**追加而非覆盖**语义。当多个节点返回 `search_results` 时，LangGraph 自动 extend 而非 overwrite。

### 2.2 四种工作流图

#### 图 1：核心研究图（create_research_graph）

最简单的线性图，无知识层，全自动：

```
 ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐     ┌──────────────┐
 │          │     │          │     │          │     │           │     │              │
 │ planner  ├────►│ searcher ├────►│validator ├────►│synthesizer├────►│  reflector   │
 │          │     │          │     │          │     │           │     │              │
 └──────────┘     └──────────┘     └──────────┘     └───────────┘     └──────┬───────┘
                                                                           │
                                                                     ┌─────┴─────┐
                                                                     │           │
                                                               pass ✅│           │not pass ⚠️
                                                                     │           │
                                                               ┌─────▼─────┐     │
                                                               │    END    │     │
                                                               └───────────┘     │
                                                                                 │
                                                                    ┌────────────▼
                                                                    │
                                                                    │
                                                             ┌──────▼───────┐
                                                             │   searcher   │ ← 循环
                                                             └──────────────┘
```

**路由函数** `should_continue`：
- `reflection_pass == True` → END
- `reflection_round >= MAX_REFLECTION_ROUNDS` → END（防无限循环）
- 否则 → searcher（补充搜索）

#### 图 2：知识研究图（create_knowledge_research_graph）

带知识层，支持增量研究：

```
 ┌─────────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
 │                 │     │          │     │          │     │          │
 │knowledge_lookup ├────►│ planner  ├────►│ searcher ├────►│validator │
 │                 │     │          │     │          │     │          │
 └─────────────────┘     └──────────┘     └────┬─────┘     └──────────┘
                                               │                 │
                                    ┌──────────┘                 │
                                    ▼                            ▼
                             ┌───────────┐              ┌───────────┐
                             │  searcher │◄─────────────┤           │
                             │  (去重)   │   not pass   │synthesizer│
                             └─────┬─────┘     ⚠️      └─────┬─────┘
                                   │                         │
                                   │                         ▼
                                   │                  ┌──────────────┐
                                   │                  │  reflector   │
                                   │                  └──────┬───────┘
                                   │                         │
                                   │                   ┌─────┴─────┐
                                   │             pass ✅│           │not pass ⚠️
                                   │                   │           │
                                   │                   ▼           │
                                   │          ┌────────────────┐   │
                                   │          │knowledge_store │   │
                                   │          │ (双写 SQLite +  │   │
                                   │          │  ChromaDB)     │   │
                                   │          └───────┬────────┘   │
                                   │                  │            │
                                   │                  ▼            │
                                   │             ┌────────┐       │
                                   └────────────►│  END   │       │
                                                 └────────┘       │
                                                                  │
                                                     (回到 searcher 补充搜索)
```

#### 图 3：追踪图（create_tracking_graph）

在知识研究图基础上，增加变化检测和通知：

```
 ┌─────────────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
 │knowledge_lookup ├────►│ planner  ├────►│ searcher ├────►│validator │
 └─────────────────┘     └──────────┘     └──────────┘     └────┬─────┘
                                                                │
                                                                ▼
                                                         ┌───────────┐
                                            not pass ⚠️  │synthesizer│
                                         ┌───────────────┤           │
                                         │               └─────┬─────┘
                                         │                     │
                                         │                     ▼
                                  ┌──────▼──────┐       ┌──────────────┐
                                  │  searcher   │       │  reflector   │
                                  │  (补充搜索)  │       └──────┬───────┘
                                  └─────────────┘              │
                                                        ┌──────┴──────┐
                                                  pass ✅│             │not pass ⚠️
                                                        ▼             │
                                               ┌────────────────┐    │
                                               │knowledge_store │    │
                                               │ (双写 SQLite +  │    │
                                               │  ChromaDB)     │    │
                                               └───────┬────────┘    │
                                                       │             │
                                                       ▼             │
                                              ┌──────────────┐      │
                                              │ diff_analyzer│      │
                                              │ (两层变化检测) │      │
                                              └───────┬──────┘      │
                                                      │              │
                                                      ▼              │
                                             ┌─────────────────┐    │
                                             │ change_notifier │    │
                                             │ (通知策略判断)    │    │
                                             └────────┬────────┘    │
                                                      │              │
                                                      ▼              │
                                                 ┌────────┐         │
                                                 │  END   │◄────────┘
                                                 └────────┘   (回到 searcher)
```

#### 图 4：HITL 研究图（create_research_graph_with_hitl）

带人机交互，使用 MemorySaver Checkpointer：

```
 ┌──────────┐         ┌──────────┐     ┌──────────┐     ┌───────────┐
 │          │    ⏸    │👤 用户   │     │          │     │           │
 │ planner  ├────────►│确认/调整 ├────►│ searcher ├────►│validator  │
 │          │ interrupt│ 子问题   │     │          │     │           │
 └──────────┘         └──────────┘     └──────────┘     └─────┬─────┘
                                                            │
                                                            ▼
                                                     ┌───────────┐         ┌──────────┐
                                                     │           │    ⏸    │👤 用户   │
                                                     │synthesizer├────────►│查看报告  │
                                                     │           │ interrupt│补充要求  │
                                                     └───────────┘         └────┬─────┘
                                                                                │
                                                                                ▼
                                                                         ┌──────────────┐
                                                                         │  reflector   │
                                                                         └──────┬───────┘
                                                                                │
                                                                                ▼
                                                                           ┌────────┐
                                                                           │  END   │
                                                                           └────────┘
```

- `interrupt_before=["searcher", "reflector"]`：在搜索前和反思前暂停
- 用户可通过 `Command(resume=...)` 恢复执行

### 2.3 节点详解

#### planner（规划节点）

**职责**：将研究问题拆解为 3-5 个子问题，生成英文搜索词

**两种模式**：
- **全新模式**：使用 `PLANNER_PROMPT`，正常拆解 3-5 个子问题
- **增量模式**：使用 `INCREMENTAL_PLANNER_PROMPT`，基于已有知识只规划缺失的 2-3 个子问题

**流程**：
1. 从 state 读取 question、has_knowledge、knowledge_context、is_incremental
2. 选择 prompt 模板（优先从 Langfuse 拉取，fallback 到本地）
3. 调用 `create_llm()` 创建 LLM 实例
4. LLM 返回 JSON → `parse_llm_json()` 解析（含 code-fence 剥离 + 错误处理）
5. 返回 `{"sub_questions": [...], "messages": [...]}`

#### searcher（搜索节点）

**职责**：并行搜索各子问题，支持补充搜索和增量去重

**流程**：
1. 合并搜索任务：validation_gaps（补充搜索）+ 未搜索的原始子问题
2. 增量模式：用 `normalize_url()` 规范化已有来源 URL，构建 known_urls 集合
3. `ThreadPoolExecutor(max_workers=min(total, 4))` 并行搜索
4. 增量模式去重：过滤 known_urls 中的搜索结果
5. 返回 `{"search_results": [...], "validation_gaps": []}`

**关键设计**：
- 并行搜索用 ThreadPoolExecutor（Tavily API 是 I/O 密集型）
- URL 去重用统一的 `normalize_url()`（去掉协议和末尾斜杠）

#### validator（验证节点）

**职责**：纯规则检查搜索结果充足性，不调用 LLM

**规则**：
- 每个子问题至少需要 `MIN_RESULTS_PER_SUB_QUESTION=2` 条有效结果
- 有效结果 = content 长度 ≥ `MIN_CONTENT_LENGTH=50` 且 URL 不重复
- 不足的子问题生成补充搜索任务（`original_query + " latest update"`）

#### synthesizer（综合节点）

**职责**：将搜索结果综合为结构化研究报告，流式输出

**三种模式**：
- **全新模式**：正常生成报告
- **增量模式**：基于已有知识补充更新，标注 🆕 新增和 ⚠️ 矛盾
- **改进模式**：根据反思反馈改进报告

**流式输出**：`create_llm(streaming=True)` → `llm.stream(prompt)` → 逐 chunk 打印

#### reflector（反思节点）

**职责**：LLM 自评报告质量，决定是否需要修正

**评分维度**：完整性(1-5) + 准确性(1-5) + 清晰度(1-5)，总分 ≥ 12 通过

**关键逻辑**：
- 有用户反馈时强制不通过（即使 LLM 说通过）
- total_score 验证与维度分数之和一致
- 未通过时生成 supplement_queries 作为补充搜索词

#### knowledge_lookup（知识查询节点）

**职责**：从知识库检索历史报告和关键事实，为增量研究提供上下文

**流程**：
1. 调用 `store.lookup(question, topic_id)` 查询知识库
2. 构建知识上下文：主题摘要 + 增量报告链 + 关键事实 + 相关片段
3. 追溯 parent_report_id 链（通过 `store.get_report()` 门面）

#### knowledge_store（知识存储节点）

**职责**：将研究报告保存到知识库（SQLite + ChromaDB 双写）

**流程**：
1. 提取关键事实（优先 state.key_facts → LLM 提取 → 搜索结果兜底）
2. 提取来源信息（URL 去重）
3. 增量模式：通过 `store.get_latest_report()` 获取 parent_report_id
4. 调用 `store.save_report()` 双写

#### diff_analyzer（变化分析节点）

**职责**：对比新旧报告，识别语义变化

**流程**：
1. 通过 `store.get_latest_report()` 获取旧报告
2. `DiffAnalyzer.analyze()` 分析差异
3. 保存变更到数据库（tracking_log + changes）
4. 返回 `{"detected_changes": [...], "similarity": float}`

#### change_notifier（变化通知节点）

**职责**：根据变化重要性决定是否发送通知

**通知策略**：
- 有 high 级别变化 → 必须通知
- 有 2+ 条 medium 变化 → 通知
- 只有 low 变化 → 不通知

---

## 三、知识层架构

### 3.1 三层结构

```
 ┌─────────────────────────────────────────────────────┐
 │                    上层代码                          │
 │  nodes/*.py      api.py      scheduler.py           │
 └─────────────┬───────────────────────────────────────┘
               │ 只调用 KnowledgeStore
               ▼
 ┌─────────────────────────────────────────────────────┐
 │              KnowledgeStore (store.py)               │
 │              统一 API 入口（门面模式）                  │
 └──────────┬──────────────────────┬───────────────────┘
            │ 委托                 │ 委托
            ▼                      ▼
 ┌──────────────────────┐  ┌──────────────────────────┐
 │  Database (SQLite)   │  │  VectorStore (ChromaDB)  │
 │  4 张表:             │  │  2 集合:                 │
 │  · topics            │  │  · report_chunks         │
 │  · reports           │  │  · key_facts             │
 │  · tracking_logs     │  │                          │
 │  · changes           │  │  多语言 Embedding:        │
 │                      │  │  multilingual-MiniLM     │
 └──────────────────────┘  └──────────────────────────┘
```

**设计原则**：上层代码只通过 `KnowledgeStore` 访问知识层，不直接使用 `Database` 或 `VectorStore`。`knowledge/__init__.py` 只导出 `KnowledgeStore` 和 `get_knowledge_store`。

### 3.2 Database（db.py）

**SQLite 数据层**，4 张表：

| 表 | 用途 | 关键字段 |
|----|------|---------|
| topics | 研究主题 | id, name, tracking_keywords, tracking_cron, tracking_enabled |
| reports | 研究报告 | id, topic_id, question, report, confidence, sources, key_facts, parent_report_id |
| tracking_logs | 追踪记录 | id, topic_id, status, changes_detected, change_summary |
| changes | 变更条目 | id, tracking_log_id, change_type, description, significance |

**线程安全**：使用 `threading.local()` 实现线程本地连接，每个线程有自己的 SQLite 连接。

**全局单例**：`get_db()` 懒初始化。

**时间戳**：`update_topic` 使用 SQLite `datetime('now')` 保持 UTC 一致。

### 3.3 VectorStore（vector.py）

**ChromaDB 向量存储**，2 个集合：

| 集合 | 用途 | Embedding 模型 |
|------|------|---------------|
| report_chunks | 报告文本分块 | paraphrase-multilingual-MiniLM-L12-v2 |
| key_facts | 关键事实 | 同上 |

**关键设计**：
- **多语言 Embedding**：`paraphrase-multilingual-MiniLM-L12-v2`（基于 ONNX，无需 torch，支持中文）
- **幂等写入**：`add_report`/`add_facts` 写入前先 `delete_report`/`delete_facts` 清理旧数据
- **距离阈值**：`max_distance=0.5`（cosine distance），过滤不相关结果
- **懒初始化**：client/collection 通过 @property 延迟创建

### 3.4 KnowledgeStore（store.py）

**统一门面**，整合 SQLite + ChromaDB：

| 方法 | 说明 |
|------|------|
| `save_report()` | 双写 SQLite + ChromaDB（报告分块 + 关键事实向量） |
| `delete_report()` | 先删 SQLite 再删向量（防止部分删除） |
| `delete_topic()` | 先删 SQLite（级联）再清理向量 |
| `lookup()` | 向量检索 chunks + facts + 结构化查表 |
| `get_knowledge_summary()` | 格式化主题知识摘要 |
| `get_report()` / `get_latest_report()` | 通过门面访问 db |

---

## 四、追踪层架构

### 4.1 DiffAnalyzer（diff.py）

**两层变化检测**：

```
 旧报告 + 新报告
       │
       ▼
 ┌─────────────────────────┐
 │ 第一层：difflib          │
 │ SequenceMatcher          │
 │ 计算文本相似度            │
 └───────────┬─────────────┘
             │
     ┌───────┴────────┐
     │                 │
 similarity ≥ 0.85   similarity < 0.85
     │                 │
     ▼                 ▼
 ┌─────────┐   ┌─────────────────────┐
 │无显著变化│   │ 第二层：LLM 语义分析  │
 │返回      │   │ 识别有意义的信息变更   │
 │has_      │   └──────────┬──────────┘
 │changes=  │              │
 │false     │     ┌────────┴────────┐
 └─────────┘     │                 │
             LLM 成功           LLM 失败
                 │                 │
                 ▼                 ▼
         ┌──────────────┐  ┌──────────────────┐
         │ 结构化变更列表│  │ Fallback:        │
         │ type         │  │ difflib 行级差异  │
         │ description  │  │ 简单变更列表      │
         │ significance │  └──────────────────┘
         └──────────────┘
```

**变更类型**：`new_info`（新增）、`update`（更新）、`contradiction`（矛盾）
**重要性级别**：`high`（重大）、`medium`（一般）、`low`（细节）

### 4.2 Notifier（notifier.py）

**多平台 Webhook 通知**：

| 平台 | URL 特征 | 格式 |
|------|---------|------|
| 企业微信 | `qyapi.weixin.qq.com` | Markdown 格式（**加粗**） |
| 钉钉 | `oapi.dingtalk.com` | Markdown 格式（无加粗） |
| 通用 | 其他 | JSON 格式 |

**频率控制**：同一主题 12 小时冷却期，防止通知轰炸。

**统一常量**：`SIGNIFICANCE_EMOJI` 和 `CHANGE_TYPE_LABEL` 从 `utils.py` 导入，消除重复。

### 4.3 TrackingScheduler（scheduler.py）

**APScheduler 定时调度器**：

```
 FastAPI lifespan 启动
         │
         ▼
 TrackingScheduler.start()
         │
         ▼
 加载所有 tracking_enabled 的主题
         │
         ▼
 为每个主题添加 CronTrigger 定时任务
         │
         │   ... 等待 cron 触发 ...
         ▼
 APScheduler cron 触发 (e.g. "0 9 * * 1-5" 工作日 9 点)
         │
         ▼
 _run_tracking(topic_id) ← async 函数
         │
         ▼
 asyncio.to_thread() ← 避免阻塞事件循环
         │
         ▼
 create_tracking_graph().stream()
 （使用追踪图，内置 diff_analyzer + change_notifier）
         │
         ▼
 结果自动保存到知识库 + 变化检测 + 通知
```

**关键设计**：
- `_run_tracking` 是 async 函数，但 `graph.stream()` 是同步阻塞的，用 `asyncio.to_thread()` 包裹
- 使用 `create_tracking_graph()` 而非 `create_knowledge_research_graph()`，让内置节点处理变化分析和通知
- 通知策略统一由 `change_notifier` 节点决定

---

## 五、评估层架构

### 5.1 Dataset（dataset.py）

**Langfuse 测试数据集**：8 个研究问题 + 预期要点

**幂等性**：`create_dataset()` 检查已有 items，跳过已存在的，防止重复调用创建重复数据。

### 5.2 Judge（judge.py）

**LLM-as-Judge 自动评分**：

| 维度 | 评分范围 | 说明 |
|------|---------|------|
| relevance | 1-5 | 报告是否紧扣研究问题 |
| completeness | 1-5 | 预期要点是否被覆盖 |
| accuracy | 1-5 | 论点是否有来源支撑 |

**分数验证**：范围 1-5 检查 + 维度完整性检查 + `parse_failed` 标记。

### 5.3 Prompts（prompts.py）

**Prompt 版本管理**：从 Langfuse 拉取最新 prompt，fallback 到本地硬编码

**注册逻辑**：
1. 先 `get_prompt()` 获取已有 prompt
2. 内容有变化 → 创建新版本
3. 不存在 → 创建新 prompt
4. 未变化 → 跳过

**注册的 prompt**：planner、planner-incremental、synthesizer、synthesizer-incremental、synthesizer-refine、reflector

---

## 六、共享工具层（utils.py）

消除各模块重复代码的统一工具集：

| 函数/常量 | 作用 | 原重复次数 |
|-----------|------|-----------|
| `parse_llm_json(content)` | 剥离 code-fence + json.loads + 错误处理 | 5 |
| `create_llm(streaming=False)` | ChatOpenAI 工厂（统一 model/api_key/base_url/temperature） | 6 |
| `get_prompt_from_langfuse(name, fallback)` | Langfuse Prompt 拉取 + fallback | 3 |
| `SIGNIFICANCE_EMOJI` | `{"high": "🔴", "medium": "🟡", "low": "🟢"}` | 4 |
| `CHANGE_TYPE_LABEL` | `{"new_info": "新增", "update": "更新", "contradiction": "⚠️ 矛盾"}` | 2 |
| `summarize_changes(changes)` | 生成变更摘要文本 | 2 |
| `normalize_url(url)` | URL 规范化（去协议 + 去末尾斜杠） | 2 |
| `stream_and_accumulate(graph, input_data, config)` | 流式执行图 + 累积最终状态 | 5 |

---

## 七、API 层架构

### 7.1 FastAPI 应用

```
 应用启动 ─────► TrackingScheduler.start() ─────► FastAPI 运行中 ─────► 应用停止 ─────► TrackingScheduler.stop()
 (lifespan)       加载定时任务                   处理 HTTP/SSE 请求      (lifespan)       关闭调度器
```

### 7.2 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/research` | 同步研究（无知识层） |
| POST/GET | `/research/stream` | SSE 流式研究 |
| POST | `/research/knowledge` | 知识研究（增量） |
| POST | `/research/knowledge/stream` | 知识研究 SSE |
| POST | `/topics` | 创建主题 |
| GET | `/topics` | 列出主题 |
| GET | `/topics/{id}` | 获取主题 |
| PUT | `/topics/{id}` | 更新主题 |
| DELETE | `/topics/{id}` | 删除主题 |
| GET | `/topics/{id}/reports` | 列出报告 |
| GET | `/reports/{id}` | 获取报告 |
| DELETE | `/reports/{id}` | 删除报告 |
| POST | `/tracking/run` | 手动触发追踪 |
| GET | `/tracking/jobs` | 列出追踪任务 |
| GET | `/topics/{id}/tracking-logs` | 追踪记录 |
| GET | `/tracking-logs/{id}/changes` | 变更条目 |
| POST | `/tracking/test-notification` | 测试通知 |
| GET | `/health` | 健康检查 |

### 7.3 SSE 流式输出

**架构**：`asyncio.Queue + asyncio.to_thread`

```
 客户端 (EventSource)
     │
     │ SSE 连接
     ▼
 EventSourceResponse (_event_generator)
     │
     │ async inner()
     ▼
 asyncio.Queue  ◄──────────────────────────────┐
     │                                          │
     │ queue.get()                              │ queue.put_nowait()
     ▼                                          │
 asyncio.to_thread(_run_graph)                  │
     │                                          │
     │ 在线程中执行                               │
     ▼                                          │
 graph.stream() ──── progress ──────────────────┤
 (同步)         ──── message ───────────────────┤
                ──── report ────────────────────┤
                ──── done ──────────────────────┤
                ──── error ─────────────────────┘
```

**事件类型**：
- `progress`：节点进度（含结构化 detail）
- `message`：节点内详细消息
- `report`：最终报告
- `done`：研究完成
- `error`：错误

---

## 八、配置层（config.py）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATA_DIR` | `./data` | SQLite + ChromaDB 持久化目录 |
| `OPENAI_API_KEY` | `""` | LLM API Key |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | API 中转站地址 |
| `OPENAI_MODEL` | `gpt-4o` | 模型名称 |
| `LANGFUSE_PUBLIC_KEY` | `""` | Langfuse 公钥 |
| `LANGFUSE_SECRET_KEY` | `""` | Langfuse 私钥 |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse 地址 |
| `MAX_SEARCH_RESULTS` | `5` | 每次搜索最大结果数 |
| `MAX_REFLECTION_ROUNDS` | `2` | 最大反思轮次 |
| `TAVILY_API_KEY` | `""` | Tavily 搜索 API Key |
| `NOTIFICATION_WEBHOOK_URL` | `""` | 通知 Webhook URL |

**副作用**：`load_dotenv()` 在 import 时调用，任何导入 config 的模块都会触发 .env 加载。

---

## 九、完整数据流

### 9.1 全新研究流程

```
 用户: "Python GIL 是什么？"
  │
  ▼
 ┌──────────┐    LLM 拆解为 3 个子问题
 │ planner  ├────────────────────────────┐
 └──────────┘                            │
  │                                      ▼
  │                               ┌──────────────────────────────────┐
  │                               │ sub_questions:                    │
  │                               │ 1. "GIL 的定义和原理"              │
  │                               │    search: "Python GIL definition"│
  │                               │ 2. "GIL 对性能的影响"              │
  │                               │    search: "Python GIL impact"    │
  │                               │ 3. "绕过 GIL 的方法"              │
  │                               │    search: "Python bypass GIL"    │
  │                               └──────────────────────────────────┘
  │
  ▼
 ┌──────────┐    ThreadPoolExecutor 并行搜索
 │ searcher ├────────────────────────────┐
 └──────────┘                            │
  │                                      ▼
  │                               ┌──────────────────────────────────┐
  │                               │ Tavily API × 3 (并行)            │
  │                               │ → 15 条搜索结果（每个子问题 5 条） │
  │                               └──────────────────────────────────┘
  │
  ▼
 ┌──────────┐    检查每个子问题 ≥ 2 条有效结果
 │validator │ ──→ 全部充足 ✅
 └──────────┘
  │
  ▼
 ┌───────────┐   LLM 流式生成结构化报告
 │synthesizer│ ──→ "# Python GIL 研究\n## 概述\n..."
 └───────────┘
  │
  ▼
 ┌──────────┐    LLM 自评
 │reflector │ ──→ completeness=4, accuracy=4, clarity=4, total=12/15
 └──────────┘    pass: true ✅
  │
  ▼
 输出: 报告 + 来源引用 + 置信度
```

### 9.2 循环修正流程

```
 ┌──────────┐
 │reflector │ 评分 9/15, pass: false ⚠️
 └────┬─────┘
      │ supplement_queries: ["Python GIL PEP 703", "Python free-threaded"]
      ▼
 ┌──────────┐
 │ searcher │ 补充搜索 2 个查询
 └────┬─────┘
      │
      ▼
 ┌───────────┐
 │synthesizer│ 根据反馈重新生成报告
 └────┬──────┘
      │
      ▼
 ┌──────────┐
 │reflector │ 评分 13/15, pass: true ✅
 └────┬─────┘
      │
      ▼
  输出改进后的报告

  ┊ (最多循环 MAX_REFLECTION_ROUNDS=2 次，防止无限循环)
```

### 9.3 增量研究流程

```
 用户: "Python GIL 最新进展" (topic_id=xxx, is_incremental=true)
  │
  ▼
 ┌─────────────────┐
 │knowledge_lookup │ ──→ 查询知识库
 └─────────────────┘    │
      │                 ├── 向量检索: 5 chunks + 10 facts
      │                 └── 结构化查表: 最新报告 + 增量链
      │
      ▼ knowledge_context
 ┌──────────┐
 │ planner  │ 增量模式: 只规划缺失的 2-3 个子问题
 └────┬─────┘
      │
      ▼
 ┌──────────┐
 │ searcher │ 搜索 + 增量去重（过滤已知 URL）
 └────┬─────┘
      │
      ▼
 ┌───────────┐
 │synthesizer│ 增量综合: 在已有知识上补充更新
 └────┬──────┘    标注 🆕 新增 / ⚠️ 矛盾
      │
      ▼
 ┌──────────┐
 │reflector │ 评估通过 ✅
 └────┬─────┘
      │
      ▼
 ┌────────────────┐
 │knowledge_store │ 保存到知识库
 └────────────────┘
      │                ├── SQLite: 元数据 + 关键事实
      │                └── ChromaDB: 报告分块 + 事实向量
      ▼
 输出: 更新后的报告
```

### 9.4 定时追踪流程

```
 APScheduler cron 触发 (e.g. 工作日 9 点)
      │
      ▼
 _run_tracking(topic_id) ← async 函数
      │
      ▼
 asyncio.to_thread() ← 避免阻塞事件循环
      │
      ▼
 create_tracking_graph().stream()
      │
      ▼
 knowledge_lookup → planner → searcher → validator → synthesizer → reflector
      │
      │ (通过后)
      ▼
 knowledge_store → diff_analyzer → change_notifier
                       │                │
                       ▼                ▼
                  对比新旧报告     通知策略判断
                  3 项变化         high > 0 → 需要通知 ✅
                  (1🔴 2🟡)            │
                                       ▼
                                  检测 Webhook 类型
                                       │
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                           企业微信   钉钉    通用 JSON
```

### 9.5 人机交互流程

```
 用户 ──► 输入研究问题
              │
              ▼
         planner 执行
              │
              ▼
         ⏸ 暂停 (interrupt_before=searcher)
         MemorySaver 保存状态
              │
              ▼
         展示子问题，等待用户确认
              │
         用户调整子问题 (或直接确认)
              │
              ▼
         Command(resume={"sub_questions": [...]})
              │
              ▼
         searcher → validator → synthesizer
              │
              ▼
         ⏸ 暂停 (interrupt_before=reflector)
         MemorySaver 保存状态
              │
              ▼
         展示报告，等待用户反馈
              │
         用户补充要求 (或直接确认)
              │
              ▼
         Command(resume={"user_feedback": "补充..."})
              │
              ▼
         reflector → END
              │
              ▼
         输出最终报告
```

### 9.6 通知策略决策流程

```
 检测到变更列表
      │
      ▼
 ┌──────────────────┐
 │ 有 high 级别变化？ │
 └────────┬─────────┘
     是 ──┤── 否
      │         │
      ▼         ▼
 发送通知   ┌────────────────────┐
      │    │ 有 ≥ 2 条 medium 变化？│
      │    └─────────┬──────────┘
      │          是 ──┤── 否
      │           │         │
      │           ▼         ▼
      │       发送通知    跳过通知
      │           │     (只有 low 变化)
      │           │
      ▼           ▼
 ┌──────────────────┐
 │ 主题在冷却期内？   │
 │ (12 小时)        │
 └────────┬─────────┘
     否 ──┤── 是
      │         │
      ▼         ▼
 检测 Webhook   跳过通知
 类型           (冷却中)
      │
      ├─► qyapi.weixin.qq.com  ──► 企业微信格式 (Markdown + **加粗**)
      │
      ├─► oapi.dingtalk.com   ──► 钉钉格式 (Markdown 无加粗)
      │
      └─► 其他                ──► 通用 JSON 格式
```

### 9.7 Langfuse 可观测性流程

```
 graph.stream()
      │
      │ CallbackHandler 自动记录
      ▼
 Langfuse
  │
  ├── Trace (每次研究执行)
  │     │
  │     ├── Span: planner
  │     │     └── Generation: LLM 调用详情 (model, tokens, latency)
  │     │
  │     ├── Span: searcher
  │     │
  │     ├── Span: synthesizer
  │     │     └── Generation: LLM 调用详情
  │     │
  │     └── Span: reflector
  │           └── Generation: LLM 调用详情
  │
  ├── Score: LLM-as-Judge 评分
  │     ├── relevance: 1-5
  │     ├── completeness: 1-5
  │     ├── accuracy: 1-5
  │     └── total: 3-15
  │
  └── Prompt 管理 (版本控制)
        ├── research-buddy-planner
        ├── research-buddy-planner-incremental
        ├── research-buddy-synthesizer
        ├── research-buddy-synthesizer-incremental
        ├── research-buddy-synthesizer-refine
        └── research-buddy-reflector
```

---

## 十、涉及的核心知识点

### 10.1 LangGraph 核心概念

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **StateGraph** | 定义工作流图，节点函数读写 ResearchState | graph.py |
| **Node** | 每个研究步骤是一个节点函数（planner, searcher, ...） | nodes/*.py |
| **State** | TypedDict + Annotated[list, operator.add] 追加语义 | state.py |
| **Conditional Edge** | reflector → END 或 searcher（基于反思结果） | graph.py |
| **Loop** | 反思不通过 → 补充搜索 → 重新综合（最多 N 轮） | graph.py |
| **interrupt_before/after** | HITL 图在 searcher 和 reflector 前暂停 | graph.py |
| **MemorySaver** | HITL 图的 Checkpointer，保存中断状态 | graph.py |
| **Command(resume=)** | 恢复中断的图执行 | graph.py |
| **graph.stream()** | 流式执行，逐节点返回状态更新 | graph.py |

### 10.2 Langfuse 核心概念

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **CallbackHandler** | 接入 LangGraph，自动记录 Trace/Span/Generation | graph.py |
| **Trace** | 每次研究执行是一个 Trace | 自动 |
| **Span** | 每个节点是一个 Span | 自动 |
| **Generation** | 每次 LLM 调用是一个 Generation | 自动 |
| **Score** | LLM-as-Judge 评分写入 Trace | eval/judge.py |
| **Dataset** | 测试数据集（研究问题 + 预期要点） | eval/dataset.py |
| **Prompt Management** | 各节点 prompt 在 Langfuse 中版本管理 | eval/prompts.py |

### 10.3 Python 并发与异步

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **ThreadPoolExecutor** | 并行搜索多个子问题（I/O 密集型） | nodes/searcher.py |
| **asyncio.to_thread()** | 在 async 函数中运行同步阻塞的 graph.stream() | tracking/scheduler.py, api.py |
| **asyncio.Queue** | SSE 事件推送（线程 → 异步生成器） | api.py |
| **threading.local()** | SQLite 线程本地连接 | knowledge/db.py |
| **AsyncIOScheduler** | APScheduler 异步调度器 | tracking/scheduler.py |

### 10.4 向量检索

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **ChromaDB** | 持久化向量存储 + 语义检索 | knowledge/vector.py |
| **Sentence Transformer** | paraphrase-multilingual-MiniLM-L12-v2（多语言 Embedding） | knowledge/vector.py |
| **Cosine Distance** | HNSW 索引 + cosine 距离 + max_distance 过滤 | knowledge/vector.py |
| **Text Chunking** | 按段落分割 + chunk_size=500 + chunk_overlap=100 | knowledge/vector.py |

### 10.5 FastAPI + SSE

| 概念 | 本项目用法 | 文件 |
|------|-----------|------|
| **lifespan** | 应用启动/停止时管理调度器 | api.py |
| **EventSourceResponse** | SSE 流式响应 | api.py |
| **StaticFiles** | 挂载 Web UI | api.py |
| **Pydantic BaseModel** | 请求/响应模型 | api.py |

### 10.6 设计模式

| 模式 | 本项目用法 | 文件 |
|------|-----------|------|
| **Facade** | KnowledgeStore 统一门面，隐藏 db + vector | knowledge/store.py |
| **Singleton (Lazy)** | get_db(), get_vector_store(), get_knowledge_store(), get_notifier(), get_scheduler() | 各模块 |
| **Factory** | create_llm() 工厂函数，统一 LLM 实例化 | utils.py |
| **Strategy** | Notifier 根据 URL 自动选择 payload 格式 | tracking/notifier.py |
| **Template Method** | _get_prompt_from_langfuse() 统一 prompt 获取策略 | utils.py |
| **Observer** | Langfuse CallbackHandler 自动观察 LLM 调用 | graph.py |

---

## 十一、部署架构

### 11.1 整体系统架构

```
 ┌──────────────────────────── 客户端 ────────────────────────────┐
 │                                                                │
 │   Web UI (index.html)            API 客户端 (curl / SDK)      │
 │                                                                │
 └──────┬──────────────────────────────┬──────────────────────────┘
        │ SSE                          │ HTTP
        ▼                              ▼
 ┌──────────────────────── FastAPI 服务 ──────────────────────────┐
 │                                                                │
 │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
 │  │HTTP 端点      │  │SSE 流式端点   │  │静态文件      │        │
 │  │research      │  │research/     │  │Web UI        │        │
 │  │topics        │  │stream        │  │              │        │
 │  │tracking      │  │              │  │              │        │
 │  └──────┬───────┘  └──────┬───────┘  └──────────────┘        │
 └─────────┼─────────────────┼──────────────────────────────────┘
           │                 │
           ▼                 ▼
 ┌───────────────────── LangGraph 工作流 ───────────────────────┐
 │                                                              │
 │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐  │
 │  │核心研究图        │  │知识研究图        │  │追踪图       │  │
 │  │planner          │  │+ knowledge_     │  │+ diff_     │  │
 │  │searcher         │  │  lookup/store   │  │  analyzer   │  │
 │  │validator        │  │                 │  │+ change_   │  │
 │  │synthesizer      │  │                 │  │  notifier   │  │
 │  │reflector        │  │                 │  │             │  │
 │  └────────┬────────┘  └────────┬────────┘  └──────┬─────┘  │
 └───────────┼────────────────────┼──────────────────┼────────┘
             │                    │                   │
     ┌───────┴──────┐     ┌──────┴──────┐     ┌──────┴──────┐
     ▼              ▼     ▼             ▼     ▼             ▼
 ┌────────┐  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
 │LLM API │  │Tavily  │ │SQLite  │ │ChromaDB│ │Langfuse│ │Webhook │
 │(中转站) │  │搜索    │ │知识库  │ │向量库  │ │可观测性│ │通知    │
 └────────┘  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

### 11.2 本地开发

```bash
# 方式 1: dev.sh 脚本
./scripts/dev.sh start    # 启动（自动杀旧进程）
./scripts/dev.sh stop     # 停止
./scripts/dev.sh status   # 查看状态

# 方式 2: 直接运行
uv run python scripts/run_api.py

# 方式 3: CLI
uv run research-buddy
```

### 11.3 Docker 部署

```bash
docker build -t research-buddy .
docker run -p 8000:8000 --env-file .env research-buddy
```

### 11.4 运行时目录

```
data/                        # 运行时数据（.gitignore 已排除）
├── research_buddy.db        # SQLite 数据库
└── chroma_db/               # ChromaDB 持久化
```

---

## 十二、测试体系

| 测试文件 | 覆盖内容 | 测试数量 |
|---------|---------|---------|
| test_routing.py | 路由函数 should_continue / should_continue_to_store | 8 |
| test_state.py | State TypedDict 字段定义 + operator.add 语义 | 4 |
| test_graph.py | 4 种图编译 + 节点拓扑验证 | 8 |
| test_diff.py | DiffAnalyzer 公共 analyze() + 私有方法 + fallback | 12 |

**端到端测试脚本**：
- `scripts/test_phase1.py`：线性图
- `scripts/test_phase2.py`：条件分支 + 循环修正
- `scripts/test_phase3.py`：人机交互

---

## 十三、环境变量一览

| 变量 | 必需 | 说明 |
|------|------|------|
| `OPENAI_API_KEY` | ✅ | LLM API Key |
| `OPENAI_API_BASE` | ✅ | API 中转站地址 |
| `OPENAI_MODEL` | 可选 | 模型名称（默认 gpt-4o） |
| `TAVILY_API_KEY` | ✅ | Tavily 搜索 API Key |
| `LANGFUSE_PUBLIC_KEY` | 可选 | Langfuse 公钥（不配置则无 Trace） |
| `LANGFUSE_SECRET_KEY` | 可选 | Langfuse 私钥 |
| `LANGFUSE_HOST` | 可选 | Langfuse 地址 |
| `DATA_DIR` | 可选 | 数据目录（默认 ./data） |
| `MAX_SEARCH_RESULTS` | 可选 | 每次搜索最大结果数（默认 5） |
| `MAX_REFLECTION_ROUNDS` | 可选 | 最大反思轮次（默认 2） |
| `NOTIFICATION_WEBHOOK_URL` | 可选 | 通知 Webhook URL |
