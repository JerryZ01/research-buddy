# Research Buddy 学习笔记

记录学习 LangGraph + Langfuse 过程中的关键概念和踩坑经验。

---

## LangGraph 核心概念

### StateGraph
- LangGraph 的核心，一个有向图，节点是处理函数，边是控制流
- 所有节点共享一个 State（TypedDict），节点返回部分更新来修改 State
- 编译后得到 `CompiledGraph`，调用 `.invoke()` 或 `.stream()` 执行

### Node（节点）
- 就是一个函数：`(State) -> partial State`
- 可以是 LLM 调用、工具调用、纯逻辑处理

### Edge（边）
- 普通边：A → B，固定流转
- 条件边：A → router_fn → {C, D}，根据 State 决定走哪条
- `START` 和 `END` 是特殊节点

### State
- 用 TypedDict 定义，是图的全局共享状态
- 节点返回 dict，自动 merge 到 State
- 需要注意：默认是覆盖，列表字段要用 `Annotated[list, operator.add]` 实现追加

### Human-in-the-loop
- `interrupt_before` / `interrupt_after`：在指定节点前/后暂停
- 暂停后用 `Command(resume=...)` 恢复，传入人工输入
- 需要 Checkpointer 来保存暂停时的状态

### Checkpoint
- 保存图的执行进度，支持中断恢复
- 内存版：`MemorySaver`（开发用）
- 持久版：`SqliteSaver` / `PostgresSaver`

---

## Langfuse 核心概念

### Trace
- 一次完整用户请求 = 一个 Trace
- 包含多个 Span，形成调用树

### Span
- Trace 下的子节点，记录一个处理步骤
- 可以嵌套

### Generation
- 记录一次 LLM 调用：prompt、completion、model、token 用量
- 自动通过 CallbackHandler 采集

### Score
- 对 Trace/Span 的评分，可以是人工或 LLM-as-Judge
- 用于评估 Agent 质量

### Dataset
- 测试数据集，包含输入+预期输出
- 用于批量评估 Agent

### Prompt Management
- 在 Langfuse 平台管理 prompt 模板
- 代码中通过 `langfuse.get_prompt()` 拉取，无需改代码即可调 prompt

---

## 踩坑记录

### should_continue 路由函数与 conditional_edges 映射不匹配
- **问题**：`should_continue` 返回 `"end"`，但知识/追踪图的 `add_conditional_edges` 映射只有 `"knowledge_store"` 和 `"search_again"` 两个 key，没有 `"end"`
- **后果**：反思通过时，知识存储节点永远走不到，或触发 LangGraph 路由错误
- **修复**：为知识/追踪图创建独立路由函数 `should_continue_to_store`，返回 `"knowledge_store"` 而非 `"end"`
- **教训**：conditional_edges 的路由函数返回值必须与映射字典的 key 完全匹配，否则运行时报错或路由到错误节点

### run_research_interactive 函数定义行丢失
- **问题**：`graph.py` 中 `run_tracking` 函数的 `return result` 之后，`run_research_interactive` 的函数体悬空，缺少 `def` 行
- **后果**：`scripts/test_phase3.py` 无法 import `run_research_interactive`，Phase 3 测试完全不可用
- **修复**：补回 `def run_research_interactive(question: str) -> dict:` 行
- **教训**：大文件编辑时注意函数边界，避免合并/删除时吞掉 def 行

### pytest 收集 scripts 目录导致测试中断
- **问题**：pytest 默认收集所有 `test_*.py`，把 `scripts/test_phase3.py` 也收集了，而该文件 import 失败导致整个 collection 中断
- **修复**：在 `pyproject.toml` 中添加 `[tool.pytest.ini_options] testpaths = ["tests"]`
- **教训**：非测试脚本不要放在 pytest 默认收集路径，或显式配置 testpaths

---

## Phase 2 学习笔记：条件分支 + 循环修正

### Conditional Edge
- `graph.add_conditional_edges(source_node, router_fn, path_map)` — 根据路由函数返回值决定下一步
- `router_fn` 接收 State，返回字符串（节点名或 END）
- `path_map` 是可选的映射字典：`{返回值: 目标节点}`，不提供时 LangGraph 用返回值直接作为节点名

### Loop（循环修正）
- 通过 conditional edge 实现：reflector → should_continue → searcher（不通过时回到搜索）
- **必须**设置最大循环次数（`MAX_REFLECTION_ROUNDS`），否则可能无限循环
- State 中用 `reflection_round` 计数器追踪轮次

### State 更新策略
- 默认行为是**覆盖**：节点返回 `{"key": value}` 会替换 State 中的 key
- 列表字段用 `Annotated[list, operator.add]` 实现**追加**：多个节点返回的列表会合并
- 清空列表：节点返回空列表 `[]`，配合 `operator.add` 不会清空——需要返回一个特殊标记或在节点内处理

---

## Phase 3 学习笔记：Human-in-the-loop

### interrupt_before / interrupt_after
- 编译图时指定：`graph.compile(checkpointer=memory, interrupt_before=["searcher"])`
- 图执行到指定节点前暂停，保存 Checkpoint
- 暂停后可以查看当前 State，等待人工输入

### Command(resume=)
- 恢复执行：`graph.stream(Command(resume=value), config=config)`
- `resume` 的值会传给被中断的节点
- 可以传 dict 来更新 State，或传空 dict 直接继续

### Checkpoint
- `MemorySaver`：内存版，开发用，进程退出即丢失
- `SqliteSaver` / `PostgresSaver`：持久版，支持中断后跨进程恢复
- 每个 thread_id 维护独立的 Checkpoint 链

---

## Phase 4 学习笔记：Langfuse 评估体系

### Dataset
- 在 Langfuse 平台创建测试数据集：输入（研究问题）+ 预期输出（关键要点）
- 代码中通过 `langfuse.get_dataset()` 拉取
- 用于批量评估 Agent 质量

### LLM-as-Judge
- 用另一个 LLM 评估 Agent 输出质量
- 评分维度：相关性、完整性、准确性
- 评分结果通过 `langfuse.score()` 写回 Trace

### Prompt Management
- 在 Langfuse 平台管理 prompt 模板
- 代码中通过 `langfuse.get_prompt(name)` 拉取最新版本
- 本地 fallback：如果 Langfuse 不可用，使用代码中的默认 prompt
- 好处：调 prompt 不需要改代码和重新部署

---

## Phase 5 学习笔记：生产化

### FastAPI + SSE
- `sse-starlette` 库提供 `EventSourceResponse`，支持 SSE 流式输出
- 事件格式：`event: type\ndata: content\n\n`
- 自定义事件类型：progress、message、report、done、error

### 流式输出
- `graph.stream()` 返回生成器，逐节点输出 State 更新
- API 层用 `asyncio.to_thread()` 包装同步的 `graph.stream()`
- SSE 推送每个节点的进度消息

---

## Phase 6 学习笔记：知识层

### SQLite + ChromaDB 双存储
- SQLite：存储报告元数据、关键事实、追踪记录（结构化查询）
- ChromaDB：存储报告文本分块和关键事实的向量（语义检索）
- 两者通过 `report_id` 关联

### 增量研究模式
- `knowledge_lookup` 节点：查询历史知识，生成上下文
- `is_incremental=True`：planner 根据历史知识调整搜索策略，避免重复搜索
- `knowledge_store` 节点：保存新报告，关联 parent_report_id

---

## Phase 7 学习笔记：定时追踪 + 变化检测

### APScheduler
- `AsyncIOScheduler`：异步定时任务调度
- 支持 cron 表达式：`scheduler.add_job(func, "cron", hour=9, minute=0)`
- 与 FastAPI 共享事件循环

### DiffAnalyzer
- 双模式：LLM 语义分析 + difflib 规则 fallback
- 相似度阈值：`similarity_threshold=0.85`，低于阈值才触发变化分析
- 输出结构化变化列表：新增/删除/修改

### Notifier
- 多平台 Webhook：企业微信、钉钉、Telegram、Generic
- 自动检测 URL 格式选择 payload 模板
- 支持测试通知：`send_test_notification()`
