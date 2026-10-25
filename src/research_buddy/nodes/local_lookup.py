"""研究探索节点 — 自主探索（agentic ReAct loop）+ 单轮启发式 fallback。

agentic 模式（默认）统一接管研究探索，LLM 自主决定：
- 搜什么（web 搜索，迭代查询）
- 抓哪些网页全文（弥补 Tavily 只有摘要的短板）
- 读哪些本地文件（代码库）
- 执行哪些只读操作（可选）

产出两类证据，统一进 local_evidence（source_type 区分 web/local），在编辑简报
和写作前与 Tavily 搜索结果统一编号合并。探索失败/不可用都返回空，绝不阻塞主流程。
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import StreamWriter

from research_buddy.config import (
    FETCH_MAX_PAGES,
    LOCAL_AGENTIC_ENABLED,
    LOCAL_AGENTIC_EXEC_ENABLED,
    LOCAL_AGENTIC_MAX_STEPS,
    LOCAL_AGENTIC_SHELL_ENABLED,
    LOCAL_EXCERPT_CHARS,
)
from research_buddy.state import ResearchState
from research_buddy.tools import local as local_tools
from research_buddy.tools import web as web_tools
from research_buddy.tools.search import search as tavily_search
from research_buddy.utils import create_llm, get_prompt_from_langfuse, invoke_llm, parse_llm_json

logger = logging.getLogger(__name__)

LOCAL_LOOKUP_PROMPT = """你是代码库检索规划专家。研究问题需要基于本地代码/文件作答，请规划要检索的内容。

## 研究问题
{question}

## 子问题
{sub_questions}

## 项目文件树（前 {file_count} 个文本文件）
{file_list}

## 任务
判断哪些本地文件/关键词与研究问题相关，返回检索计划。规则：
- keywords：用于 grep 的关键词/符号（函数名、类名、配置项、错误码等），最多 8 个
- file_suffixes：优先检索的文件扩展名（如 .py/.ts/.md），最多 5 个；不确定就留空
- candidate_paths：值得直接 read 的关键文件相对路径（如入口文件、README、配置文件），最多 6 个，必须是文件树里存在的路径
- 如果研究问题与本地代码无关（如纯政策/新闻/历史话题），返回空列表

只返回 JSON 对象：
```json
{{
  "keywords": ["关键词1", "关键词2"],
  "file_suffixes": [".py", ".md"],
  "candidate_paths": ["README.md", "src/main.py"]
}}
```"""

# agentic 探索的 system prompt：统一的研究探索 agent（web + 本地）
AGENTIC_SYSTEM_PROMPT = """你是一个自主研究 agent。你的任务是**主动探索并收集足够证据**来回答下面的研究问题，供后续写作者基于真实材料撰写高质量文章。

## 研究问题
{question}

## 调查方向（规划阶段拆解出的子问题）
{sub_questions}

## 可用的探索工具
- web_search(query)：搜索互联网，返回相关网页的标题、URL 和内容摘要
- fetch_webpage(url)：抓取指定网页的**全文**并提取正文（摘要不够时用它拿到完整内容）
- list_project_files()：列出本地项目文件树
- search_code(pattern, file_types)：在本地代码内容里按关键词搜索
- read_local_file(path, offset, limit)：读取本地文件的一段内容
{shell_tool}
{exec_tool}

## 探索策略
1. 先 web_search 了解主题，找到权威一手来源（官方文档、论文、官网、行业报告）；
2. 对最重要的来源用 fetch_webpage 抓全文，拿到完整、准确的事实，而不是只依赖摘要；
3. 如果问题涉及某个具体代码库/项目，用 list_project_files + search_code + read_local_file 读真实代码；
4. 交叉验证：同一事实尽量有多个独立来源；发现矛盾要记下；
5. 迭代：根据已有发现，继续搜缺失的关键点，直到能完整回答研究问题。

## 硬性约束
- 只报告「确实读到/抓到」的内容，不要凭常识补写没看到的事实；
- fetch_webpage 只抓权威、相关的网页，不要抓导航页、列表页、登录页；
- 不要读 node_modules、.venv、dist、build 等依赖/构建产物（list 已自动过滤）；
- 当你已收集到足够证据、能完整回答研究问题时，停止调用工具，输出一段简短结论；
- 探索步数有限，优先抓最关键的来源，不要试图穷尽整个互联网。

现在开始探索。先搜主题、找到关键来源，再抓全文精读。"""

AGENTIC_SUMMARY_PROMPT = """根据你刚才的探索记录，总结你收集到的关键证据，用于后续写作。

## 研究问题
{question}

## 探索记录（你搜过的查询、抓过的网页、读过的文件）
{exploration_log}

请用中文简要总结（300 字以内）：
1. 与研究问题最相关的核心事实/结论是什么
2. 你确认了哪些关键证据（附来源 URL 或文件路径）
3. 还有哪些关键点因为没读到而不确定（如果有）

只返回总结正文，不要 Markdown 标题。"""


def _format_sub_questions(sub_questions: list[dict]) -> str:
    if not sub_questions:
        return "（无子问题）"
    return "\n".join(
        f"- {sq.get('question', '')}" for sq in sub_questions if sq.get("question")
    )


def _plan_lookup(question: str, sub_questions: list[dict],
                 file_list: list[str], config: RunnableConfig | None = None) -> dict:
    """单轮模式：让 LLM 决定检索关键词、文件类型和候选路径。失败返回空计划。"""
    try:
        prompt = get_prompt_from_langfuse(
            "research-buddy-local-lookup", LOCAL_LOOKUP_PROMPT,
            question=question,
            sub_questions=_format_sub_questions(sub_questions),
            file_count=len(file_list),
            file_list="\n".join(file_list[:300]),
        )
        response = invoke_llm(create_llm(), prompt, config=config)
        plan = parse_llm_json(response.content)
        if not isinstance(plan, dict):
            return {}
        return plan
    except Exception as exc:
        logger.warning("本地检索规划失败，返回空计划: %s", exc)
        return {}


def _build_web_evidence(url: str, title: str, content: str, query: str) -> dict:
    """把抓取的网页全文封装成 web 证据条目。"""
    return {
        "url": url,
        "title": title or url,
        "content": content[:LOCAL_EXCERPT_CHARS],
        "score": 1.0,
        "sub_question_id": "",
        "sub_question": "",
        "query": query,
        "source_type": "web",
        "path": "",
    }


def _build_local_evidence(path: str, content: str, query: str) -> dict:
    """把一段本地文件内容封装成本地证据条目。"""
    return {
        "url": "",
        "title": f"本地文件：{path}",
        "content": content[:LOCAL_EXCERPT_CHARS],
        "score": 1.0,
        "sub_question_id": "",
        "sub_question": "",
        "query": query,
        "source_type": "local",
        "path": path,
    }


# ── agentic ReAct loop ──────────────────────────────────

class _Collector:
    """线程安全的探索状态收集器（并发执行 tool_calls 时共享）。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.evidence: list[dict] = []
        self.exploration_log: list[str] = []
        self.fetched_pages = 0
        # path → 该文件累计读到的内容（用于同文件多次读取的合并）
        self._local_contents: dict[str, str] = {}
        # path → 该文件在 evidence 里的下标（首次出现时记录）
        self._local_index: dict[str, int] = {}

    def add_evidence(self, item: dict) -> None:
        with self._lock:
            self.evidence.append(item)

    def add_log(self, entry: str) -> None:
        with self._lock:
            self.exploration_log.append(entry)

    def try_reserve_fetch(self) -> bool:
        """原子地尝试占一个抓取名额；达到上限返回 False。"""
        with self._lock:
            if self.fetched_pages >= FETCH_MAX_PAGES:
                return False
            self.fetched_pages += 1
            return True

    def merge_local_file(self, path: str, content: str, query: str) -> None:
        """合并同一本地文件的内容：首次读入建条目，后续读取追加内容到同一条证据。

        避免同一文件读多个片段时产生多条碎片证据（之前 read 3 次 local.py
        就产生 3 条证据），让 agent 最终拿到的是「该文件最完整的一份内容」。
        """
        with self._lock:
            if path in self._local_index:
                idx = self._local_index[path]
                existing = self.evidence[idx]
                # 追加新片段（去重：若新片段是已有内容的子串则跳过）
                new_content = content
                if new_content not in existing["content"]:
                    existing["content"] = (existing["content"] + "\n" + new_content)[:LOCAL_EXCERPT_CHARS]
                return
            item = _build_local_evidence(path, content, query)
            self._local_index[path] = len(self.evidence)
            self.evidence.append(item)


def _make_agent_tools(collector: _Collector):
    """构建 agent 探索工具集（web 搜索 + 抓全文 + 本地文件 + 只读 shell + 执行验证）。

    读取类工具会把内容积累进 collector；fetch_webpage 受 FETCH_MAX_PAGES 上限约束，
    并发调用时通过 collector.try_reserve_fetch() 保证不超限。
    """

    @tool
    def web_search(query: str) -> str:
        """搜索互联网，返回相关网页的标题、URL 和内容摘要。

        Args:
            query: 搜索关键词（用目标来源的语言）
        """
        try:
            results = tavily_search(query, max_results=5)
        except Exception as exc:
            return f"搜索失败：{exc}"
        if not results:
            return f"没有搜索结果：{query}"
        lines = []
        for i, r in enumerate(results, 1):
            content = str(r.get("content", ""))[:200]
            lines.append(f"[{i}] {r.get('title', '')}\n    URL: {r.get('url', '')}\n    {content}")
        return "\n".join(lines)

    @tool
    def fetch_webpage(url: str) -> str:
        """抓取指定网页的全文并提取正文。摘要不够时用它拿到完整、准确的内容。

        Args:
            url: 目标网页 URL（http/https）
        """
        if not collector.try_reserve_fetch():
            return f"已抓取 {collector.fetched_pages} 个网页，达到本次上限，请基于已有证据继续。"
        page = web_tools.fetch_webpage(url)
        if not page:
            return f"抓取失败：{url}"
        collector.add_evidence(_build_web_evidence(page["url"], page["title"], page["text"], f"fetch {url}"))
        collector.add_log(f"[fetch] {url}")
        return f"标题：{page['title']}\n正文（前 4000 字）：\n{page['text'][:4000]}"

    @tool
    def list_project_files() -> str:
        """列出本地项目文件树（相对路径，已过滤依赖/构建产物）。"""
        files = local_tools.list_files(max_files=300)
        if not files:
            return "（没有可列出的文件）"
        return "\n".join(files[:300])

    @tool
    def search_code(pattern: str, file_types: str = "") -> str:
        """在本地代码内容里按关键词/正则搜索，返回命中行（文件路径 + 行号 + 内容）。

        Args:
            pattern: 要搜索的关键词或正则（大小写不敏感）
            file_types: 可选，逗号分隔的文件扩展名，如 ".py,.js"；留空搜所有文本文件
        """
        globs = [s.strip() for s in file_types.split(",") if s.strip()] or None
        hits = local_tools.grep(pattern, file_globs=globs, max_results=30)
        if not hits:
            return f"没有命中 '{pattern}'"
        lines = [f"{h['path']}:{h['line']}: {h['text']}" for h in hits]
        return "\n".join(lines)

    @tool
    def read_local_file(path: str, offset: int = 1, limit: int = 400) -> str:
        """读取本地文件的一段内容（带行号），用于精读关键代码。

        Args:
            path: 相对项目根目录的文件路径
            offset: 起始行号（1-based，默认 1）
            limit: 读取行数（默认 400；文件更长时结果会标注续读 offset，可分段读完）
        """
        data = local_tools.read_file(path, offset=offset, limit=limit)
        if not data:
            return f"无法读取 {path}（不存在、是二进制文件或超过大小上限）"
        # 同文件多次读取会合并成一条证据（内容按 path 累积），避免碎片化
        collector.merge_local_file(path, data["content"], f"read {path}:{offset}")
        collector.add_log(f"[read] {path} L{data['line_start']}-L{data['line_end']}")
        header = (f"{path}（行 {data['line_start']}-{data['line_end']}，"
                  f"共 {data['total_lines']} 行）")
        if data["truncated"]:
            header += f" [已截断，用 offset={data['line_end'] + 1} 续读剩余部分]"
        return f"{header}\n{data['content']}"

    tools = [web_search, fetch_webpage, list_project_files, search_code, read_local_file]

    if LOCAL_AGENTIC_SHELL_ENABLED:
        @tool
        def run_readonly_command(command: str) -> str:
            """执行一条只读 shell 命令（仅白名单：git log/show/diff/status/blame、find、ls、tree、wc、cat、grep 等）。

            Args:
                command: 单条命令，禁止管道/重定向/分号分隔
            """
            result = local_tools.run_readonly_shell(command)
            if result["ok"]:
                return result["output"] or "（命令执行成功，无输出）"
            return f"命令失败：{result['error']}"

        tools.append(run_readonly_command)

    if LOCAL_AGENTIC_EXEC_ENABLED:
        @tool
        def run_verification(command: str) -> str:
            """执行一条「验证类」命令，用于验证你写的内容是否真实可运行。

            典型用途：
            - python -c "代码片段" / node -e "代码片段"：验证示例代码能否运行
            - pytest --collect-only：只收集测试结构（不实际跑测试）
            - pip show 包名 / 包名 --version：查真实版本号，避免写错
            - 编译/运行一段最小复现，确认 API 用法正确

            限制：只能跑白名单程序（python/node/pytest/pip/npm/uv/go 等），
            禁止 install/build/commit 等会修改环境的操作，禁止管道/重定向。

            Args:
                command: 单条命令（如 "python -c \"print(1+1)\""）
            """
            result = local_tools.run_exec_verification(command)
            if result["ok"]:
                return result["output"] or "（执行成功，无输出）"
            return f"执行失败（{result['error']}）：\n{result['output']}"

        tools.append(run_verification)

    return tools


def _compact_history(messages: list, keep_recent: int = 6,
                     max_tool_chars: int = 3000) -> list:
    """压缩 agentic 对话历史，防止长探索撑爆上下文窗口。

    策略（保守，不破坏消息结构）：
    - 只截断 ToolMessage 的 content（这些是工具返回的大段文本，最占空间）；
    - 保留最近 keep_recent 条消息完整不截断（含 system + 首条 user + 末尾的决策）；
    - ToolMessage 的 tool_call_id 必须保留，否则 LLM 会认为工具调用未配对而报错；
    - SystemMessage / HumanMessage / AIMessage 不动（体积小，截断会破坏语义）。
    """
    if len(messages) <= keep_recent:
        return messages
    boundary = len(messages) - keep_recent
    compacted = []
    for i, msg in enumerate(messages):
        if i < boundary and isinstance(msg, ToolMessage):
            content = str(msg.content)
            if len(content) > max_tool_chars:
                truncated = content[:max_tool_chars] + "\n…（工具输出过长，已截断）"
                compacted.append(ToolMessage(
                    content=truncated,
                    tool_call_id=msg.tool_call_id,
                ))
                continue
        compacted.append(msg)
    return compacted


def _execute_tool_calls(tool_calls, tool_map) -> list[tuple]:
    """并发执行一批 tool_calls，返回 [(tool_call_id, result_str), ...]。

    不同工具相互独立（读文件/搜索/抓网页），并发可显著缩短探索耗时；
    结果按原顺序返回，保证消息结构与 LLM 发出的 tool_calls 顺序一致。
    """
    if not tool_calls:
        return []

    def _run_one(tc):
        name = tc.get("name", "")
        args = tc.get("args", {}) or {}
        fn = tool_map.get(name)
        if fn is None:
            return tc["id"], f"未知工具: {name}"
        try:
            return tc["id"], fn.invoke(args)
        except Exception as exc:
            return tc["id"], f"工具执行失败: {exc}"

    if len(tool_calls) == 1:
        return [_run_one(tool_calls[0])]

    results = []
    with ThreadPoolExecutor(max_workers=min(4, len(tool_calls))) as pool:
        futures = {pool.submit(_run_one, tc): tc for tc in tool_calls}
        for future in as_completed(futures):
            results.append(future.result())
    # 按原始 tool_call 顺序重排
    order = {tc["id"]: i for i, tc in enumerate(tool_calls)}
    results.sort(key=lambda pair: order.get(pair[0], 0))
    return results


def _agentic_lookup(question: str, sub_questions: list[dict],
                    config: RunnableConfig | None = None,
                    writer: StreamWriter | None = None) -> list[dict]:
    """agentic ReAct loop：LLM 自主探索（web + 本地），收集证据。

    每一步 = 一次带工具的 LLM 调用，同一步的多个 tool_calls 并发执行；
    历史超过阈值时压缩旧工具输出，防止上下文膨胀。
    LLM 不再返回 tool_calls 时（认为收集够了）或达到 LOCAL_AGENTIC_MAX_STEPS 时停止。

    每步通过 writer 推送 agent_step 事件，让前端实时看到 agent 在做什么。
    """
    collector = _Collector()
    tools = _make_agent_tools(collector)

    def _emit_step(step_no: int, brief: str) -> None:
        """向前端推一条 agent 步骤进度；writer 为 None（CLI 路径）时静默跳过。"""
        if writer:
            writer({"type": "agent_step", "step": step_no, "brief": brief})

    shell_tool = ("- run_readonly_command(command)：执行只读 shell 命令（git 历史、find、tree 等）"
                  if LOCAL_AGENTIC_SHELL_ENABLED else "")
    exec_tool = ("- run_verification(command)：执行验证类命令（跑代码片段、查版本、收集测试结构），"
                 "用于验证你写的内容是否真实可运行"
                 if LOCAL_AGENTIC_EXEC_ENABLED else "")
    system_prompt = AGENTIC_SYSTEM_PROMPT.format(
        question=question,
        sub_questions=_format_sub_questions(sub_questions),
        shell_tool=shell_tool,
        exec_tool=exec_tool,
    )

    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="请开始探索，收集证据来回答研究问题。"),
    ]

    tool_map = {t.name: t for t in tools}
    # 工具调用的简短可读描述（用于日志，避免把长结果刷进后台）
    def _tool_brief(name: str, args: dict) -> str:
        if name == "web_search":
            return f"搜索「{str(args.get('query', ''))[:60]}」"
        if name == "fetch_webpage":
            return f"抓取全文 {str(args.get('url', ''))[:80]}"
        if name == "read_local_file":
            return f"读文件 {str(args.get('path', ''))[:60]}"
        if name == "search_code":
            return f"搜代码「{str(args.get('pattern', ''))[:60]}」"
        if name == "list_project_files":
            return "列项目文件树"
        if name == "run_readonly_command":
            return f"只读命令 {str(args.get('command', ''))[:60]}"
        if name == "run_verification":
            return f"验证命令 {str(args.get('command', ''))[:60]}"
        return name

    try:
        for step in range(LOCAL_AGENTIC_MAX_STEPS):
            messages = _compact_history(messages)
            response = llm_with_tools.invoke(messages, config=config)
            messages.append(response)

            tool_calls = list(getattr(response, "tool_calls", None) or [])
            if not tool_calls:
                logger.info("agentic 探索在第 %d 步结束，收集到 %d 条证据",
                            step + 1, len(collector.evidence))
                break

            # 过程日志：让后台能看到 agent 每一步自主决策做了什么
            call_briefs = "; ".join(
                _tool_brief(tc.get("name", ""), tc.get("args", {}) or {})
                for tc in tool_calls
            )
            logger.info("agentic 第 %d 步：%s", step + 1, call_briefs)
            _emit_step(step + 1, call_briefs)

            for tool_call_id, result in _execute_tool_calls(tool_calls, tool_map):
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
    except Exception as exc:
        logger.warning("agentic 探索失败，退回空证据: %s", exc)
        return []

    evidence = collector.evidence
    if not evidence:
        return []

    # 用探索记录生成一句证据总结，帮助写作者理解证据与研究问题的关系
    try:
        summary_prompt = get_prompt_from_langfuse(
            "research-buddy-local-summary", AGENTIC_SUMMARY_PROMPT,
            question=question,
            exploration_log="\n".join(collector.exploration_log[:200]),
        )
        summary = invoke_llm(create_llm(), summary_prompt, config=config).content.strip()
    except Exception:
        summary = ""

    if summary:
        evidence.append(_build_local_evidence("探索总结", summary, "agentic-summary"))

    return evidence


# ── 单轮 fallback ───────────────────────────────────────

def _single_pass_lookup(question: str, sub_questions: list[dict],
                        file_list: list[str],
                        config: RunnableConfig | None = None) -> list[dict]:
    """单轮启发式检索：LLM 规划关键词/路径 → grep + 读文件（仅本地）。"""
    plan = _plan_lookup(question, sub_questions, file_list, config=config)
    keywords = [str(k).strip() for k in plan.get("keywords", []) if str(k).strip()][:8]
    suffixes = [str(s).strip() for s in plan.get("file_suffixes", []) if str(s).strip()][:5]
    candidate_paths = [str(p).strip() for p in plan.get("candidate_paths", []) if str(p).strip()][:6]

    if not keywords and not candidate_paths:
        return []

    evidence: list[dict] = []
    seen_paths: set[str] = set()

    for path in candidate_paths:
        if path in seen_paths:
            continue
        file_data = local_tools.read_file(path)
        if not file_data:
            continue
        seen_paths.add(path)
        evidence.append(_build_local_evidence(path, file_data["content"], f"read {path}"))

    for keyword in keywords:
        hits = local_tools.grep(keyword, file_globs=suffixes or None)
        for hit in hits[:8]:
            path = hit["path"]
            if path in seen_paths:
                continue
            seen_paths.add(path)
            ctx = local_tools.read_context(path, hit["line"])
            if not ctx:
                continue
            evidence.append(_build_local_evidence(path, ctx["content"], f"grep {keyword}"))
        if len(evidence) >= 30:
            break

    return evidence


def local_lookup(state: ResearchState, config: RunnableConfig | None = None,
                 *, writer: StreamWriter | None = None) -> dict:
    """研究探索节点：产出 local_evidence（web 全文 + 本地文件证据）。

    优先走 agentic ReAct loop；关闭或异常时退回单轮本地检索。
    返回空 dict（无证据）时不影响主流程。
    """
    if not local_tools.is_enabled():
        logger.info("本地检索未配置（LOCAL_REPO_ROOT 不可用），跳过本地探索")
        return {"local_lookup_done": True, "messages": []}

    question = state.get("question", "")
    sub_questions = state.get("sub_questions", [])

    if LOCAL_AGENTIC_ENABLED:
        evidence = _agentic_lookup(question, sub_questions, config=config, writer=writer)
    else:
        file_list = local_tools.list_files()
        evidence = _single_pass_lookup(question, sub_questions, file_list, config=config)

    if evidence:
        web_count = sum(1 for e in evidence if e.get("source_type") == "web")
        local_count = len(evidence) - web_count
        logger.info("研究探索产出 %d 条证据（web %d，本地 %d）",
                    len(evidence), web_count, local_count)

    return {
        "local_evidence": evidence,
        "local_lookup_done": True,
        "messages": [f"🔍 自主探索完成，得到 {len(evidence)} 条证据（含网页全文与本地文件）"]
        if evidence else ["🔍 自主探索未命中相关内容"],
    }
