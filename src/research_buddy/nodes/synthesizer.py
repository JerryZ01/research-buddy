"""综合节点 - 将搜索结果综合为可发布的结构化文章（支持流式输出）

报告 = 纯文章正文（无评价性内容、无内嵌 URL、无 [编号] 引用标注）
       + 文末核心参考文献（LLM 从全部来源中筛选 + 代码重编号生成）。
- 矛盾/不足/置信度等评价性信息写入 research_notes / confidence（state 字段），不进正文
"""

import logging
import time

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from research_buddy.config import MAX_ARTICLE_TOKENS, MAX_IMAGES_IN_ARTICLE, MAX_REFERENCES
from research_buddy.state import ResearchState
from research_buddy.styles import get_style_section
from research_buddy.tools.images import select_images
from research_buddy.utils import (_is_transient_error, create_llm, get_prompt_from_langfuse,
                              invoke_llm, normalize_url, parse_llm_json)

logger = logging.getLogger(__name__)


# ── Prompt 模板（可发布文章风格） ────────────────────────
# 写作规范统一收敛在 WRITING_RULES（单一事实来源），三个 prompt 用
# {writing_rules} 注入——改规则只改一处，避免三处复制导致漂移。

WRITING_RULES = """## 写作要求

### 主线与观点
1. 文章必须有一条贯穿始终的分析主线——一个核心判断或问题视角（例如「GIL 正在退出历史舞台，但真正的瓶颈是 C 扩展生态」）。开篇点明主线，正文围绕它展开，而不是把资料平铺罗列。
2. 每个章节都要有自己的分析结论，而不是转述资料：先立观点，再用证据支撑。

### 深度与读者价值
3. 严禁综述腔（「根据 A 报道…」「B 研究显示…」式的堆砌拼接）。把检索到的证据**内化成自己的分析叙述**：解释背后的原理、动机、因果与影响。
4. **先理解问题意图再决定写什么**：判断问题是「解释原理/机制」「对比」「选型评估」「趋势分析」还是「实操教程」——文章内容和结尾只覆盖问题明确问到的范围。纯原理/机制解析类问题专注把机制讲透，不要擅自展开「代价」「发展趋势」「影响」「该不该用」等超出问题范围的内容；只有当问题本身涉及选择、评估或影响时，才写权衡、适用场景与实践建议。
5. 关键概念给出准确精炼的定义与背景即可，不要花篇幅科普人人皆知的基础知识。

### 可信度与表达
6. 事实与数据必须来自检索证据，保持准确；资料有限或不确定处明确说明（如「截至本文写作时」「现有资料显示」），并区分事实、分析与推测，不做没有依据的断言。
**归因节制（重要）**：允许在关键事实上点名权威来源（如「AWS 官方文档建议」「GitHub 官方博客指出」「PEP 703 提案明确」），但要用得克制——只在不点名来源就缺乏说服力的权威论断、或该来源确实是此观点的出处时点名。大部分论断应当用你自己的分析叙述来表达，不要每个段落都「某公司说/建议/经验」式归因；多个来源结论一致时合并成一句综合陈述，不要逐一罗列「A 公司认为…，B 公司认为…」。全篇点名归因控制在少数几处（平均每千字不超过 1~2 处）。

**去 AI 味（重要）**：
- 不要写元评论/模板句：「本节的关键结论是」「值得注意的是」「综上所述」「这揭示了」
- 不要滥用排比对仗：「不是…而是…」整篇最多 2 处；禁止「从 X 到 Y，从 A 到 B，从 C 到 D」式三连排比；句子长短自然交错
- 不要自问自答（「这意味着什么？因为…」「为什么如此重要？因为它…」），不要用「拆解这个定义需要一点耐心」式引导句铺设
- 不要用 AI 高频套话和生造词：「赋能」「抓手」「闭环」「颗粒度」「底层逻辑」「破圈」「新范式」「重塑」等；不要为了显得专业堆砌华丽词——用最平实的词，逻辑严密比辞藻重要
- 不要平均用力：重点章节写透，次要章节可以一笔带过
- 观点要明确、有取舍，允许明确的态度判断（「在我看来」「这个方案我不太看好」），也允许诚实地写「这块资料有限」「这个结论仍有争议」
- 结尾用散文收束或直接给建议，不要表格化、不要「结论：1. 2. 3.」
- **句式要有节奏差异**：长短句交错，允许短句独立成段，禁止连续两句结构相同（同长度/同「主谓宾」骨架）；段落长短也应有差异，偶尔出现一行段落
- **案例必须具体可查**：禁止「某文档 pipeline」「社区报告显示」这类模糊编造——要么给出真实名称/可查的细节，要么不写
- **各章节结构差异化**：篇幅允许 3:1 的差距；不要每节都是「定义→展开→小结」同一骨架——有的节以具体例子开头、有的以问题开头、有的直接下判断
7. 小标题要有信息量，能看出这一节的观点（如「为什么 I/O 密集场景不受 GIL 影响」），避免「概述」「背景介绍」这类空标题；章节按主题逻辑重组，不要机械地按子问题逐条罗列。**小标题风格要多样**：不要连续用「名词：解释」式冒号标题（如「自注意力：当每个 token 都成为检索者」），同一篇里交替使用陈述句、疑问句、短语式标题，冒号式标题整篇最多 1~2 个。
8. 语气专业、克制、自信；语言自然流畅，像一个有经验的人认真讲清楚一件事，避免套话、模板腔和「综上」「众所周知」式的空泛表达。
9. 论点直接陈述，**不需要**标注引用编号或来源链接（文末参考文献由系统自动生成）。
10. 如果提供了「可用插图」，在内容最相关的位置插入插图（**整篇最多 {image_limit} 张、通常 4~6 张即可**，各章节合理分配）。插图按优先级：① 能说明内容的图（图表/架构图/截图）② 与主题相关的配图 ③ 氛围装饰图（全篇不超过 2 张）。每张图插入前自问：它能帮助读者理解这一段吗？不能就不插——宁可少一张，不放无关的图：`![alt文本](图片URL)`。图片 URL 必须原样来自「可用插图」列表，禁止使用列表之外的图片；alt 文本用「可用插图」里给出的描述。
11. 不要在正文中直接写出任何 URL，也不要出现**研究过程元评论**——包括但不限于：「信息存在矛盾」「证据不足」「研究局限」「本报告基于……搜索」「从检索到的资料看」「有来源提到」「检索到的证据」「多个来源指出」「据资料显示」「公开资料称」等；不要自行添加「参考文献」「来源」「置信度」章节（文末参考文献由系统自动生成）。
12. 使用中文撰写。

### 像真人作者（表达层手法，底盘仍是专业可信）
13. **允许第一人称做判断**：核心判断、取舍与倾向可以用「我的判断是」「我更倾向于」「在我看来」引出，不必全程客观腔；第一人称只用于观点与判断，不要写成个人经历流水账，也不要编造个人体验。
14. **鼓励生活化类比**：用读者熟悉的事物打比方（新员工入职第一天拿到的说明书、爬山的安全带、工具箱里的扳手），一个贴切的比喻胜过一段抽象描述；类比只服务于讲清机制，不要为比喻而比喻。
15. **允许承认不确定**：资料不足或判断没把握时，可以诚实地写「这块资料有限，我的判断是……」「这一点目前没有定论，更可能的解释是……」——真诚的局限说明比生硬的免责声明更可信；但要区分「证据不足的推测」与「事实」，推测不能伪装成结论。
16. **开篇多样化**：开篇可以是反常识的判断、一个具体场景、一个真实案例或一个直击要害的问题，不要每次都「随着……的发展」「近年来……」式起手；正文段落开头也避免整篇同一句式。
17. **细节要具体可查**：涉及具体工具、命令、数字、版本、组织时给出真实名称（如「pi install npm:pi-web-access」这类可查的细节），禁止「某团队」「相关研究」式虚指。
18. **口语颗粒点缀**：允许「说白了」「其实」「顺手」「这活儿」这类口语词少量点缀，每千字一两处即可，不破坏专业感；正式论述仍保持克制（与第 8 条一致）。

### 技术图解（涉及架构/原理/流程时必做）
19. 如果主题涉及**技术架构、系统原理、工作流程**（如微服务架构、编译流程、协议交互、算法机制），用 Mermaid 绘制图解（架构图/流程图/时序图/状态图）。**数量取决于内容需要，不设死数**：复杂架构或长流程可以在每个关键环节各画一张（3~6 张），简单原理 1~2 张即可，用文字能讲清就不画——图解的价值是让读者少读一段字，画之前自问「这张图能省下读者的理解成本吗」。格式示例：
```mermaid
graph TD
    A[用户请求] --> B[网关]
    B --> C{{鉴权}}
    C -- 通过 --> D[业务服务]
    C -- 失败 --> E[拒绝]
```
20. 图解服务于理解：节点用简短名词、边用动词说明关系；不要画与正文无关的装饰图。
涉及数学公式/复杂度/推导时用 LaTeX 书写：行内公式 $...$（如 $O(n \log n)$），独立公式用 $$...$$ 独占一行。

### 文章结构（因题而异，不要套模板）
结构必须为**这个具体问题**定制：先想清楚这个问题最自然的展开方式，
再定章节与标题，不要每次都复用同一套格式。常见问题的自然结构（仅参考）：
- 对比/选型类：先定对比维度，再逐维度分析，结尾给适用场景与选择建议
  （标题可以是「如何选择」「适合什么场景」，而不是千篇一律的「结论」）
- 原理/机制类：从外层现象切入，逐层拆解机制（组件职责、数据流、关键权衡），
  配图解；结尾可以是「这套机制给我们的启示」
- 决策/评估类：先摆问题与约束，再给评估框架，逐项分析，结尾给可执行的
  决策清单或建议
- 教程/实践类：按「为什么 → 怎么做 → 会踩什么坑 → 怎么验证」组织
结尾避免固定格式：可以是一段有力的收束 + 少量要点，也可以直接给行动建议；
结尾标题按问题的具体语义起（如「如何落地」「下一步建议」「关键取舍」），
不要每次都写「结论：1. 2. 3.」式的公式化结尾。"""

SYNTHESIZER_PROMPT = """你是一位资深技术作者，作品常见于顶级技术媒体与行业研究机构，以专业、深入、可信著称。请基于研究问题和检索证据，撰写一篇读者读完会觉得有收获的深度技术文章。

**先理解问题的意图与范围再动笔**：判断问题是「解释原理/机制」「对比」「选型评估」「趋势分析」还是「实操教程」——文章只覆盖问题明确问到的内容，不要为了显得全面而写超出问题范围的东西。

## 研究问题
{question}

## 检索证据
{search_results}

## 可用插图（可选）
{image_section}

{writing_rules}

## 文风要求（当前所选风格）
{style_section}"""

SYNTHESIZER_INCREMENTAL_PROMPT = """你是一位资深技术作者，作品常见于顶级技术媒体与行业研究机构，以专业、深入、可信著称。请基于已有知识与最新检索证据，撰写一篇更新后的完整深度技术文章（不是只写增量部分），读者读完会觉得有收获。

**先理解问题的意图与范围再动笔**：判断问题是「解释原理/机制」「对比」「选型评估」「趋势分析」还是「实操教程」——文章只覆盖问题明确问到的内容，不要为了显得全面而写超出问题范围的东西。

## 研究问题
{question}

## 已有知识
{knowledge_context}

## 新检索证据
{search_results}

## 可用插图（可选）
{image_section}

**增量补充**：客观陈述最新进展与既有结论之间的关系（如「截至 2024 年……」「此前的结论在最新资料中得到确认」），不使用 🆕/⚠️ 等进度标记符号。

{writing_rules}

## 文风要求（当前所选风格）
{style_section}"""

SYNTHESIZER_REFINE_PROMPT = """你是一位资深技术作者，作品常见于顶级技术媒体与行业研究机构，以专业、深入、可信著称。以下是之前生成的文章和改进建议，请根据建议改进，产出一篇专业、深入、可信、读者读完会觉得有收获的最终稿。

**先理解问题的意图与范围再动笔**：判断问题是「解释原理/机制」「对比」「选型评估」「趋势分析」还是「实操教程」——文章只覆盖问题明确问到的内容，不要为了显得全面而写超出问题范围的东西。

## 研究问题
{question}

## 检索证据（包含补充搜索的新结果）
{search_results}

## 可用插图（可选）
{image_section}

## 当前文章
{report}

## 改进建议
{feedback}

{writing_rules}

## 文风要求（当前所选风格）
{style_section}"""


# ── 来源编号表 ──────────────────────────────────────────

def build_source_table(state: ResearchState) -> list[dict]:
    """从 search_results + known_source_urls 构建编号引用表。

    顺序：本次检索结果（按收集顺序）在前，历史知识来源在后。
    按 normalize_url 去重；knowledge 来源没有标题，用 URL 充当。
    """
    table: list[dict] = []
    seen: set[str] = set()
    for r in state.get("search_results", []):
        url = r.get("url", "")
        key = normalize_url(url)
        if url and key and key not in seen:
            seen.add(key)
            table.append({
                "index": len(table) + 1,
                "title": r.get("title", "") or url,
                "url": url,
                "source": "search",
            })
    for url in state.get("known_source_urls", []):
        key = normalize_url(url)
        if url and key and key not in seen:
            seen.add(key)
            table.append({
                "index": len(table) + 1,
                "title": url,
                "url": url,
                "source": "knowledge",
            })
    return table


def format_source_table(table: list[dict]) -> str:
    """把编号表格式化为 prompt 可读文本。"""
    if not table:
        return "（无可用来源）"
    return "\n".join(
        f"[{item['index']}] {item.get('title', '')} — {item.get('url', '')}"
        for item in table
    )


def render_references(refs: list[dict]) -> str:
    """生成文末核心参考文献的 Markdown（代码生成，重新编号 1..k）。"""
    if not refs:
        return ""
    lines = [
        "",
        "## 参考文献",
        "",
        *[f"{i}. [{item.get('title', item.get('url', ''))}]({item.get('url', '')})"
          for i, item in enumerate(refs, 1)],
        "",
    ]
    return "\n".join(lines)


# ── 核心文献筛选 ──────────────────────────────────────

CORE_REFS_PROMPT = """你是学术文献专家。以下是研究问题及检索到的全部来源，请选出最核心、最值得作为文章参考文献的 {max_count} 个来源。

## 研究问题
{question}

## 来源列表
{source_list}

## 选择原则
- 优先权威一手来源：官方文档、学术论文、官方报告、行业权威站
- 优先与问题直接相关、信息密度高的来源
- 来源足够多时尽量覆盖不同子问题的关键证据
- 数量不超过 {max_count} 个；没有足够核心的来源时可以少于 {max_count}

## 输出格式
只返回 JSON（不要包含其他内容）：
```json
{{"indexes": [1, 3, 7]}}
```
- indexes 是来源列表里的编号，按重要程度排序"""


def curate_core_references(question: str, source_table: list[dict],
                           max_refs: int | None = None,
                           config: RunnableConfig | None = None) -> list[dict]:
    """从全部来源中筛选核心参考文献子集。

    LLM 选择（一次性非流式调用）；失败时降级取来源列表前 max_refs 个
    （search_results 按 Tavily 相关度排序，天然近似核心）。

    Returns:
        source_table 的子集（保持原条目结构），数量 ≤ max_refs。
    """
    max_refs = max_refs or MAX_REFERENCES
    if not source_table:
        return []
    # 控制 prompt 大小：候选超过 max_refs*2 时先截断
    candidates = source_table[: max_refs * 2]

    source_list = "\n".join(
        f"[{item['index']}] {item.get('title', '')} — {item.get('url', '')}"
        for item in candidates
    )
    try:
        llm = create_llm()
        prompt = get_prompt_from_langfuse(
            "research-buddy-core-refs", CORE_REFS_PROMPT,
            question=question,
            source_list=source_list,
            max_count=max_refs,
        )
        response = invoke_llm(llm, prompt, config=config)
        parsed = parse_llm_json(response.content)
        indexes_raw = parsed.get("indexes", []) if isinstance(parsed, dict) else []
        picked: list[dict] = []
        seen: set[int] = set()
        for raw in indexes_raw:
            try:
                index = int(raw)
            except (TypeError, ValueError):
                continue
            # 编号必须落在候选范围内，且不重复（防幻觉）
            if index < 1 or index > len(candidates) or index in seen:
                continue
            seen.add(index)
            picked.append(candidates[index - 1])
            if len(picked) >= max_refs:
                break
        if picked:
            return picked
        logger.warning("核心文献筛选输出无效，降级为按来源顺序截取")
    except Exception as exc:
        logger.warning("核心文献筛选失败，降级为按来源顺序截取: %s", exc)

    # 兜底：search_results 在 source_table 里按 Tavily 相关度排序
    return source_table[:max_refs]


# ── 标题去模板化（防「名词：副题」式冒号标题） ──────────

import re as _re

_HEADING_MD_RE = _re.compile(r"^#{2,3}\s*(.+?)\s*$", _re.M)
_HEADING_BOLD_RE = _re.compile(r"^\*\*(.+?)\*\*\s*$")
_COLON_HEADING_RATIO = 0.5      # 冒号式标题占比 ≥ 50% 且 ≥2 个 → 重写
_COLON_HEADING_MIN = 2

HEADING_REWRITE_PROMPT = """下面是一篇文章的标题列表，其中大量是「名词：副题」式冒号标题，读起来千篇一律。请把它们重写为风格多样的标题：

要求：
- 不要再用冒号「：」
- 交替使用陈述句标题、疑问句标题、短语式标题
- 标题必须准确对应原内容，不能改变含义或丢失信息
- 输出 JSON 对象：{"titles": ["新标题1", "新标题2", ...]}，与输入顺序一一对应

## 文章主题
{question}

## 原标题（按顺序）
{headings}"""


def _collect_headings(report: str) -> list[dict]:
    """收集文章标题（##/### markdown 或独立一行的 **粗体**），跳过代码块内部。"""
    items: list[dict] = []
    in_code = False
    for line in report.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = _HEADING_MD_RE.match(line)
        if m:
            items.append({"text": m.group(1), "is_md": True})
            continue
        m = _HEADING_BOLD_RE.match(line)
        if m:
            items.append({"text": m.group(1), "is_md": False})
    return items


def _colon_heading_ratio(headings: list[dict]) -> tuple[float, int]:
    """返回 (冒号式标题占比, 冒号式标题数量)。"""
    if not headings:
        return 0.0, 0
    colon = sum(1 for h in headings if "：" in h["text"])
    return colon / len(headings), colon


def _normalize_headings(question: str, report: str,
                          config: RunnableConfig | None = None) -> str:
    """若冒号式标题过多，用一次小 LLM 调用重写标题（代码级保底）。

    在 synthesizer 出稿前执行，不依赖反思循环是否拦截；
    LLM 失败/结果不合法时保留原标题，不影响出稿。
    """
    headings = _collect_headings(report)
    ratio, colon_count = _colon_heading_ratio(headings)
    if not headings or colon_count < _COLON_HEADING_MIN or ratio < _COLON_HEADING_RATIO:
        return report
    try:
        llm = create_llm()
        prompt = get_prompt_from_langfuse(
            "research-buddy-heading-rewrite", HEADING_REWRITE_PROMPT,
            question=question,
            headings="\n".join(f"- {h['text']}" for h in headings),
        )
        response = invoke_llm(llm, prompt, config=config)
        parsed = parse_llm_json(response.content)
        titles = parsed.get("titles", []) if isinstance(parsed, dict) else []
        if not isinstance(titles, list) or len(titles) != len(headings):
            raise ValueError(f"标题重写数量不匹配: {len(titles)} != {len(headings)}")
        titles = [str(t).strip() for t in titles]
        if any(not t or "：" in t for t in titles):
            raise ValueError("重写结果仍含冒号或为空标题")
        # 按原文精确替换（保留 ## / ** 前缀），从前往后逐个替换
        changed = 0
        for h, new_title in zip(headings, titles):
            if h["text"] in report:
                report = report.replace(h["text"], new_title, 1)
                changed += 1
        if changed:
            logger.info("标题去模板化：重写 %d/%d 个冒号式标题", changed, len(headings))
        return report
    except Exception as exc:
        logger.warning("标题重写失败，保留原标题: %s", exc)
        return report


def compute_confidence(state: ResearchState) -> str:
    """由代码从证据质量确定性计算置信度（高/中/低），不进报告正文。

    规则（按优先级）：
    - 搜索层不可用（无新证据）→ 低
    - 语义证据评估降级（仅机械校验）→ 中
    - 预算耗尽/无新查询/反思轮次用尽 → 中
    - 仍有未解决缺口 → 中
    - 其余（无缺口、无降级、无预算问题）→ 高
    """
    if state.get("search_unavailable"):
        return "低"
    if state.get("evidence_assessment_degraded"):
        return "中"
    if state.get("stop_reason") in {"search_budget_exhausted", "no_new_queries",
                                    "reflection_budget_exhausted"}:
        return "中"
    if state.get("validation_gaps"):
        return "中"
    return "高"


def _build_research_notes(state: ResearchState) -> list[str]:
    """收集不进正文的研究说明（局限/降级/未解决缺口）。

    面向读者而非开发者：不出现内部字段名（如 search_budget_exhausted），
    措辞温和、不惊悚；缺口只列高优先级的最多 3 条，避免整屏刷清单。
    """
    unresolved_gaps = state.get("validation_gaps", [])
    stop_reason = state.get("stop_reason", "")

    # 预算/轮次停止的读者友好措辞（key 顺序即匹配顺序）
    _BUDGET_NOTES = {
        "search_budget_exhausted": "本次研究已用尽搜索预算，部分内容证据有限，结论仅供参考。",
        "no_new_queries": "本次研究未能产生新的搜索方向，部分内容证据有限。",
        "reflection_budget_exhausted": "本次研究已用尽优化轮次，部分章节可能还有完善空间。",
    }

    notes: list[str] = []
    if state.get("search_unavailable"):
        notes.append("本次检索未获得任何新证据（搜索层不可用），结论缺少来源支撑。")
    if state.get("evidence_assessment_degraded"):
        notes.append("语义证据评估不可用，仅做了来源数/域名/覆盖度的机械校验，未做语义充分性判断。")
    if stop_reason in _BUDGET_NOTES:
        notes.append(_BUDGET_NOTES[stop_reason])
        high_priority = [gap for gap in unresolved_gaps if gap.get("priority") == "high"]
        shown_gaps = high_priority[:3] or unresolved_gaps[:3]
        notes.extend(
            f"- {gap.get('question', '未解决问题')}：{gap.get('reason', '证据不足')}"
            for gap in shown_gaps
        )
    return notes


def synthesizer(state: ResearchState, config: RunnableConfig, *, writer: StreamWriter) -> dict:
    """综合节点：流式输出可发布的研究文章

    支持三种模式：
    - 全新模式：正常生成文章
    - 增量模式：基于已有知识，补充更新文章
    - 改进模式：根据反思反馈重写文章

    使用 streaming 模式，报告内容"打字机式"逐步输出到终端和前端。
    通过 writer 参数将每个 chunk 推送到 SSE 层，实现前端实时显示。

    writer 必须标注为 StreamWriter：LangGraph 只在注解命中
    `(StreamWriter, "StreamWriter", inspect.Parameter.empty)` 白名单时才注入该参数
    （见 langgraph/_internal/_runnable.py 的 KWARGS_CONFIG_KEYS）。标成
    `Callable | None` 会让注入被静默跳过，writer 恒为 None，report_chunk 事件全部消失。
    这里不给默认值，缺少注入时直接 TypeError，而不是退化成无声失效。
    """
    question = state["question"]
    search_results = state.get("search_results", [])
    report = state.get("report", "")
    feedback = state.get("reflection_feedback", "")
    is_incremental = state.get("is_incremental", False)
    has_knowledge = state.get("has_knowledge", False)
    knowledge_context = state.get("knowledge_context", "")
    evidence_assessments = state.get("evidence_assessments", [])
    unresolved_gaps = state.get("validation_gaps", [])
    stop_reason = state.get("stop_reason", "")

    # 编号引用表：正文 [n] 与文末参考文献的单一事实来源
    source_table = build_source_table(state)

    # 视觉选图（可选）：配置 VISION_MODEL 才生效；未配置/失败返回 []，
    # 文章退化为无插图，不阻塞出稿。
    # 跨轮次复用：反思后的重写/回环轮直接复用上一轮选中的图，
    # 不再重新下载图片 + 调视觉模型（省 40~90s/轮）。
    image_candidates = state.get("image_candidates", [])
    selected_images: list[dict] = state.get("selected_images", []) or []
    if not selected_images and image_candidates:
        try:
            selected_images = select_images(state.get("sub_questions", []), image_candidates)
        except Exception as exc:
            logger.warning("视觉选图整体失败，文章不含插图: %s", exc)
    elif selected_images:
        logger.info("复用上一轮选中的 %d 张插图", len(selected_images))

    if selected_images:
        image_section = "\n".join(
            f"- 图{i}: {img.get('url', '')}（子问题：{img.get('sub_question_id', '')}，alt：{img.get('alt', '')}）"
            for i, img in enumerate(selected_images, 1)
        )
    else:
        image_section = "（无）"

    # 格式化搜索结果
    formatted_results = ""
    for i, r in enumerate(search_results, 1):
        formatted_results += f"\n### 结果 {i}（子问题：{r['sub_question']}）\n"
        formatted_results += f"- 标题：{r['title']}\n"
        formatted_results += f"- 来源：{r['url']}\n"
        formatted_results += f"- 内容：{r['content']}\n"
        formatted_results += f"- 相关度：{r['score']}\n"

    if evidence_assessments:
        formatted_results += "\n## 证据覆盖状态（仅供你判断，不要写进正文）\n"
        for assessment in evidence_assessments:
            formatted_results += (
                f"- {assessment.get('sub_question_id', '')}: "
                f"{assessment.get('status', '')}, 覆盖度 {assessment.get('coverage', 0):.0%}, "
                f"有效来源 {assessment.get('valid_results', 0)}，独立域名 {assessment.get('distinct_domains', 0)}\n"
            )
    if unresolved_gaps:
        formatted_results += "\n## 未解决证据缺口（仅供你判断，不要写进正文）\n"
        for gap in unresolved_gaps:
            formatted_results += f"- {gap.get('question', '')}: {gap.get('reason', '信息不足')}\n"
    if state.get("evidence_assessment_degraded"):
        formatted_results += (
            "\n注意：本次语义证据评估不可用，只做了来源数/域名/覆盖度的机械校验。"
            "正文中不得声称结论已交叉验证，但也不要把这条写进正文。\n"
        )
    if state.get("search_unavailable"):
        formatted_results += (
            "\n注意：本次检索没有获得任何新证据（搜索层不可用）。"
            "只能基于上面的已有知识作答，且不要声称有来源支撑。\n"
        )
    if stop_reason in {"search_budget_exhausted", "no_new_queries", "reflection_budget_exhausted"}:
        formatted_results += f"\n研究因 {stop_reason} 停止。正文不得将有限结论描述为已完全验证。\n"

    # 文章正文生成：MAX_ARTICLE_TOKENS 控制最大长度（0 = 不限制）
    llm = create_llm(streaming=True, max_tokens=MAX_ARTICLE_TOKENS or None)
    prompt_kwargs = {
        "question": question,
        "search_results": formatted_results,
        "image_section": image_section,
        "style_section": get_style_section(state.get("style")),
        "image_limit": str(MAX_IMAGES_IN_ARTICLE),
        # 共享写作规范（单一事实来源；image_limit 在这里渲染进规则文本）
        "writing_rules": WRITING_RULES.format(image_limit=str(MAX_IMAGES_IN_ARTICLE)),
    }

    # 选择模式
    if feedback and report:
        # 改进模式（反思后重写）
        prompt = get_prompt_from_langfuse(
            "research-buddy-synthesizer-refine", SYNTHESIZER_REFINE_PROMPT,
            **prompt_kwargs,
            report=report,
            feedback=feedback,
        )
        mode = "改进"
    elif is_incremental and has_knowledge and knowledge_context:
        # 增量模式也走 Langfuse Prompt 管理
        prompt = get_prompt_from_langfuse(
            "research-buddy-synthesizer-incremental", SYNTHESIZER_INCREMENTAL_PROMPT,
            **prompt_kwargs,
            knowledge_context=knowledge_context,
        )
        mode = "增量"
    else:
        # 全新模式
        prompt = get_prompt_from_langfuse(
            "research-buddy-synthesizer", SYNTHESIZER_PROMPT,
            **prompt_kwargs,
        )
        mode = "全新"

    logger.info("正在生成研究文章（%s模式）...", mode)

    # 流式输出正文。流式中途失败无法干净重试（已推送的 chunk 会重复），
    # 但首个 chunk 前的瞬时错误（503/429 等）可以整段重试——否则一次
    # 上游抖动就让整个研究失败。
    full_report = ""
    stream = llm.stream(prompt, config=config)
    first_chunk = None
    for attempt in range(3):
        try:
            first_chunk = next(stream)
            break
        except StopIteration:
            break
        except Exception as exc:
            if not _is_transient_error(exc) or attempt >= 2:
                raise
            logger.warning("文章流式生成在首个 chunk 前失败，整段重试（第 %d/3 次）: %s",
                           attempt + 1, str(exc)[:120])
            time.sleep(1.5 * (attempt + 1))
            stream = llm.stream(prompt, config=config)

    def _emit(chunk) -> None:
        nonlocal full_report
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_report += content
            if writer:
                writer({"type": "report_chunk", "content": content})

    if first_chunk is not None:
        _emit(first_chunk)
        try:
            while True:
                _emit(next(stream))
        except StopIteration:
            pass

    # 标题去模板化（代码级保底）：模型即使反复写「名词：副题」式冒号标题，
    # 这里也直接重写标题让风格多样——不依赖反思循环是否拦住。
    full_report = _normalize_headings(question, full_report, config=config)

    # 文末核心参考文献：LLM 从全部来源中筛选子集，代码重新编号生成。
    # 跨轮次复用：来源集（URL 签名）未变时直接复用上一轮的筛选结果，
    # 不再调 LLM（省 ~10s/轮）；补充搜索带来了新来源才重新筛选。
    source_signature = "|".join(sorted(
        normalize_url(item.get("url", "")) for item in source_table if item.get("url")
    ))
    if state.get("core_refs_signature") == source_signature and state.get("core_refs"):
        core_refs = state["core_refs"]
    else:
        core_refs = curate_core_references(question, source_table, config=config)
    references = render_references(core_refs)
    if references:
        full_report += references
        if writer:
            writer({"type": "report_chunk", "content": references})

    print()  # 换行
    logger.info("报告生成完成（%s模式）", mode)

    return {
        "report": full_report,
        "confidence": compute_confidence(state),
        "research_notes": _build_research_notes(state),
        "source_table": source_table,
        "selected_images": selected_images,
        "core_refs": core_refs,
        "core_refs_signature": source_signature,
        "messages": [f"📝 报告生成完成（{mode}模式）"],
    }
