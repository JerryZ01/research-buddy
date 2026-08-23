"""综合节点 - 将搜索结果综合为可发布的结构化文章（支持流式输出）

报告 = 纯文章正文（无评价性内容、无内嵌 URL、无 [编号] 引用标注）
       + 文末核心参考文献（LLM 从全部来源中筛选 + 代码重编号生成）。
- 矛盾/不足/置信度等评价性信息写入 research_notes / confidence（state 字段），不进正文
"""

import logging

from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from research_buddy.config import MAX_IMAGES_IN_ARTICLE, MAX_REFERENCES
from research_buddy.state import ResearchState
from research_buddy.styles import get_style_section
from research_buddy.tools.images import select_images
from research_buddy.utils import create_llm, get_prompt_from_langfuse, normalize_url, parse_llm_json

logger = logging.getLogger(__name__)


# ── Prompt 模板（可发布文章风格） ────────────────────────

SYNTHESIZER_PROMPT = """你是一位资深技术作者，作品常见于顶级技术媒体与行业研究机构，以专业、深入、可信著称。请基于研究问题和检索证据，撰写一篇读者读完会觉得有收获的深度技术文章。

## 研究问题
{question}

## 检索证据
{search_results}

## 可用插图（可选）
{image_section}

## 写作要求

### 主线与观点
1. 文章必须有一条贯穿始终的分析主线——一个核心判断或问题视角（例如「GIL 正在退出历史舞台，但真正的瓶颈是 C 扩展生态」）。开篇点明主线，正文围绕它展开，而不是把资料平铺罗列。
2. 每个章节都要有自己的分析结论，而不是转述资料：先立观点，再用证据支撑。

### 深度与读者价值
3. 严禁综述腔（「根据 A 报道…」「B 研究显示…」式的堆砌拼接）。把检索到的证据**内化成自己的分析叙述**：解释背后的原理、动机、因果与影响。
4. 多写「所以呢」：给出权衡（tradeoff）、适用场景、优劣势对比、实践建议。让读者读完能带走判断与行动指引——这是文章价值的核心。
5. 关键概念给出准确精炼的定义与背景即可，不要花篇幅科普人人皆知的基础知识。

### 可信度与表达
**归因节制（重要）**：允许在关键事实上点名权威来源（如「AWS 官方文档建议」「GitHub 官方博客指出」「PEP 703 提案明确」），但要用得克制——只在不点名来源就缺乏说服力的权威论断、或该来源确实是此观点的出处时点名。大部分论断应当用你自己的分析叙述来表达，不要每个段落都「某公司说/建议/经验」式归因；多个来源结论一致时合并成一句综合陈述，不要逐一罗列「A 公司认为…，B 公司认为…」。全篇点名归因控制在少数几处（平均每千字不超过 1~2 处）。

**去 AI 味（重要）**：
- 不要写元评论/模板句：「本节的关键结论是」「值得注意的是」「综上所述」「这揭示了」
- 不要滥用排比对仗（「不是…而是…」「既…又…」），句子长短自然交错
- 不要自问自答（「这意味着什么？因为…」「为什么如此重要？因为它…」），不要用「拆解这个定义需要一点耐心」式引导句铺设
- 不要用 AI 高频套话：「赋能」「底层逻辑」「重构」「闭环」「颗粒度」「破圈」
- 不要平均用力：重点章节写透，次要章节可以一笔带过
- 观点要明确、有取舍，允许诚实地写「这块资料有限」「这个结论仍有争议」
- 结尾用散文收束或直接给建议，不要表格化、不要「结论：1. 2. 3.」
事实与数据必须来自检索证据，保持准确；资料有限或不确定处明确说明（如「截至本文写作时」「现有资料显示」），并区分事实、分析与推测，不做没有依据的断言。
7. 小标题要有信息量，能看出这一节的观点（如「为什么 I/O 密集场景不受 GIL 影响」），避免「概述」「背景介绍」这类空标题；章节按主题逻辑重组，不要机械地按子问题逐条罗列。**小标题风格要多样**：不要连续用「名词：解释」式冒号标题（如「自注意力：当每个 token 都成为检索者」），同一篇里交替使用陈述句、疑问句、短语式标题，冒号式标题整篇最多 1~2 个。
8. 语气专业、克制、自信；语言自然流畅，像一个有经验的人认真讲清楚一件事，避免套话、模板腔和「综上」「众所周知」式的空泛表达。
9. 论点直接陈述，**不需要**标注引用编号或来源链接（文末参考文献由系统自动生成）。
10. 如果提供了「可用插图」，在内容最相关的位置插入插图（**整篇最多 {image_limit} 张**，各章节合理分配）：优先插入能说明内容的图，也允许 2~3 张与主题氛围相符的装饰图点缀；每张图都应与所在内容相关或契合（根据 alt 描述判断），不要堆砌无关的图：`![alt文本](图片URL)`。图片 URL 必须原样来自「可用插图」列表，禁止使用列表之外的图片；alt 文本用「可用插图」里给出的描述。
11. 不要在正文中直接写出任何 URL，也不要写「信息存在矛盾」「证据不足」「研究局限」「本报告基于……搜索」等研究过程评论；不要自行添加「参考文献」「来源」「置信度」章节（文末参考文献由系统自动生成）。
12. 使用中文撰写。

### 技术图解（涉及架构/原理/流程时必做）
13. 如果主题涉及**技术架构、系统原理、工作流程**（如微服务架构、编译流程、协议交互、算法机制），用 Mermaid 绘制 1~2 张图解（架构图/流程图/时序图）放在最相关的章节，让读者直观理解。格式示例：
```mermaid
graph TD
    A[用户请求] --> B[网关]
    B --> C{{鉴权}}
    C -- 通过 --> D[业务服务]
    C -- 失败 --> E[拒绝]
```
14. 图解服务于理解：节点用简短名词、边用动词说明关系；不要画与正文无关的装饰图。
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
不要每次都写「结论：1. 2. 3.」式的公式化结尾。

## 文风要求（当前所选风格）
{style_section}"""

SYNTHESIZER_INCREMENTAL_PROMPT = """你是一位资深技术作者，作品常见于顶级技术媒体与行业研究机构，以专业、深入、可信著称。请基于已有知识与最新检索证据，撰写一篇更新后的完整深度技术文章（不是只写增量部分），读者读完会觉得有收获。

## 研究问题
{question}

## 已有知识
{knowledge_context}

## 新检索证据
{search_results}

## 可用插图（可选）
{image_section}

## 写作要求

### 主线与观点
1. 文章必须有一条贯穿始终的分析主线——一个核心判断或问题视角。开篇点明主线，正文围绕它展开，而不是把资料平铺罗列。
2. 每个章节都要有自己的分析结论，而不是转述资料：先立观点，再用证据支撑。

### 深度与读者价值
3. 严禁综述腔（「根据 A 报道…」「B 研究显示…」式的堆砌拼接）。把检索到的证据**内化成自己的分析叙述**：解释背后的原理、动机、因果与影响。
4. 多写「所以呢」：给出权衡（tradeoff）、适用场景、优劣势对比、实践建议，让读者读完能带走判断与行动指引。
5. 客观陈述最新进展与既有结论之间的关系（如「截至 2024 年……」「此前的结论在最新资料中得到确认」），不使用 🆕/⚠️ 等进度标记符号。
6. 关键概念给出准确精炼的定义与背景即可，不要花篇幅科普人人皆知的基础知识。

### 可信度与表达
**归因节制（重要）**：允许在关键事实上点名权威来源（如「AWS 官方文档建议」「GitHub 官方博客指出」「PEP 703 提案明确」），但要用得克制——只在不点名来源就缺乏说服力的权威论断、或该来源确实是此观点的出处时点名。大部分论断应当用你自己的分析叙述来表达，不要每个段落都「某公司说/建议/经验」式归因；多个来源结论一致时合并成一句综合陈述，不要逐一罗列「A 公司认为…，B 公司认为…」。全篇点名归因控制在少数几处（平均每千字不超过 1~2 处）。

**去 AI 味（重要）**：
- 不要写元评论/模板句：「本节的关键结论是」「值得注意的是」「综上所述」「这揭示了」
- 不要滥用排比对仗（「不是…而是…」「既…又…」），句子长短自然交错
- 不要自问自答（「这意味着什么？因为…」「为什么如此重要？因为它…」），不要用「拆解这个定义需要一点耐心」式引导句铺设
- 不要用 AI 高频套话：「赋能」「底层逻辑」「重构」「闭环」「颗粒度」「破圈」
- 不要平均用力：重点章节写透，次要章节可以一笔带过
- 观点要明确、有取舍，允许诚实地写「这块资料有限」「这个结论仍有争议」
- 结尾用散文收束或直接给建议，不要表格化、不要「结论：1. 2. 3.」
事实与数据必须来自检索证据，保持准确；资料有限或不确定处明确说明（如「截至本文写作时」「现有资料显示」），并区分事实、分析与推测，不做没有依据的断言。
8. 小标题要有信息量，能看出这一节的观点，避免「概述」「背景介绍」这类空标题；章节按主题逻辑重组，不要机械地按子问题逐条罗列。**小标题风格要多样**：不要连续用「名词：解释」式冒号标题（如「自注意力：当每个 token 都成为检索者」），同一篇里交替使用陈述句、疑问句、短语式标题，冒号式标题整篇最多 1~2 个。
9. 语气专业、克制、自信；语言自然流畅，避免套话、模板腔和「综上」「众所周知」式的空泛表达。
10. 论点直接陈述，**不需要**标注引用编号或来源链接（文末参考文献由系统自动生成）。
11. 如果提供了「可用插图」，在内容最相关的位置插入插图（**整篇最多 {image_limit} 张**，各章节合理分配）：优先插入能说明内容的图，也允许 2~3 张与主题氛围相符的装饰图点缀；每张图都应与所在内容相关或契合（根据 alt 描述判断），不要堆砌无关的图：`![alt文本](图片URL)`。图片 URL 必须原样来自「可用插图」列表，禁止使用列表之外的图片；alt 文本用「可用插图」里给出的描述。
12. 不要在正文中直接写出任何 URL，也不要用 `[编号]` 形式标注来源；不要写「信息存在矛盾」「证据不足」「研究局限」等研究过程评论；不要自行添加「参考文献」「来源」「置信度」章节。
13. 使用中文撰写。

### 技术图解（涉及架构/原理/流程时必做）
14. 如果主题涉及**技术架构、系统原理、工作流程**，用 Mermaid 绘制 1~2 张图解（架构图/流程图/时序图）放在最相关的章节，让读者直观理解。格式示例：
```mermaid
graph TD
    A[用户请求] --> B[网关]
    B --> C{{鉴权}}
    C -- 通过 --> D[业务服务]
    C -- 失败 --> E[拒绝]
```
15. 图解服务于理解：节点用简短名词、边用动词说明关系；不要画与正文无关的装饰图。
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
不要每次都写「结论：1. 2. 3.」式的公式化结尾。

## 文风要求（当前所选风格）
{style_section}"""

SYNTHESIZER_REFINE_PROMPT = """你是一位资深技术作者，作品常见于顶级技术媒体与行业研究机构，以专业、深入、可信著称。以下是之前生成的文章和改进建议，请根据建议改进，产出一篇专业、深入、可信、读者读完会觉得有收获的最终稿。

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

## 写作要求

### 主线与观点
1. 文章必须有一条贯穿始终的分析主线——一个核心判断或问题视角。开篇点明主线，正文围绕它展开，而不是把资料平铺罗列。
2. 每个章节都要有自己的分析结论，而不是转述资料：先立观点，再用证据支撑。

### 深度与读者价值
3. 严禁综述腔（「根据 A 报道…」「B 研究显示…」式的堆砌拼接）。把检索到的证据**内化成自己的分析叙述**：解释背后的原理、动机、因果与影响。
4. 多写「所以呢」：给出权衡（tradeoff）、适用场景、优劣势对比、实践建议，让读者读完能带走判断与行动指引。
5. 关键概念给出准确精炼的定义与背景即可，不要花篇幅科普人人皆知的基础知识。

### 可信度与表达
**归因节制（重要）**：允许在关键事实上点名权威来源（如「AWS 官方文档建议」「GitHub 官方博客指出」「PEP 703 提案明确」），但要用得克制——只在不点名来源就缺乏说服力的权威论断、或该来源确实是此观点的出处时点名。大部分论断应当用你自己的分析叙述来表达，不要每个段落都「某公司说/建议/经验」式归因；多个来源结论一致时合并成一句综合陈述，不要逐一罗列「A 公司认为…，B 公司认为…」。全篇点名归因控制在少数几处（平均每千字不超过 1~2 处）。

**去 AI 味（重要）**：
- 不要写元评论/模板句：「本节的关键结论是」「值得注意的是」「综上所述」「这揭示了」
- 不要滥用排比对仗（「不是…而是…」「既…又…」），句子长短自然交错
- 不要自问自答（「这意味着什么？因为…」「为什么如此重要？因为它…」），不要用「拆解这个定义需要一点耐心」式引导句铺设
- 不要用 AI 高频套话：「赋能」「底层逻辑」「重构」「闭环」「颗粒度」「破圈」
- 不要平均用力：重点章节写透，次要章节可以一笔带过
- 观点要明确、有取舍，允许诚实地写「这块资料有限」「这个结论仍有争议」
- 结尾用散文收束或直接给建议，不要表格化、不要「结论：1. 2. 3.」
事实与数据必须来自检索证据，保持准确；资料有限或不确定处明确说明（如「截至本文写作时」「现有资料显示」），并区分事实、分析与推测，不做没有依据的断言。
7. 小标题要有信息量，能看出这一节的观点，避免「概述」「背景介绍」这类空标题；章节按主题逻辑重组，不要机械地按子问题逐条罗列。**小标题风格要多样**：不要连续用「名词：解释」式冒号标题（如「自注意力：当每个 token 都成为检索者」），同一篇里交替使用陈述句、疑问句、短语式标题，冒号式标题整篇最多 1~2 个。
8. 语气专业、克制、自信；语言自然流畅，避免套话、模板腔和「综上」「众所周知」式的空泛表达。
9. 论点直接陈述，**不需要**标注引用编号或来源链接（文末参考文献由系统自动生成）。
10. 如果提供了「可用插图」，在内容最相关的位置插入插图（**整篇最多 {image_limit} 张**，各章节合理分配）：优先插入能说明内容的图，也允许 2~3 张与主题氛围相符的装饰图点缀；每张图都应与所在内容相关或契合（根据 alt 描述判断），不要堆砌无关的图：`![alt文本](图片URL)`。图片 URL 必须原样来自「可用插图」列表，禁止使用列表之外的图片；alt 文本用「可用插图」里给出的描述。
11. 不要在正文中直接写出任何 URL，也不要用 `[编号]` 形式标注来源；不要写「信息存在矛盾」「证据不足」「研究局限」等研究过程评论；不要自行添加「参考文献」「来源」「置信度」章节。
12. 使用中文撰写。

### 技术图解（涉及架构/原理/流程时必做）
13. 如果主题涉及**技术架构、系统原理、工作流程**，用 Mermaid 绘制 1~2 张图解（架构图/流程图/时序图）放在最相关的章节，让读者直观理解。格式示例：
```mermaid
graph TD
    A[用户请求] --> B[网关]
    B --> C{{鉴权}}
    C -- 通过 --> D[业务服务]
    C -- 失败 --> E[拒绝]
```
14. 图解服务于理解：节点用简短名词、边用动词说明关系；不要画与正文无关的装饰图。
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
不要每次都写「结论：1. 2. 3.」式的公式化结尾。

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
                           max_refs: int | None = None) -> list[dict]:
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
        response = llm.invoke(prompt)
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
    """收集不进正文的研究说明（局限/降级/未解决缺口）。"""
    unresolved_gaps = state.get("validation_gaps", [])
    stop_reason = state.get("stop_reason", "")

    notes: list[str] = []
    if state.get("search_unavailable"):
        notes.append("本次检索未获得任何新证据（搜索层不可用），结论缺少来源支撑。")
    if state.get("evidence_assessment_degraded"):
        notes.append("语义证据评估不可用，仅做了来源数/域名/覆盖度的机械校验，未做语义充分性判断。")
    if stop_reason in {"search_budget_exhausted", "no_new_queries", "reflection_budget_exhausted"}:
        notes.append(f"本次研究因 `{stop_reason}` 停止，以下内容仍需进一步验证。")
        notes.extend(
            f"- {gap.get('question', '未解决问题')}：{gap.get('reason', '证据不足')}"
            for gap in unresolved_gaps
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

    llm = create_llm(streaming=True)
    prompt_kwargs = {
        "question": question,
        "search_results": formatted_results,
        "image_section": image_section,
        "style_section": get_style_section(state.get("style")),
        "image_limit": str(MAX_IMAGES_IN_ARTICLE),
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

    # 流式输出正文
    full_report = ""
    for chunk in llm.stream(prompt):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_report += content
            # 推送 chunk 到 SSE 层，实现前端实时显示
            if writer:
                writer({"type": "report_chunk", "content": content})

    # 文末核心参考文献：LLM 从全部来源中筛选子集，代码重新编号生成。
    # 跨轮次复用：来源集（URL 签名）未变时直接复用上一轮的筛选结果，
    # 不再调 LLM（省 ~10s/轮）；补充搜索带来了新来源才重新筛选。
    source_signature = "|".join(sorted(
        normalize_url(item.get("url", "")) for item in source_table if item.get("url")
    ))
    if state.get("core_refs_signature") == source_signature and state.get("core_refs"):
        core_refs = state["core_refs"]
    else:
        core_refs = curate_core_references(question, source_table)
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
