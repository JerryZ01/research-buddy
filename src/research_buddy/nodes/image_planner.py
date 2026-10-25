"""配图规划节点 — 判定每个位置需要「真实素材」还是「画一张图」

核心设计：LLM 只输出**需求类型**（need），图源（kind）由代码推导。
这样「要真实截图却去生成」这类矛盾在结构上不可能发生——之前让 LLM
同时输出 need 和 kind，实测会出现「reason 写必须搜索真实图表、kind 却给
generate」的自相矛盾，生成出通用图标拼盘。

需求类型与图源的固定映射：

    need             含义                                    图源      搜不到时
    ─────────────────────────────────────────────────────────────────────────
    real_asset       现实中有明确来源：产品/人物/界面/公告/      搜图      不配图
                     现场照片/官方架构图
    data_viz         真实数值的对比、趋势、占比                 搜图      不配图
    structure        精细流程图、架构图、模块关系、状态机、     生图      —
                     时间线、概念对比
    decorative       好看的配图、概念意象、氛围图               生图      —
    none             无需配图                                 —         —

为什么 real_asset / data_viz 搜不到就「不配图」而不是生图兜底：
生成的「官方公告页面」或「基准测试柱状图」等于伪造证据——图上没有真实
数据却看起来像真的。这类位置宁可不配图，也不能用示意图冒充。

只有 structure / decorative 走生图，因为它们本来就是「画出来给人看关系」的，
不存在真实来源，生成反而比搜来的图更贴切正文。
"""

import json
import logging

from langchain_core.runnables import RunnableConfig

from research_buddy.state import ResearchState
from research_buddy.tools.image_gen import is_enabled as image_gen_enabled
from research_buddy.utils import (
    create_llm,
    get_prompt_from_langfuse,
    invoke_llm,
    parse_llm_json,
)

logger = logging.getLogger(__name__)

IMAGE_PLAN_PROMPT = """你是技术文章配图策划。下面是一次研究的主题与各子问题，请判断每个子问题该配什么图。

## 研究主题
{question}

## 子问题
{sub_questions}

## 判断方法

对每个子问题，问一句：**这个位置要展示的东西，现实中有没有一张已经存在的、权威的图？**

- 有 → need 选 `real_asset` 或 `data_viz`（走搜图）
- 没有，需要把关系画出来 → need 选 `structure` 或 `decorative`（走生图）
- 配图没有意义 → `none`

### 五个选项

**real_asset —— 现实中有明确来源的图**
产品界面、控制台、代码编辑器、官方公告页、定价页、产品外观、人物、发布会现场、
公司 Logo、官方发布的架构图、论文原图。
特征：你只能用一句话描述它「是什么」，描述不出具体内容 → 必须搜真图。

**data_viz —— 真实数值的对比/趋势/占比**
基准测试成绩、市场份额、参数规模、价格对比、性能曲线。
特征：图上的数字是真的，生成出来的数字是编的 → 必须搜真图。
注意：如果只是「示意某指标上涨/下降」而不涉及具体数值，选 structure。

**structure —— 精细的流程/结构图（生图主力）**
机制原理、系统架构、模块关系、数据流向、训练流程、状态机、时间线、
决策树、概念之间的对比与分类。
特征：这类图现实中通常没有现成版本，需要按本文语境画，且要求结构准确、
标签清晰 —— 生成比搜图更贴切。

**decorative —— 好看的配图（生图）**
抽象概念的意象化表达、章节头图、氛围图。
特征：不需要承载精确结构，作用是为文章增色、给读者一个视觉锚点。

**none —— 不配图**
纯观点、纯论述、纯数据罗列，任何图都是凑数。

## 写 want（决定成图质量的关键）

**need = real_asset 或 data_viz**：want 写**检索目标**——要什么样的图，
画面里应该能看到什么关键信息（如「官方定价页，能看到模型名称与每百万 token 价格」）。

**need = structure**：want 是**给画师的完整构图指令**，必须写清：
1. 画面结构：分几层/几栏/几个模块，怎么排列（横向/纵向/环形/矩阵），谁指向谁
2. 每个元素的具体形态：「一个圆柱形数据库带三道横纹」而不是「一个图标」
3. **图上的中文标签**：逐条列出每个模块要写的字（2~6 字，取自该子问题的关键术语）。
   模型中文渲染准确，这是图传递信息的主要手段，务必写全。
4. 视觉风格：配色（如「蓝为主、橙为强调」）、统一线宽、圆角、留白

好的 structure want 示例：
「环形流程图。中央六边形标注『策略模型』，顺时针弧形箭头依次指向：上方圆角矩形
『采样轨迹』（内含三条虚线路径）、右侧菱形『奖励评估』（内含刻度尺与对勾/叉）、
下方齿轮『参数更新』（内含旋转箭头），再回到中心。左下角小方块标注『可验证任务』，
虚线箭头汇入奖励菱形。蓝色描边统一线宽，奖励节点用橙色强调。」

**need = decorative**：want 写画面意象与风格即可，如「用层叠的半透明山峦表示
知识积累的渐进过程，极简线条，蓝紫渐变」。

禁止写法（会打回）：「一张展示 XX 的示意图」「XX 的架构图」「XX 的界面截图」。

## 硬性约束
- 每个子问题只给一个判断。
- want 里**禁止写真实人名、真实公司名、具体数值**（如「月之暗面」「2.8 万亿参数」
  「1679 分」）。这些内容画出来就是编造。需要指代时用中性角色名，如「该公司」
  「待测模型」「基线模型 A」。真实数据一律靠 real_asset / data_viz。
- structure 的 want 控制在 300 字内，但必须具体到能照着画。

## 输出格式
只返回 JSON（不要其他内容）：
{{"plans": [{{"id": "子问题 id", "need": "real_asset|data_viz|structure|decorative|none", "reason": "一句话理由", "want": "检索目标或构图指令"}}]}}"""

# need → kind 的固定映射。这张表是全模块「搜图 vs 生图」判断的唯一来源，
# LLM 无权决定图源，只能决定需求类型。
NEED_TO_KIND: dict[str, str] = {
    "real_asset": "search",
    "data_viz": "search",
    "structure": "generate",
    "decorative": "generate",
    "none": "skip",
}

# 走搜图的需求类型：搜不到时「不配图」，绝不用生成图兜底。
SEARCH_ONLY_NEEDS = frozenset({"real_asset", "data_viz"})
# 走生图的需求类型。
GENERATE_NEEDS = frozenset({"structure", "decorative"})

_MAX_WANT_CHARS = 600


def _sub_questions_text(sub_questions: list[dict], limit: int = 12) -> str:
    """把子问题渲染成紧凑清单，控制 prompt 体积。"""
    lines = []
    for sq in sub_questions[:limit]:
        sq_id = str(sq.get("id", "")).strip()
        question = str(sq.get("question", "")).strip()
        if not sq_id or not question:
            continue
        lines.append(f"- [{sq_id}] {question}")
    return "\n".join(lines)


def kind_for_need(need: str) -> str:
    """由需求类型推导图源。未知类型按 skip 处理（保守，不乱配图）。"""
    return NEED_TO_KIND.get((need or "").strip().lower(), "skip")


def normalize_plans(payload: object, sub_questions: list[dict]) -> list[dict]:
    """校验并收敛 LLM 输出。

    图源完全由 need 推导，LLM 输出的任何 kind 字段都被忽略——这从结构上
    消除了「需要真实素材却去生成」的矛盾。
    """
    if not isinstance(payload, dict):
        return []
    raw_plans = payload.get("plans")
    if not isinstance(raw_plans, list):
        return []

    known_ids = {str(sq.get("id", "")).strip() for sq in sub_questions if sq.get("id")}
    generate_ready = image_gen_enabled()
    plans: list[dict] = []
    seen: set[str] = set()

    for item in raw_plans:
        if not isinstance(item, dict):
            continue
        sq_id = str(item.get("id", "")).strip()
        if sq_id not in known_ids or sq_id in seen:
            continue

        need = str(item.get("need", "")).strip().lower()
        if need not in NEED_TO_KIND:
            # 兼容旧格式：LLM 可能仍输出 kind 字段
            legacy = str(item.get("kind", "")).strip().lower()
            need = {"search": "real_asset", "generate": "structure",
                    "skip": "none"}.get(legacy, "")
        if not need:
            continue

        kind = kind_for_need(need)
        if kind == "generate" and not generate_ready:
            # 生图未配置：structure/decorative 无处安放，不配图而不是硬塞搜图
            # （搜来的图未必符合「画结构」的意图）。
            logger.info("生图未配置，%s 计划跳过: %s", need, sq_id)
            kind = "skip"

        want = str(item.get("want", "")).strip()[:_MAX_WANT_CHARS]
        reason = str(item.get("reason", "")).strip()[:200]
        seen.add(sq_id)
        plans.append({
            "sub_question_id": sq_id,
            "need": need,
            "kind": kind,
            "reason": reason,
            "want": want,
            # 搜图型需求失败时不允许生图兜底，下游据此决定「宁缺毋滥」
            "never_generate": need in SEARCH_ONLY_NEEDS,
        })
    return plans


def build_image_plans(state: ResearchState, config: RunnableConfig | None = None,
                      use_local_prompt: bool = False) -> list[dict]:
    """调用 LLM 生成配图需求；失败返回 []。"""
    sub_questions = state.get("sub_questions", []) or []
    if not sub_questions:
        return []
    # 搜图与生图都没配置时配图整体关闭，不必浪费一次 LLM 调用。
    if not image_gen_enabled() and not _vision_ready():
        return []

    kwargs = {
        "question": state.get("question", ""),
        "sub_questions": _sub_questions_text(sub_questions),
    }
    try:
        prompt = (IMAGE_PLAN_PROMPT.format(**kwargs) if use_local_prompt else
                  get_prompt_from_langfuse(
                      "research-buddy-image-plan", IMAGE_PLAN_PROMPT, **kwargs,
                  ))
        response = invoke_llm(create_llm(), prompt, config=config)
        return normalize_plans(parse_llm_json(response.content), sub_questions)
    except Exception as exc:  # noqa: BLE001 - 任何失败都降级
        logger.warning("配图规划失败，本次不做配图: %s", exc)
        return []


def _vision_ready() -> bool:
    """视觉选图是否可用（决定搜图型需求有没有意义）。"""
    from research_buddy.config import VISION_MODEL
    from research_buddy.tools.images import _vision_key
    return bool(VISION_MODEL and _vision_key())


def image_planner(state: ResearchState, config: RunnableConfig | None = None) -> dict:
    """LangGraph 节点：产出 image_plans（覆盖语义，每轮重算）。"""
    use_local_prompt = bool(state.get("eval_use_local_prompts"))
    plans = build_image_plans(state, config=config, use_local_prompt=use_local_prompt)

    if not plans:
        summary = "配图规划不可用，本次不做配图"
    else:
        counts = {"search": 0, "generate": 0, "skip": 0}
        for plan in plans:
            counts[plan["kind"]] = counts.get(plan["kind"], 0) + 1
        summary = (
            f"配图规划完成：搜图 {counts['search']} 处、"
            f"生图 {counts['generate']} 处、不配图 {counts['skip']} 处"
        )
        needs = {}
        for plan in plans:
            needs[plan["need"]] = needs.get(plan["need"], 0) + 1
        detail = "、".join(f"{k} {v}" for k, v in sorted(needs.items()) if k != "none")
        if detail:
            summary += f"（{detail}）"

    return {
        "image_plans": plans,
        "messages": [summary],
    }


def plans_by_sub_question(plans: list[dict]) -> dict[str, dict]:
    """把计划列表转成 sub_question_id → plan 的索引。"""
    return {
        str(plan.get("sub_question_id", "")): plan
        for plan in (plans or []) if plan.get("sub_question_id")
    }


if __name__ == "__main__":
    # 冒烟测试：python -m research_buddy.nodes.image_planner
    print("need→kind 映射:", json.dumps(NEED_TO_KIND, ensure_ascii=False))
    demo = normalize_plans(
        {"plans": [
            {"id": "sq_01", "need": "real_asset", "want": "官方公告页"},
            {"id": "sq_02", "need": "structure", "want": "环形流程图"},
            {"id": "sq_03", "need": "data_viz", "want": "基准成绩柱状图"},
        ]},
        [{"id": "sq_01", "question": "?"}, {"id": "sq_02", "question": "?"},
         {"id": "sq_03", "question": "?"}],
    )
    for plan in demo:
        print(f"  {plan['sub_question_id']}: need={plan['need']} kind={plan['kind']} "
              f"never_generate={plan['never_generate']}")
