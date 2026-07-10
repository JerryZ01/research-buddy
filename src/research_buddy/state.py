"""Research State - LangGraph 全局共享状态"""

import operator
from typing import Annotated, TypedDict


class SubQuestion(TypedDict):
    """子问题"""
    question: str
    search_query: str  # 用于搜索的查询词


class SearchResult(TypedDict):
    """单个搜索结果"""
    sub_question: str
    title: str
    url: str
    content: str
    score: float


class ValidationGap(TypedDict):
    """信息缺口 — 子问题搜索结果不足，需要补充搜索"""
    question: str
    search_query: str  # 补充搜索词


class ResearchState(TypedDict):
    """研究工作流的全局状态

    节点返回 dict，自动 merge 到 State。
    列表字段用 Annotated[list, operator.add] 实现追加而非覆盖。
    """
    # 输入
    question: str  # 原始研究问题

    # 知识层（Phase 6）
    topic_id: str  # 关联的研究主题 ID（可选，为空则不持久化）
    knowledge_context: str  # 历史知识上下文（knowledge_lookup 生成）
    has_knowledge: bool  # 是否有历史知识
    is_incremental: bool  # 是否增量研究模式
    known_source_urls: list[str]  # 已有知识的来源 URL 列表（增量去重用）
    key_facts: Annotated[list[str], operator.add]  # 提取的关键事实
    saved_report_id: str  # 保存后的报告 ID

    # 规划阶段
    sub_questions: Annotated[list[SubQuestion], operator.add]  # 拆解的子问题

    # 搜索阶段
    search_results: Annotated[list[SearchResult], operator.add]  # 搜索结果

    # 验证阶段（Phase 2）
    validation_gaps: Annotated[list[ValidationGap], operator.add]  # 信息缺口

    # 综合阶段
    report: str  # 最终报告

    # 反思阶段（Phase 2）
    reflection_pass: bool  # 反思是否通过
    reflection_feedback: str  # 反思反馈/改进建议
    reflection_round: int  # 当前反思轮次

    # Human-in-the-loop（Phase 3）
    user_feedback: str  # 用户在中断时提供的反馈/调整

    # 追踪层（Phase 7）
    detected_changes: Annotated[list[dict], operator.add]  # 检测到的变化
    similarity: float  # 新旧报告相似度
    tracking_log_id: str  # 追踪记录 ID
    notification_sent: bool  # 是否已发送通知

    # 进度消息（供 API 层推送）
    messages: Annotated[list[str], operator.add]  # 节点进度消息
