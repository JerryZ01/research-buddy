"""Research State - LangGraph 全局共享状态"""

import operator
from typing import Annotated, TypedDict


class SubQuestion(TypedDict):
    """子问题"""
    id: str
    question: str
    search_query: str  # 用于搜索的查询词
    search_queries: list[dict]  # 可包含中英双语查询及地区偏好
    language: str
    region: str
    source_preference: str


class SearchResult(TypedDict):
    """单个搜索结果"""
    sub_question_id: str
    sub_question: str
    query: str
    language: str
    region: str
    title: str
    url: str
    content: str
    score: float


class ValidationGap(TypedDict):
    """信息缺口 — 子问题搜索结果不足，需要补充搜索"""
    sub_question_id: str
    question: str
    search_query: str  # 补充搜索词
    reason: str
    priority: str
    language: str
    region: str


class EvidenceAssessment(TypedDict):
    """单个子问题的证据覆盖评估"""
    sub_question_id: str
    status: str
    coverage: float
    valid_results: int
    distinct_domains: int
    missing_evidence: list[str]
    contradictions: list[str]


class ResearchState(TypedDict):
    """研究工作流的全局状态

    节点返回 dict，自动 merge 到 State。
    大部分列表字段用 Annotated[list, operator.add] 实现追加而非覆盖。
    sub_questions 和 validation_gaps 使用覆盖语义，便于用户编辑和清空已处理缺口。
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
    sub_questions: list[SubQuestion]  # 拆解的子问题（覆盖语义，HITL 可编辑替换）

    # 搜索阶段
    search_results: Annotated[list[SearchResult], operator.add]  # 搜索结果

    # 验证阶段（Phase 2）
    validation_gaps: list[ValidationGap]  # 信息缺口（覆盖语义，搜索后可清空）
    evidence_assessments: list[EvidenceAssessment]  # 当前证据评估（覆盖语义）
    evidence_assessment_degraded: bool  # 语义评估不可用，仅确定性判断（报告需披露）
    search_history: Annotated[list[dict], operator.add]  # 已执行搜索任务
    search_round: int
    total_queries: int
    stop_reason: str
    research_complete: bool
    search_unavailable: bool  # 搜索层不可用（无 key 或全部请求失败）

    # 综合阶段
    report: str  # 最终报告（可发布的文章正文，不含评价性内容与内嵌链接）
    confidence: str  # 置信度（高/中/低），由代码从证据质量计算，不进报告正文
    research_notes: list[str]  # 研究说明（局限/降级/未解决缺口），不进正文，供 API/前端展示
    source_table: list[dict]  # 编号引用表 [{index, title, url, source}]，synthesizer 构建，reflector 校验用

    # 反思阶段（Phase 2）
    reflection_pass: bool  # 反思是否通过
    reflection_feedback: str  # 反思反馈/改进建议
    reflection_round: int  # 当前反思轮次
    reflection_score: int  # 三维度总分（满分 15），供 API 层展示证据质量

    # Human-in-the-loop（Phase 3）
    user_feedback: str  # 用户在中断时提供的反馈/调整

    # 追踪层（Phase 7）
    detected_changes: Annotated[list[dict], operator.add]  # 检测到的变化
    similarity: float  # 新旧报告相似度
    tracking_log_id: str  # 追踪记录 ID
    notification_sent: bool  # 是否已发送通知

    # 进度消息（供 API 层推送）
    messages: Annotated[list[str], operator.add]  # 节点进度消息
