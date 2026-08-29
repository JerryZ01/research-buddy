# 文章素材库与独立评审模型

## 自动归档

文章素材库使用 SQLite 的 `article_generations`、`article_versions` 和
`article_reviews` 三张表，与知识库 `reports`/ChromaDB 分离。普通研究、知识研究、
SSE、HITL、CLI 和追踪任务完成后都会自动归档；运行失败也会保留错误记录。
服务首次启动时还会把已有 `reports` 幂等回填为 `knowledge_legacy` 档案；历史数据缺失的
模型和中间稿字段保持为空，不会用当前配置伪造。

每条完成记录包含最终稿、问题与风格、来源、选中图片、写作/反思模型、配置白名单、
反思分数与轮次、语言和事实审校修改、Token 用量。`article_versions` 依次保存综合、
语言审校、事实审校和反思阶段的稿件，可以比较哪一步改善或损害了文章。

档案不会自动进入向量知识库。只有经过人工查看的文章才应标记为 `approved`，避免失败稿
或低质量稿成为后续研究材料。

## 查询与评价

生成响应和 SSE `report` 事件都包含 `article_id`。

```bash
# 最近的文章（摘要，不返回全文和版本）
curl "http://localhost:8000/articles?limit=20"

# 完整文章、版本历史和已有评价
curl "http://localhost:8000/articles/ARTICLE_ID"

# 策展状态：raw / candidate / approved / excluded
curl -X PATCH "http://localhost:8000/articles/ARTICLE_ID/curation" \
  -H "Content-Type: application/json" \
  -d '{"curation_status":"candidate"}'

# 添加人工评价；overall_score 使用 0-10 分
curl -X POST "http://localhost:8000/articles/ARTICLE_ID/reviews" \
  -H "Content-Type: application/json" \
  -d '{
    "overall_score": 8.0,
    "dimension_scores": {"accuracy": 9, "depth": 8, "naturalness": 7},
    "issue_tags": ["标题偏多"],
    "notes": "事实扎实，表达仍可收紧",
    "include_in_evaluation": true
  }'
```

列表接口支持 `status=completed|error`、`curation_status=raw|candidate|approved|excluded`
及 `limit`、`offset` 筛选。

## 独立模型配置

```env
# 负责写作及其他主流程节点
OPENAI_MODEL=gpt-4o

# 生产反思 Judge；留空则回退到 OPENAI_MODEL
REFLECTOR_MODEL=另一个支持当前 API 的模型名

# 离线文章质量回归 Judge；比较 Prompt 版本时应固定
ARTICLE_EVAL_JUDGE_MODEL=另一个支持当前 API 的模型名
```

建议写作模型和 Judge 至少来自不同模型系列。这样能降低模型偏爱自身措辞和推理习惯的
相关偏差，但不能替代人工评价。生产反思仍会叠加 URL、证据缺口和模板化表达等确定性
规则；Judge API 故障时只降级到这些规则并在档案中标记
`reflection_judge_degraded=true`，返回格式无法解析则继续按未通过处理。

后续回归集应只选 `approved` 且 `include_in_evaluation=true` 的人工评价文章，并结合：

1. 确定性规则指标；
2. 固定独立 Judge 的盲评或成对比较；
3. 人工的准确性、深度、自然度与图片有效性评分。
