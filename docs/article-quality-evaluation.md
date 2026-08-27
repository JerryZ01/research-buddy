# 文章质量回归评测

这套评测解决的问题不是“代码有没有报错”，而是“修改写作规则以后，同一批文章是否整体变好”。它与 `pytest` 分工如下：

- `pytest` 验证工作流、状态合并、解析和容错等程序行为。
- Article Eval 固定搜索证据，实际调用写作模型，比较文章内容、结构和文风。
- 人工审阅最终决定是否发布 Prompt；Judge 只提供一致的第一轮筛选。

## 快速开始

先用一题、每版本一次采样检查配置：

```bash
uv run python scripts/run_article_eval.py \
  --candidate-rules eval/prompts/candidate-v2.md \
  --candidate-label candidate-v2-pipeline \
  --candidate-editorial-brief \
  --candidate-evidence-editor \
  --limit 1 \
  --samples 1
```

只复查某一道失败用例时使用 `--case-id`，可重复传入多个 ID：

```bash
uv run python scripts/run_article_eval.py \
  --candidate-rules eval/prompts/candidate-v2.md \
  --candidate-editorial-brief \
  --candidate-evidence-editor \
  --case-id comparison-docker-kubernetes \
  --samples 1
```

正式回归至少使用每版本三次采样：

```bash
uv run python scripts/run_article_eval.py \
  --candidate-rules eval/prompts/candidate-v2.md \
  --candidate-label candidate-v2-pipeline \
  --candidate-editorial-brief \
  --candidate-evidence-editor \
  --samples 3 \
  --fail-on-regression
```

结果写入 `eval/results/<时间戳>/`：

- `results.json`：完整机器可读结果、文章、指标、Judge 理由和 Prompt 摘要。
- `report.html`：完全离线的双栏文章对比，直接用浏览器打开。

`--skip-judge` 只跳过质量 Judge 和 A/B Judge，文章生成仍会调用写作模型。完整评测每题每次包含 baseline、candidate 两篇文章以及对应 Judge，先用 `--limit 1 --samples 1` 估算费用。

Judge 临时失败或调整 Judge 协议后，可复用已经生成的文章重评，避免再次支付写作费用：

```bash
uv run python scripts/rejudge_article_eval.py \
  --input eval/results/<原结果>/results.json \
  --output eval/results/<重评结果>
```

质量 Judge 和 A/B Judge 的原文引文校验失败时会自动重试一次；两次都失败仍明确标为不可用，门禁不会把缺失分数当作通过。

默认 Judge 与写作共用 `OPENAI_MODEL`。条件允许时设置 `ARTICLE_EVAL_JUDGE_MODEL` 使用不同模型，降低同一模型偏爱自身表达习惯的相关偏差。比较不同 Prompt 版本时必须固定 Judge 模型；实际写作和 Judge 模型都会写入结果文件。

## 评测结构

每道题包含：

- 稳定 `id`、问题、题型和写作风格；
- 固定的搜索证据，不在评测过程中重新调用 Tavily；
- 预期要点，用于判断内容覆盖；
- `risk_tags`，记录这道题过去容易出现的写作问题。

起始集位于 `eval/cases/starter.json`。新增线上失败样本时，应复制当时实际使用的搜索结果，而不是手写一份更容易回答的证据。不得把用户隐私、密钥或未授权全文放进仓库。

已有完整 ResearchState JSON 时，可以直接捕获：

```bash
uv run python scripts/capture_article_eval_case.py \
  --result /tmp/failed-research.json \
  --id rag-answer-repeats-001 \
  --category practice \
  --risk-tag "章节内容重复" \
  --expected "给出按层诊断顺序"
```

也可以用 `--question "..."` 主动跑一次完整研究后捕获；该模式会调用 LLM 和 Tavily。捕获工具按 URL 和内容指纹去重，并拒绝覆盖相同 ID。

## 如何读结果

六个 Judge 维度各自独立：

| 维度 | 关注点 |
|---|---|
| `content_depth` | 是否解释机制、原因和影响 |
| `evidence_fidelity` | 关键事实是否忠于冻结证据 |
| `structure_coherence` | 主线是否推进、章节是否必要 |
| `naturalness` | 是否自然，是否存在模板腔或刻意口语化 |
| `specificity` | 是否包含具体事实、机制和真实例子 |
| `reader_value` | 读者是否获得可理解、判断或行动的信息 |

A/B Judge 会用稳定哈希交换文章顺序，避免永远把 baseline 放在 A 造成位置偏差。`winner` 已映射回 `baseline`、`candidate` 或 `tie`。

确定性指标不判断文章“好不好”，只监控具体症状：模板短语、硬规则命中、反问密度、模板化粗体序号、冒号标题比例、内部证据编号泄露、段落和句长变化、重复句首。这些指标适合发现退化，不适合单独优化；刻意追求句长方差同样会制造新的 AI 味。

## 当前实验结论（2026-08-26）

`candidate-v2 + 编辑简报 + 证据编辑` 在机制、决策、实践三类各一次采样中，将平均正文从 3437 字收紧到 1935 字，确定性 AI 腔命中率从 66.7% 降到 33.3%。使用 `deepseek-v4-flash` 的三次 A/B 盲评中候选胜 2 次、负 1 次。

当前结论仍不足以发布：`deepseek-v4-pro` 和 `deepseek-v4-flash` 的绝对评分都无法稳定逐字复制文章引文，完整性门禁未通过；候选还暴露过近义重复、局部编辑标点残留和内部 E 编号泄露。相关上下文保护和编号检测已补入代码，但尚未完成每题至少三次采样的正式回归。因此候选管线保持评测专用，生产 LangGraph 不启用。

## 发布规则

建议采用以下流程：

1. 把线上问题加入用例集，并标明 `risk_tags`。
2. 保存当前生产规则为 baseline，候选修改只表达一个清楚假设。
3. 先跑 `--limit 1 --samples 1`，确认格式与模型可用。
4. 全量至少跑三次采样，查看门禁、逐题退化和 A/B 理由。
5. 人工盲审所有退化题，以及随机抽取至少 20% 的胜出题。
6. 只有没有严重事实退化、候选败率可接受且人工确认后，才把候选合入生产规则。

默认门禁会阻止以下变化：任一 Judge 维度平均下降超过 0.25、AI 模板硬规则命中率上升，或至少三次有效盲评时候选败率超过 40%。门禁是最低线，不代表自动批准发布。

## 基线管理

默认 baseline 是运行时当前代码中的 `WRITING_RULES`。需要比较历史版本时，把当时规则保存为文本文件并传入：

```bash
uv run python scripts/run_article_eval.py \
  --baseline-rules eval/prompts/baseline-2026-08.md \
  --baseline-label production-2026-08 \
  --candidate-rules eval/prompts/candidate-v2.md \
  --candidate-label candidate-v2 \
  --samples 3
```

应用启动默认不再自动发布 Prompt。需要先同步到 Langfuse `staging` 检查时运行：

```bash
LANGFUSE_PROMPT_REGISTER_LABEL=staging \
  uv run python -m research_buddy.eval.prompts
```

回归和人工审阅通过后，再用 `LANGFUSE_PROMPT_REGISTER_LABEL=production` 显式发布。评测结果会保留 Prompt SHA-256 和完整规则快照，确保报告可以追溯到实际文本。兼容旧部署时可以设置 `LANGFUSE_AUTO_REGISTER_PROMPTS=true` 恢复启动同步，但不建议在生产环境使用。
