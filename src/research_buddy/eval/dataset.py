"""Langfuse Dataset 构建 - 创建测试用例"""

import logging

from langfuse import Langfuse

logger = logging.getLogger(__name__)

DATASET_NAME = "research-buddy-eval"

# 测试用例：研究问题 + 预期要点
TEST_CASES = [
    {
        "input": "LangGraph 和 LangChain 的区别是什么？",
        "expected_output": [
            "LangChain 是线性链式工作流，LangGraph 是图结构工作流",
            "LangGraph 支持循环和条件分支",
            "LangGraph 有状态管理，LangChain 状态管理较弱",
            "两者可以结合使用",
        ],
    },
    {
        "input": "Python 的 GIL 是什么？它有什么影响？",
        "expected_output": [
            "GIL 是全局解释器锁",
            "GIL 阻止多线程真正并行执行 Python 字节码",
            "GIL 对 CPU 密集型任务影响大",
            "多进程可以绕过 GIL",
        ],
    },
    {
        "input": "Docker 和 Kubernetes 的区别是什么？",
        "expected_output": [
            "Docker 是容器运行时，Kubernetes 是容器编排平台",
            "Kubernetes 管理 Docker 容器的部署和扩缩",
            "Docker 适合单机，Kubernetes 适合集群",
        ],
    },
    {
        "input": "RAG 是什么？怎么实现？",
        "expected_output": [
            "RAG 是检索增强生成",
            "RAG 结合了检索和生成两个步骤",
            "RAG 需要向量数据库存储文档",
            "RAG 可以减少 LLM 的幻觉",
        ],
    },
    {
        "input": "FastAPI 和 Flask 的区别是什么？",
        "expected_output": [
            "FastAPI 是异步框架，Flask 是同步框架",
            "FastAPI 自带 API 文档（Swagger）",
            "FastAPI 基于 ASGI，Flask 基于 WSGI",
            "FastAPI 有类型校验，Flask 需要手动校验",
        ],
    },
    {
        "input": "Git rebase 和 merge 的区别是什么？",
        "expected_output": [
            "merge 保留分支历史，rebase 重写历史",
            "rebase 产生线性提交记录",
            "merge 会产生合并提交",
            "rebase 不适合已推送到远程的分支",
        ],
    },
    {
        "input": "Transformer 的注意力机制原理是什么？",
        "expected_output": [
            "自注意力机制计算 Query、Key、Value",
            "注意力权重通过 softmax 归一化",
            "多头注意力并行计算多个注意力",
            "位置编码补充序列位置信息",
        ],
    },
    {
        "input": "什么是微服务架构？和单体架构有什么区别？",
        "expected_output": [
            "微服务将应用拆分为独立服务",
            "每个微服务可以独立部署",
            "微服务之间通过 API 通信",
            "单体架构所有功能在一个应用中",
        ],
    },
]


def _item_id(index: int) -> str:
    """测试用例的稳定 ID —— 让 create_dataset_item 变成 upsert 而不是每次追加。"""
    return f"{DATASET_NAME}-{index:02d}"


def create_dataset() -> None:
    """创建或更新 Langfuse Dataset（幂等：同一条用例只会存在一份）"""
    langfuse = Langfuse()

    # 创建 Dataset（如已存在则复用）
    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Research Buddy 评估测试集",
    )

    # 读取已有 items 只用于区分「新增」和「更新」。
    # 注意：create_dataset() 返回的是 API 的 Dataset 模型，它没有 items 属性；
    # 只有 get_dataset() 返回的 DatasetClient 才有。之前对 create_dataset()
    # 的返回值取 .items 会 AttributeError，被 except 吞掉后幂等检查形同虚设，
    # 于是每跑一次就重复插入一整份用例。
    existing_ids: set[str] = set()
    try:
        existing_ids = {item.id for item in langfuse.get_dataset(DATASET_NAME).items}
    except Exception as exc:
        logger.warning("读取 Dataset '%s' 已有条目失败，按全部新增处理: %s", DATASET_NAME, exc)

    added = 0
    for i, case in enumerate(TEST_CASES):
        item_id = _item_id(i)
        # 显式传 id → 已存在则覆盖，不存在则新建，天然幂等
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item_id,
            input=case["input"],
            expected_output=case["expected_output"],
            metadata={"index": i},
        )
        if item_id not in existing_ids:
            added += 1

    logger.info("Dataset '%s' 处理完成：新增 %d 条，更新 %d 条（共 %d 条测试用例）",
                DATASET_NAME, added, len(TEST_CASES) - added, len(TEST_CASES))


def get_dataset():
    """获取 Langfuse Dataset"""
    langfuse = Langfuse()
    return langfuse.get_dataset(DATASET_NAME)


if __name__ == "__main__":
    create_dataset()
