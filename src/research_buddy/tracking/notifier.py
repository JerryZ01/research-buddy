"""智能通知 — Webhook 推送（企业微信/钉钉/Telegram 等）

设计原则：
1. 支持任何 Webhook URL（企业微信、钉钉、Telegram、Slack 等通用）
2. 自动识别 Webhook 类型，生成对应格式
3. 频率控制：同一主题每 12 小时最多推送 1 次
4. 配置简单：只需设置 NOTIFICATION_WEBHOOK_URL 环境变量
"""

import json
import logging
from datetime import datetime, timedelta

import httpx

from research_buddy.config import NOTIFICATION_WEBHOOK_URL
from research_buddy.utils import SIGNIFICANCE_EMOJI, CHANGE_TYPE_LABEL

logger = logging.getLogger(__name__)


class Notifier:
    """通知推送器

    支持：
    - 企业微信机器人 Webhook
    - 钉钉机器人 Webhook
    - Telegram Bot Webhook
    - 通用 Webhook（POST JSON）
    """

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or NOTIFICATION_WEBHOOK_URL
        self._last_sent: dict[str, datetime] = {}  # topic_id -> last_sent_time
        self._cooldown = timedelta(hours=12)  # 同一主题冷却时间

    def send_change_notification(self, topic_name: str, topic_id: str,
                                 changes: list[dict]) -> bool:
        """发送变化通知

        Args:
            topic_name: 主题名称
            topic_id: 主题 ID
            changes: 变更列表

        Returns: 是否发送成功
        """
        if not self.webhook_url:
            logger.info("通知跳过：未配置 NOTIFICATION_WEBHOOK_URL")
            return False

        # 频率控制
        now = datetime.now()
        last = self._last_sent.get(topic_id)
        if last and (now - last) < self._cooldown:
            logger.info("通知跳过：主题 %s 冷却中", topic_name)
            return False

        # 构建通知内容
        payload = self._build_payload(topic_name, topic_id, changes)

        # 发送
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(self.webhook_url, json=payload)
                if resp.status_code == 200:
                    self._last_sent[topic_id] = now
                    logger.info("通知已发送: %s", topic_name)
                    return True
                else:
                    logger.warning("通知发送失败: HTTP %d", resp.status_code)
                    return False
        except Exception as e:
            logger.warning("通知发送失败: %s", e)
            return False

    def send_test_notification(self) -> bool:
        """发送测试通知"""
        if not self.webhook_url:
            logger.info("无法测试：未配置 NOTIFICATION_WEBHOOK_URL")
            return False

        payload = self._build_payload(
            topic_name="Research Buddy",
            topic_id="test",
            changes=[{
                "type": "new_info",
                "description": "这是一条测试通知，确认 Webhook 配置正确",
                "significance": "low",
            }],
        )

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(self.webhook_url, json=payload)
                return resp.status_code == 200
        except Exception as e:
            logger.warning("测试通知失败: %s", e)
            return False

    def _build_payload(self, topic_name: str, topic_id: str,
                       changes: list[dict]) -> dict:
        """构建通知 payload

        自动检测 Webhook 类型：
        - 企业微信：URL 含 qyapi.weixin.qq.com
        - 钉钉：URL 含 oapi.dingtalk.com
        - 其他：通用 JSON 格式
        """
        url = self.webhook_url

        if "qyapi.weixin.qq.com" in url:
            return self._build_wechat_payload(topic_name, changes)
        elif "oapi.dingtalk.com" in url:
            return self._build_dingtalk_payload(topic_name, changes)
        else:
            return self._build_generic_payload(topic_name, topic_id, changes)

    @staticmethod
    def _format_change_line(change: dict, markdown: bool = True) -> str:
        """格式化单条变更行"""
        sig = SIGNIFICANCE_EMOJI.get(change.get("significance", "medium"), "⚪")
        ctype = CHANGE_TYPE_LABEL.get(change.get("type", "new_info"), "变更")
        desc = change.get("description", "")
        if markdown:
            return f"{sig} **[{ctype}]** {desc}"
        else:
            return f"{sig} [{ctype}] {desc}"

    def _build_wechat_payload(self, topic_name: str,
                              changes: list[dict]) -> dict:
        """企业微信机器人格式"""
        lines = [f"## 🔔 {topic_name} 追踪更新\n"]
        for c in changes:
            lines.append(self._format_change_line(c, markdown=True))

        return {
            "msgtype": "markdown",
            "markdown": {
                "content": "\n".join(lines),
            },
        }

    def _build_dingtalk_payload(self, topic_name: str,
                                changes: list[dict]) -> dict:
        """钉钉机器人格式"""
        lines = [f"## 🔔 {topic_name} 追踪更新\n"]
        for c in changes:
            lines.append(self._format_change_line(c, markdown=False))

        return {
            "msgtype": "markdown",
            "markdown": {
                "title": f"🔔 {topic_name} 追踪更新",
                "text": "\n".join(lines),
            },
        }

    def _build_generic_payload(self, topic_name: str, topic_id: str,
                               changes: list[dict]) -> dict:
        """通用 JSON 格式"""
        return {
            "title": f"🔔 {topic_name} 追踪更新",
            "topic_id": topic_id,
            "changes": changes,
            "timestamp": datetime.now().isoformat(),
        }


# ── 全局单例 ────────────────────────────────────────────

_notifier: Notifier | None = None


def get_notifier() -> Notifier:
    """获取全局 Notifier 实例"""
    global _notifier
    if _notifier is None:
        _notifier = Notifier()
    return _notifier
