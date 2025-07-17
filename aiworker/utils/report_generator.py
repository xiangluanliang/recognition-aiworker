import os
import logging
from openai import OpenAI
from tabulate import tabulate
from aiworker.config import DEEPSEEK_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (ReportGenerator) - %(message)s')
logger = logging.getLogger(__name__)

class ReportGeneratorService:
    """
    一个封装了 DeepSeek API 调用和文本生成逻辑的服务类，包含表格生成。
    """
    DEEPSEEK_API_BASE = "https://api.deepseek.com"
    DEEPSEEK_MODEL = "deepseek-chat"

    def __init__(self):
        """
        初始化服务，配置 DeepSeek API 客户端。
        """
        self.client = None
        logger.info("开始配置 DeepSeek API 客户端...")
        try:
            api_key = DEEPSEEK_API_KEY
            if not api_key:
                raise ValueError("未找到 DEEPSEEK_API_KEY 环境变量。请设置你的 DeepSeek API 密钥。")
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.DEEPSEEK_API_BASE
            )
            logger.info("DeepSeek API 客户端初始化成功。")
        except Exception as e:
            logger.error(f"初始化 DeepSeek API 客户端失败: {e}", exc_info=True)
            raise ConnectionError("Failed to initialize DeepSeek API client.")

    def build_chat_prompt(self, summary_data: dict) -> list[dict]:
        """
        根据摘要数据构建用于聊天的 prompt 列表 (messages)。

        Args:
            summary_data: 包含监控统计数据的字典。

        Returns:
            一个列表，包含 system 和 user 角色的消息字典。
        """
        prompt_lines = [
            "你是一个安防监控系统的智能助手，请根据以下监控统计数据生成一段简明扼要的中文日报。",
            "内容应包括以下四部分（使用小标题分段）：①事件总体情况，②事件类型分布，③摄像头状态，④风险提示建议。",
            "注意：事件类型分布部分将由系统以表格形式呈现，你无需生成该部分的描述，只需对其他部分生成自然语言内容。",
            "不要编造我未提供的信息，不要做过多主观猜测。",
            "特别说明：'区域入侵' 是指检测到人员进入了不允许进入的安全区域。\n\n",
            "数据摘要如下："
        ]

        event_stats = summary_data.pop('各类型事件统计', {})
        for key, value in summary_data.items():
            prompt_lines.append(f"- {key}: {value}")

        prompt_lines.append("\n请输出一段自然语言格式的中文报告，包含事件总体情况、摄像头状态和风险提示建议。事件类型分布部分将由系统另行处理为表格。")

        user_prompt = "\n".join(prompt_lines)

        messages = [
            {"role": "system", "content": "你是一个安防监控系统的智能助手，负责生成专业的安防日报。"},
            {"role": "user", "content": user_prompt}
        ]
        return messages, event_stats  # 返回 messages 和 event_stats

    def generate_text(self, messages: list[dict]) -> str:
        """
        使用 DeepSeek API 根据输入的 messages 生成文本。

        Args:
            messages: 包含聊天历史的列表。

        Returns:
            由 DeepSeek API 生成的报告内容字符串。
        """
        if not self.client:
            raise RuntimeError("服务未正确初始化，DeepSeek API 客户端不可用。")

        try:
            response = self.client.chat.completions.create(
                model=self.DEEPSEEK_MODEL,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                stream=False
            )
            report_content = response.choices[0].message.content.strip()
            return report_content
        except Exception as e:
            logger.error(f"调用 DeepSeek API 失败: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate text with DeepSeek API: {e}")

try:
    report_service_instance = ReportGeneratorService()
except Exception as e:
    logger.error(f"无法实例化 ReportGeneratorService: {e}")
    report_service_instance = None

def process_report_generation(summary_data: dict) -> str:
    """
    处理日报生成的核心流程，包含表格生成。
    """
    if report_service_instance is None:
        raise ConnectionError("ReportGeneratorService failed to initialize.")

    # 获取 prompt 和事件统计
    messages, event_stats = report_service_instance.build_chat_prompt(summary_data.copy())
    logger.info(f"为AI生成的聊天消息: {messages}")

    logger.info(f"hhhhhhhhh不是这里的问题")
    # 生成表格（事件类型分布）
    table_content = ""
    if event_stats:
        table_data = [
            [
                event_type,
                data["count"],
                data["unprocessed"],
                data["processing"],
                data["processed"],
                data["cameras"],
            ]
            for event_type, data in event_stats.items()
        ]
        table_content = tabulate(
            table_data,
            headers=["事件类型", "发生次数", "未处理数", "处理中数", "已处理数", "涉及摄像头数"],
            tablefmt="github",
            stralign="center",
            numalign="center"
        )
    else:
        table_content = "无事件记录"

    logger.info(f"ahahaha准备开始生成咯")
    # 生成 AI 文本
    report_content = report_service_instance.generate_text(messages)
    logger.info(f"从AI生成的文本: '{report_content}'")

    # 组合最终报告
    final_report = ""
    if report_content:
        # 插入表格到事件类型分布部分
        sections = report_content.split("###")
        if len(sections) >= 2:
            # 假设 AI 生成了小标题分段的内容，插入表格到“事件类型分布”部分
            final_report += sections[1].strip() + "\n\n"  # 事件总体情况
            final_report += "### 事件类型分布\n" + table_content + "\n\n"
            final_report += "\n".join(sections[2:]).strip()  # 剩余部分（摄像头状态、风险提示建议）
        else:
            final_report += report_content + "\n\n### 事件类型分布\n" + table_content
    else:
        logger.warning("AI模型返回了空内容，将使用备用报告。")
        online_cameras = summary_data.get('在线摄像头', 'N/A')
        total_cameras = summary_data.get('摄像头总数', 'N/A')
        final_report += (
            "本日系统运行平稳，未监测到任何安防事件记录。一切正常。\n\n"
            f"### 事件类型分布\n{table_content}\n\n"
            f"### 摄像头状态\n{online_cameras}/{total_cameras} 台在线。\n\n"
            "### 风险提示建议\n当前无明显风险，建议继续保持设备正常运行。"
        )

    return final_report
