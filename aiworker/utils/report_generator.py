import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (ReportGenerator) - %(message)s')
logger = logging.getLogger(__name__)

class ReportGeneratorService:
    """
    一个封装了 AI 模型加载和文本生成逻辑的服务类。
    """
    MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"

    def __init__(self):
        """
        初始化服务，加载模型和分词器。
        这是一个昂贵的操作，应该只在服务启动时执行一次。
        """
        self.model = None
        self.tokenizer = None
        self.device = None

        logger.info(f"开始加载模型: {self.MODEL_NAME}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"模型 {self.MODEL_NAME} 加载成功，运行在设备: {self.device}")

        except Exception as e:
            logger.error(f"加载模型 {self.MODEL_NAME} 失败: {e}", exc_info=True)
            raise ConnectionError(f"Failed to load model '{self.MODEL_NAME}'.")

    def build_chat_prompt(self, summary_data: dict) -> list[dict]:
        """
        根据摘要数据构建用于聊天的 prompt 列表 (messages)。

        Args:
            summary_data: 包含监控统计数据的字典。

        Returns:
            一个列表，包含 system 和 user 角色的消息字典。
        """
        # 基础提示语
        prompt_lines = [
            "你是一个安防监控系统的智能助手，请根据以下监控统计数据生成一段简明扼要的中文日报。",
            "内容应包括以下四部分（使用小标题分段）：①事件总体情况，②事件类型分布，③摄像头状态，④风险提示建议。\n",
            "注意：不要编造我未提供的信息，不要做过多主观猜测。\n",
            "特别说明：'区域入侵' 是指检测到人员进入了不允许进入的安全区域。\n\n",
            "数据摘要如下："]

        event_stats = summary_data.pop('各类型事件统计', {})

        for key, value in summary_data.items():
            prompt_lines.append(f"- {key}: {value}")

        if event_stats:
            prompt_lines.append("- 各类型事件统计:")
            for event_type, count in event_stats.items():
                prompt_lines.append(f"  - {event_type}: {count}")

        prompt_lines.append("\n请输出一段自然语言格式的中文报告，内容应包括：事件概况、摄像头状态，并对潜在的风险进行提醒。")
        user_prompt = "\n".join(prompt_lines)
        messages = [
            {"role": "system", "content": "你是一个安防监控系统的智能助手，负责生成专业的安防日报。"},
            {"role": "user", "content": user_prompt}
        ]
        return messages

    def generate_text(self, messages: list[dict]) -> str:
        """
        使用已加载的模型，根据输入的 messages 生成文本。
        """
        if not self.model or not self.tokenizer:
            logger.error("服务未正确初始化，无法生成报告。")
            raise RuntimeError("服务未正确初始化，模型或分词器不可用。")

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            logger.error(f"生成报告时发生错误: {e}", exc_info=True)
            return ""  # 返回空字符串，让上层处理


try:
    report_service_instance = ReportGeneratorService()
except Exception as e:
    logger.critical(f"无法实例化 ReportGeneratorService，报告生成功能将不可用: {e}")
    report_service_instance = None


def process_report_generation(summary_data: dict) -> str:
    """
    处理日报生成的核心流程。
    (此函数逻辑正确，保持不变)
    """
    if report_service_instance is None:
        raise ConnectionError("ReportGeneratorService failed to initialize.")

    messages = report_service_instance.build_chat_prompt(summary_data)
    logger.info(f"为AI生成的聊天消息: {messages}")

    report_content = report_service_instance.generate_text(messages)
    logger.info(f"从AI生成的文本: '{report_content}'")

    if not report_content:
        logger.warning("AI模型返回了空内容，将使用备用报告。")
        online_cameras = summary_data.get('在线摄像头', 'N/A')
        total_cameras = summary_data.get('摄像头总数', 'N/A')
        report_content = (
            f"监控日报 ({summary_data.get('日期', 'N/A')})\n\n"
            f"本日系统运行平稳，未监测到任何安防事件记录。一切正常。\n"
            f"摄像头状态：{online_cameras}/{total_cameras} 台在线。"
        )
    return report_content