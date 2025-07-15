import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReportGeneratorService:
    """
    一个封装了 AI 模型加载和文本生成逻辑的服务类。
    """
    MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"

    def __init__(self):
        """
        初始化服务，加载模型和分词器。
        这是一个昂贵的操作，应该只在服务启动时执行一次。
        """
        self.model = None
        self.tokenizer = None
        logger.info(f"开始加载模型: {self.MODEL_NAME}...")
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                trust_remote_code=True
            ).eval()
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info("模型已成功加载到 CUDA (GPU)。")
            else:
                logger.info("模型已成功加载到 CPU。")

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
        prompt_lines = ["你是一个安防监控系统的智能助手，请根据以下监控统计数据生成一段简明扼要的中文日报。"]
        prompt_lines.append("数据摘要如下：")

        # 优雅地处理嵌套的事件统计
        event_stats = summary_data.pop('各类型事件统计', {})

        # 格式化主要数据
        for key, value in summary_data.items():
            prompt_lines.append(f"- {key}: {value}")

        # 格式化各类型事件统计
        if event_stats:
            prompt_lines.append("- 各类型事件统计:")
            for event_type, count in event_stats.items():
                prompt_lines.append(f"  - {event_type}: {count}")

        # 最终指令
        prompt_lines.append("\n请输出一段自然语言格式的中文报告，内容应包括：事件概况、摄像头状态，并对潜在的风险进行提醒。")

        user_prompt = "\n".join(prompt_lines)

        messages = [
            {"role": "system", "content": "你是一个安防监控系统的智能助手，负责生成专业的安防日报。"},
            {"role": "user", "content": user_prompt}
        ]
        return messages

    def generate_text(self, messages: list[dict]) -> str:
        """
        使用加载的模型，根据输入的 messages 生成文本。

        Args:
            messages: 包含聊天历史的列表。

        Returns:
            由 AI 模型生成的报告内容字符串。
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("服务未正确初始化，模型或分词器不可用。")

        # 使用 apply_chat_template 来格式化输入，这是与聊天模型交互的推荐方式
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        device = self.model.device
        inputs = self.tokenizer(input_text, return_tensors="pt").to(device)

        # 模型推理
        outputs = self.model.generate(**inputs, max_new_tokens=512)

        # 解码模型输出，并跳过 prompt 部分
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        return response.strip()


try:
    report_service_instance = ReportGeneratorService()
except Exception as e:
    logger.error(f"无法实例化 ReportGeneratorService: {e}")
    report_service_instance = None


def process_report_generation(summary_data: dict) -> str:
    """
    处理日报生成的核心流程。
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


if __name__ == '__main__':
    # 示例1: 有事件发生的情况
    print("\n" + "=" * 20 + " 测试案例 1: 有事件发生 " + "=" * 20)
    summary_data_with_events = {
        '日期': '2025-07-13',
        '总事件数': 14,
        '未处理事件数': 3,
        '处理中事件数': 4,
        '已处理事件数': 7,
        '摄像头总数': 20,
        '在线摄像头': 18,
        '各类型事件统计': {
            '人脸识别匹配': 5,
            '火警': 2,
            '区域入侵': 6,
            '人员冲突': 1,
        }
    }

    try:
        final_report_1 = process_report_generation(summary_data_with_events)
        print("\n--- ✅ 生成的最终报告 1 ---\n")
        print(final_report_1)
    except ConnectionError as e:
        print(f"\n--- ❌ 报告生成失败 ---")
        print(e)

    # 示例2: 你提供的无事件的真实输入
    print("\n\n" + "=" * 20 + " 测试案例 2: 无事件发生 " + "=" * 20)
    summary_data_no_events = {
        "日期": "2025-07-14",
        "总事件数": 0,
        "未处理事件数": 0,
        "处理中事件数": 0,
        "已处理事件数": 0,
        "摄像头总数": 4,
        "在线摄像头": 3,
        "各类型事件统计": {}
    }

    try:
        final_report_2 = process_report_generation(summary_data_no_events)
        print("\n--- ✅ 生成的最终报告 2 ---\n")
        print(final_report_2)
    except ConnectionError as e:
        print(f"\n--- ❌ 报告生成失败 ---")
        print(e)