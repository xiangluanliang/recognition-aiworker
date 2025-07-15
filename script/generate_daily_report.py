# scripts/generate_daily_report.py

import os
import django
from datetime import datetime, timedelta
from django.db.models import Count
from django.utils import timezone

import sys
# print(sys.path)

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.test')
django.setup()

from api.models import EventLog, Camera, DailyReport, AlarmLog

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 替换模型加载部分：
model_name = "Qwen/Qwen1.5-1.8B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).eval()

model = model.eval()

def collect_data():
    today = timezone.localdate()
    start = timezone.make_aware(datetime.combine(today, datetime.min.time()))
    end = timezone.make_aware(datetime.combine(today, datetime.max.time()))

    alarms = AlarmLog.objects.filter(time__range=(start, end))

    summary = {
        '日期': str(today),
        '总事件数': alarms.count(),
        '未处理事件数': alarms.filter(status=0).count(),
        '处理中事件数': alarms.filter(status=1).count(),
        '已处理事件数': alarms.filter(status=2).count(),
    }

    type_counts = alarms.values('event__event_type').annotate(count=Count('id'))
    for item in type_counts:
        summary[f"类型:{item['event__event_type']}"] = item['count']

    total_cameras = Camera.objects.count()
    active_cameras = Camera.objects.filter(is_active=True).count()
    summary['摄像头总数'] = total_cameras
    summary['在线摄像头'] = active_cameras

    # summary = {
    #     '日期': '2025-07-13',
    #     '总事件数': 14,
    #     '未处理事件数': 3,
    #     '处理中事件数': 4,
    #     '已处理事件数': 7,
    #     '类型:face_match': 5,
    #     '类型:fire': 2,
    #     '类型:intrusion': 6,
    #     '类型:conflict': 1,
    #     '摄像头总数': 20,
    #     '在线摄像头': 18,
    # }

    return summary


def build_prompt(summary):
    prompt = "你是一个安防监控系统的智能助手，请根据以下监控统计数据生成一段简明扼要的中文日报：\n\n"
    for k, v in summary.items():
        prompt += f"- {k}: {v}\n"
    prompt += "\n请输出一段自然语言中文报告，包含：事件概况、类型分布、摄像头状态，若有异常请提醒。"
    return prompt.strip()

#     prompt = """
# 你是一个安防监控系统的智能助手，请根据以下数据生成一份简明的中文监控日报。
# 包括：总体事件概况、各类型事件情况、摄像头在线状态，以及必要时的风险提示。
#
# 下面是当天的监控数据摘要：
# """
#     for k, v in summary.items():
#         prompt += f"- {k}: {v}\n"
#     prompt += """
#
# 请输出一段自然语言描述，总结当天监控情况。
# """
#     return prompt.strip()


def generate_text_report():
    summary = collect_data()
    prompt = build_prompt(summary)

    print("\n=== 📥 Prompt 输入 ===\n")
    print(prompt)

    messages = [
        {"role": "system", "content": "你是一个安防监控系统的智能助手"},
        {"role": "user", "content": prompt}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 模型推理
    outputs = model.generate(**inputs, max_new_tokens=512)

    # 解码模型输出部分（去掉输入内容）
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    print("\n=== 📤 模型输出（生成的日报） ===\n")
    print(response)
    # 写入数据库
    # DailyReport.objects.update_or_create(date=timezone.localdate(), defaults={'content': content})
    # print("✅ 日报已生成并写入数据库：\n", content)


if __name__ == '__main__':
    print("\n=== 🧪 collect_data() 返回结果 ===\n")
    from pprint import pprint
    pprint(collect_data())

    generate_text_report()
