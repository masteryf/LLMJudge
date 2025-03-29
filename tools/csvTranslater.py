import pandas as pd
import openai
import os
import time
from requests.exceptions import RequestException


def translate_text(text, max_retries=3):
    """
    使用本地大模型进行专业翻译，保留专有名词
    """
    instruction = """请将以下学术内容精确翻译为中文，遵循这些规则：
1. 专有名词（人名、地名、理论名称、科技术语等）保留英文并在括号内注明
2. 数学公式保持LaTeX格式不变
3. 专业术语参照各学科官方译法
4. 保持原文的精确性和学术严谨性
5. 涉及多义词时根据上下文选择最准确的译法

翻译前请先判断内容，如果是数字，字母或者数学公式等，则不需要翻译，直接输出即可，你只需要翻译其中的英文语句部分,只翻译你确保翻译不会扭曲意思的部分，你无法理解的原文直接输出原文即可

示例转换：
原文：The Schrödinger equation is fundamental in quantum mechanics.
译文：薛定谔方程（Schrödinger equation）是量子力学（Quantum Mechanics）中的基本方程。

原文：D
译文：D

原文：yeyo
译文：yeyo

原文：114514
译文：114514

原文：$1 + 3x + 6x^2 + 8x^3 + 6x^4 + 3x^5 + x^6$
译文：$1 + 3x + 6x^2 + 8x^3 + 6x^4 + 3x^5 + x^6$


请直接输出译文，不要输出任何多余的东西
现在请翻译："""

    client = openai.Client(
        base_url="http://192.168.3.192:1234/v1",
        api_key="none"
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="qwen2.5-7b-instruct-1m",
                messages=[
                    {"role": "system", "content": "你是一位精通多学科的专业翻译官"},
                    {"role": "user", "content": instruction + text}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"请求失败，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception(f"翻译失败：{str(e)}")


def translate_csv(input_path):
    """
    处理CSV文件翻译流程
    """
    # 读取原始文件
    df = pd.read_csv(input_path)

    # 检查必要列是否存在
    required_columns = ['question', 'answer']
    if not all(col in df.columns for col in required_columns):
        missing = set(required_columns) - set(df.columns)
        raise ValueError(f"CSV文件缺少必要列：{missing}")

    # 准备新DataFrame
    translated_df = df.copy()

    # 遍历处理每个需要翻译的字段
    for index, row in df.iterrows():
        for field in ['question', 'answer']:
            original_text = row[field]
            if pd.isna(original_text):
                continue

            print(f"正在翻译 {field} {index + 1}/{len(df)}...")
            translated_text = translate_text(str(original_text))
            translated_df.at[index, field] = translated_text

    # 生成输出路径
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_Chinese.csv"

    # 保存文件时保留原始索引和其他信息
    translated_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CSV文件专业翻译工具')
    parser.add_argument('input_path', type=str, help='输入CSV文件路径')
    args = parser.parse_args()

    try:
        output = translate_csv(args.input_path)
        print(f"翻译完成！文件已保存至：{output}")
    except Exception as e:
        print(f"处理失败：{str(e)}")