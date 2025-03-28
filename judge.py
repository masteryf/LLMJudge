import csv
import openai
import argparse
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import yaml
from typing import List, Dict


class LocalModelEvaluator:
    def __init__(self, base_url, api_key=None,
                 query_model="qwen2.5-7b-instruct-1m",
                 judge_model="qwen2.5-7b-instruct-1m"):
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key or "sk-any-string"
        )
        self.query_model = query_model
        self.judge_model = judge_model
        self.retries = 3
        self.delay = 1

    def get_response(self, prompt, model):
        """API请求带重试机制"""
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    timeout=30
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"API错误（第{attempt + 1}次尝试）: {str(e)[:100]}")
                if attempt < self.retries - 1:
                    time.sleep(self.delay)
        return None

    def query_question(self, prompt_template, question):
        """获取问题回答"""
        query_prompt = prompt_template.replace("{question}", question)
        return self.get_response(query_prompt, self.query_model)

    def judge_answer(self, judge_template, question, model_answer, correct_answer):
        """使用大模型判断答案"""
        judge_prompt = judge_template.replace("{question}", question) \
            .replace("{model_answer}", model_answer) \
            .replace("{correct_answer}", correct_answer)
        return self.get_response(judge_prompt, self.judge_model)


def load_template(template_path: Path) -> str:
    """加载提示模板文件"""
    try:
        return template_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"错误：模板文件 {template_path} 不存在")
        exit(1)


def process_dataset(dataset: Dict, query_tpl: str, judge_tpl: str, evaluator: LocalModelEvaluator,
                    output_dir: Path) -> Dict:
    """处理单个数据集"""
    csv_path = Path(dataset["path"])
    if not csv_path.exists():
        print(f"警告：跳过不存在的文件 {csv_path}")
        return None

    print(f"\n{'=' * 40}")
    print(f"处理数据集：{dataset['name']} (权重: {dataset['weight']})")

    model_score = 0
    total_score = 0
    report_path = output_dir / f"{csv_path.stem}_report.csv"

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not all(field in reader.fieldnames for field in ["question", "answer", "score"]):
                print(f"错误：文件 {csv_path} 缺少必要字段")
                return None

            # 初始化报告文件
            with open(report_path, "w", encoding="utf-8", newline="") as report_file:
                writer = csv.writer(report_file)
                writer.writerow(["Question", "CorrectAnswer", "ModelAnswer", "Judgement", "Timestamp"])

            for row in reader:
                try:
                    question = row["question"]
                    correct_answer = row["answer"]
                    score = int(row["score"])
                except (KeyError, ValueError) as e:
                    print(f"数据行格式错误: {e}")
                    continue

                total_score += score

                # 获取模型回答
                model_answer = evaluator.query_question(query_tpl, question)
                if not model_answer:
                    print("  获取回答失败，跳过此题")
                    continue

                # 判断答案
                judgement = evaluator.judge_answer(judge_tpl, question, model_answer, correct_answer)

                # 记录结果
                with open(report_path, "a", encoding="utf-8", newline="") as report_file:
                    writer = csv.writer(report_file)
                    writer.writerow([
                        question,
                        correct_answer,
                        model_answer,
                        judgement,
                        int(time.time())
                    ])

                # 累计得分
                if judgement and judgement.strip() == "正确":
                    model_score += score

        score_rate = model_score / total_score if total_score > 0 else 0
        return {
            "name": dataset["name"],
            "weight": float(dataset["weight"]),
            "model_score": model_score,
            "total_score": total_score,
            "score_rate": score_rate,
            "weighted_score": score_rate * dataset["weight"]
        }
    except Exception as e:
        print(f"处理数据集失败: {str(e)}")
        return None


def generate_chart(results: List[Dict], output_dir: Path) -> Path:
    """生成可视化图表"""
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(14, 8))

    names = [res["name"] for res in results]
    weights = [res["weight"] for res in results]
    score_rates = [res["score_rate"] * 100 for res in results]
    weighted_avg = sum(res["weighted_score"] for res in results) / sum(weights) * 100
    x = np.arange(len(names))

    # 柱状图
    bars = ax.bar(x, score_rates, color="steelblue", alpha=0.7)

    # 权重标记
    for i, (rect, weight) in enumerate(zip(bars, weights)):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + 2,
                f"权重: {weight}×",
                ha="center", va="bottom", fontsize=10)

    # 平均线
    ax.axhline(weighted_avg, color="darkorange", linestyle="--", linewidth=2)
    ax.text(len(names) + 0.5, weighted_avg, f" 加权平均: {weighted_avg:.1f}%",
            color="darkorange", va="center")

    ax.set_ylabel("得分率 (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("大模型测评结果 - 加权得分率分析", pad=20, fontsize=14)

    plt.tight_layout()
    chart_path = output_dir / "score_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    return chart_path


def main():
    parser = argparse.ArgumentParser(
        description="大模型加权评估系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default='config.yaml', help="配置文件路径")
    parser.add_argument("--query-prompts", default="query_prompt.txt", help="提问模板路径")
    parser.add_argument("--judge-prompts", default="judge_prompt.txt", help="判断模板路径")
    parser.add_argument("--api-key", default="sk-any-string", help="API密钥")
    parser.add_argument("--out-dir", default="reports", help="输出目录路径")

    args = parser.parse_args()

    # 初始化输出目录
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置文件
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            datasets = config["datasets"]
            for ds in datasets:
                ds["name"] = ds.get("name", Path(ds["path"]).stem)
                ds["weight"] = float(ds.get("weight", 1.0))
    except Exception as e:
        print(f"配置文件错误: {str(e)}")
        exit(1)

    # 初始化评估器
    evaluator = LocalModelEvaluator(
        base_url="http://192.168.3.192:1234/v1/",
        api_key=args.api_key
    )

    # 加载模板
    query_tpl = load_template(Path(args.query_prompts))
    judge_tpl = load_template(Path(args.judge_prompts))

    # 处理所有数据集
    results = []
    for dataset in datasets:
        result = process_dataset(dataset, query_tpl, judge_tpl, evaluator, output_dir)
        if result:
            results.append(result)

    if not results:
        print("没有成功处理的数据集")
        exit()

    # 生成报告
    total_weight = sum(res["weight"] for res in results)
    weighted_avg = sum(res["weighted_score"] for res in results) / total_weight
    summary_path = output_dir / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("大模型测评加权分析报告\n\n")
        f.write(f"{'数据集':<20}{'权重':<8}{'得分率':<10}{'加权贡献':<12}\n")
        f.write("-" * 50 + "\n")

        for res in results:
            contribution = res["score_rate"] * res["weight"]
            f.write(f"{res['name']:<20}{res['weight']:<8.1f}{res['score_rate']:<10.1%}{contribution:<12.3f}\n")

        f.write("\n" + "-" * 50 + "\n")
        f.write(f"总权重: {total_weight:.1f}\n")
        f.write(f"加权平均得分率: {weighted_avg:.1%}")

    # 生成图表
    chart_path = generate_chart(results, output_dir)

    print(f"\n{'=' * 40}")
    print(f"文本报告已保存至: {summary_path}")
    print(f"可视化图表已保存至: {chart_path}")
    print(f"加权平均得分率: {weighted_avg:.1%}")


if __name__ == "__main__":
    main()