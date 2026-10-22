"""读取实际训练结果，生成图表和报告。此文件属于辅助工具，第一课不用读。"""

import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def load_history(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def save_figure(figure, path):
    figure.tight_layout(pad=1.8, w_pad=2.0, h_pad=2.0)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def create_report(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    experiments = summary["experiments"]
    selected = experiments[summary["selected_experiment_index"]]
    test = summary["test"]
    histories = [load_history(run_dir / experiment["history_file"])
                 for experiment in experiments]
    history = histories[summary["selected_experiment_index"]]
    epochs = [row["epoch"] for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, name, color in [("train", "训练集", "#2563eb"), ("val", "验证集", "#e56b2f")]:
        axes[0].plot(epochs, [row[f"{key}_loss"] for row in history], label=name, color=color)
        axes[1].plot(epochs, [row[f"{key}_accuracy"] for row in history], label=name, color=color)
    for axis in axes:
        axis.axvline(selected["best_epoch"], color="#64748b", linestyle="--", label="选中轮次")
        axis.set_xlabel("训练轮次（Epoch）")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set(title="交叉熵损失：越低越好", ylabel="损失（Loss）")
    axes[1].set(title="准确率：答对的比例", ylabel="准确率", ylim=(0, 1.03))
    save_figure(figure, run_dir / "learning_curves.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for experiment, rows in zip(experiments, histories):
        label = f"学习率={experiment['config']['learning_rate']:g}"
        x = [row["epoch"] for row in rows]
        axes[0].plot(x, [row["val_loss"] for row in rows], label=label)
        axes[1].plot(x, [row["val_accuracy"] for row in rows], label=label)
    for axis in axes:
        axis.set_xlabel("训练轮次（Epoch）")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set(title="不同学习率：验证损失", ylabel="损失（Loss）")
    axes[1].set(title="不同学习率：验证准确率", ylabel="准确率", ylim=(0, 1.03))
    save_figure(figure, run_dir / "learning_rate_comparison.png")

    matrix = np.asarray(test["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(7, 6))
    display = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(display, ax=axis, label="图片数量")
    axis.set(xticks=range(10), yticks=range(10), xlabel="模型预测", ylabel="正确答案",
             title="测试集混淆矩阵：对角线表示预测正确")
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "#1e293b")
    save_figure(figure, run_dir / "confusion_matrix.png")

    with (run_dir / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    correct_rows = [row for row in predictions if row["correct"] == "1"]
    error_rows = [row for row in predictions if row["correct"] == "0"]
    examples = correct_rows[:6] + error_rows[:6]
    with np.load(run_dir / "data_and_splits.npz", allow_pickle=False) as snapshot:
        images = snapshot["raw_features"].reshape(-1, 8, 8)
        figure, axes = plt.subplots(math.ceil(len(examples) / 4), 4, figsize=(10, 8.3), squeeze=False)
        for axis in axes.flat:
            axis.axis("off")
        for axis, row in zip(axes.flat, examples):
            axis.imshow(images[int(row["dataset_index"])], cmap="gray", vmin=0, vmax=16)
            axis.set_title(f"答案 {row['true_label']}，预测 {row['predicted_label']}\n"
                           f"编号 {row['dataset_index']}，p={float(row['probability']):.2f}",
                           color="#137547" if row["correct"] == "1" else "#c33434", fontsize=10)
        figure.suptitle("测试实例：前 6 个正确预测和前 6 个错误预测")
        save_figure(figure, run_dir / "prediction_examples.png")

    config = selected["config"]
    hidden1, hidden2 = config["hidden_sizes"]
    parameter_count = 64 * hidden1 + hidden1 + hidden1 * hidden2 + hidden2 + hidden2 * 10 + 10
    steps = math.ceil(summary["split_sizes"]["train"] / config["batch_size"])
    lines = [
        "# 基于多层感知机的手写数字十分类实验", "",
        "本报告依据同目录 summary.json、逐轮 CSV 和测试预测自动生成。", "",
        f"运行时间：{summary['created_at']}。", "",
        f"执行情况：{summary['execution_note']} 参考代码和报告工具由 AI 助手辅助准备，理解和分析的完成情况单独记录。", "",
        "## 任务与数据", "",
        "输入一张 8×8 的手写数字灰度图片，输出 0～9 中的一个类别。每张图片只有一个正确类别。",
        "数据来自 scikit-learn 的 load_digits，共 1797 张；这是公开教学数据，不是自行采集的数据。",
        f"划分：训练 {summary['split_sizes']['train']}、验证 {summary['split_sizes']['validation']}、测试 {summary['split_sizes']['test']}。采用分层随机划分，种子为 {config['seed']}。",
        "像素值固定除以 16，将已知的 0～16 范围缩放到 0～1。无需从验证集或测试集估计均值、方差。",
        f"唯一图片数量为 {summary['unique_image_count']}；原始数据及划分已保存到 data_and_splits.npz。", "",
        "## 网络与训练设计", "",
        f"网络：64 → {hidden1} → {hidden2} → 10。两个隐藏层使用 ReLU，输出原始类别分数。参数量 {parameter_count}。",
        f"使用交叉熵和普通 SGD，batch size={config['batch_size']}，每个实验训练 {config['epochs']} 轮，每轮 {steps} 次参数更新。",
        "CrossEntropyLoss 接收 logits。Softmax 仅用于展示类别概率，不提前放进训练网络。",
        "全部训练在 CPU 上进行。两个实验使用相同初始化、数据划分和批次洗牌顺序，仅改变学习率。", "",
        "## 验证集对照实验", "",
        "| 学习率 | 未训练验证准确率 | 最佳轮次 | 最低验证损失 | 对应验证准确率 | 训练耗时（秒） |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for experiment in experiments:
        lines.append(f"| {experiment['config']['learning_rate']:g} | {experiment['initial_val_accuracy']:.2%} | "
                     f"{experiment['best_epoch']} | {experiment['best_val_loss']:.4f} | "
                     f"{experiment['best_val_accuracy']:.2%} | {experiment['elapsed_seconds']:.2f} |")
    lines += ["", f"按验证交叉熵最低的规则，选择学习率 {config['learning_rate']:g} 的第 {selected['best_epoch']} 轮权重。",
              "测试集不参与学习率或轮次选择。训练曲线记录完整数据集的平均损失，不是最后一批的损失。", "",
              "![训练曲线](learning_curves.png)", "", "![学习率对比](learning_rate_comparison.png)", "",
              "## 最终测试", "",
              f"正确 {test['correct']} / {test['total']}，错误 {test['total'] - test['correct']}。",
              f"准确率：{test['accuracy']:.2%}。宏平均 F1：{test['macro_f1']:.4f}。交叉熵：{test['loss']:.4f}。",
              f"对照：始终预测训练集中最多的类别 {test['majority_class']}，在该测试集上的准确率为 {test['majority_baseline_accuracy']:.2%}。", "",
              "| 数字 | Precision | Recall | F1 | 测试样本数 |", "| --- | --- | --- | --- | --- |"]
    for label in range(10):
        row = test["classification_report"][str(label)]
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines += ["", "Precision 表示预测为这一类的样本中有多少正确；Recall 表示真实属于这一类的样本有多少被找出来；F1 综合两者。宏平均 F1 先分别算每类，再让十类等权平均。", "",
              "![混淆矩阵](confusion_matrix.png)", "", "混淆矩阵行是真实类别、列是预测类别，对角线是预测正确的数量。", "",
              "![预测示例](prediction_examples.png)", "", "示例按测试集保存顺序取前六个正确和前六个错误样本，未挑选最好看的结果。p 是模型 Softmax 输出，未经概率校准。", "",
              "错误样本（数据集编号：真实 → 预测）：", ""]
    lines += [f"- #{row['dataset_index']}：{row['true_label']} → {row['predicted_label']}。" for row in error_rows]
    if not error_rows:
        lines.append("本次测试未出现错误；这不代表真实场景一定没有错误。")
    lines += ["", "## 结果解释与局限", "",
              "训练损失下降表示模型更适应训练样本。验证曲线用于观察对未参与参数更新的数据是否同样有效；仅训练准确率高不足以说明效果好。",
              "本实验选取两个学习率，不代表穷尽了最优配置。对比结论只适用于这里的网络、轮数、随机种子及划分。", "",
              *[f"- {limitation}" for limitation in summary["limitations"]], "",
              "## 用户需要亲自完成的学习记录", "",
              "以下内容不能根据助手的参考运行自动认定完成：", "",
              "- [ ] 运行 01_认识数据.py，修改 SAMPLE_INDEX，并解释输入与标签。",
              "- [ ] 解释 64 个输入和 10 个输出分别从哪里来。",
              "- [ ] 找到 zero_grad、前向传播、loss、backward、step，并解释它们。",
              f"- [{'x' if 'run_training' in summary.get('user_completed_steps', []) else ' '}] 按指导亲手运行训练，记录自己的结果目录。",
              "- [ ] 只改变一项设置，记录预期、实际变化及原因。",
              "- [ ] 根据真实结果补写自己的结论和仍不理解的地方。", "",
              "## 环境与复现", "",
              f"环境：{json.dumps(summary['versions'], ensure_ascii=False)}。",
              f"数据内容 SHA-256：{summary['dataset_sha256']}。",
              f"选定权重：{selected['checkpoint_file']}。所有候选的权重、逐轮 CSV、测试预测 CSV 均保存在同目录。",
              "在作业目录运行：`.\\.venv\\Scripts\\python.exe -X utf8 .\\作业01.py`。每次运行生成新目录，保留原实验。", "",
              "## 参考资料", "",
              "- 数据说明：https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
              "- PyTorch 训练循环：https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
              "- 交叉熵：https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html", ""]
    (run_dir / "实验报告_参考.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    latest = (ROOT / "outputs" / "latest_run.txt").read_text(encoding="utf-8").strip()
    create_report(ROOT / "outputs" / latest)
