"""完整参考实验。核心训练循环在 train_one_experiment，按注释分段阅读。"""

import copy
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent

# 默认复现 A 组；B/C 组配置与原始结果见 README 和实验报告。
SEED = 42
EPOCHS = 20           # 全部训练样本学一遍，叫一轮（epoch）。
BATCH_SIZE = 32       # 每用一小批数据，更新一次参数。
HIDDEN_SIZES = [32, 16]
LEARNING_RATES = [0.1, 0.01]  # 对照实验只改变学习率，其余条件相同。


def reset_random_seed():
    """固定起点和洗牌顺序，便于比较；不代表多个随机种子的统计结论。"""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def build_model(hidden_sizes):
    """输入 64 个数，最终输出 10 个原始分数（logits）。"""
    return nn.Sequential(
        nn.Linear(64, hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(hidden_sizes[1], 10),
        # CrossEntropyLoss 接收 logits，这里不提前加 Softmax。
    )


def evaluate(model, features, labels, criterion):
    """只做检查，不反向传播，也不更新参数。"""
    model.eval()
    with torch.no_grad():
        scores = model(features)
        loss = criterion(scores, labels).item()
        predictions = scores.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
    return loss, accuracy, predictions.numpy()


def train_one_experiment(x_train, y_train, x_val, y_val, learning_rate, run_dir):
    reset_random_seed()
    model = build_model(HIDDEN_SIZES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    dataset = TensorDataset(x_train, y_train)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=generator, num_workers=0)

    config = {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
              "hidden_sizes": HIDDEN_SIZES, "learning_rate": learning_rate,
              "optimizer": "SGD", "device": "cpu", "normalization": "divide_by_16"}
    initial_loss, initial_accuracy, _ = evaluate(model, x_val, y_val, criterion)
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_features, batch_labels in loader:
            # 课件第 38 页的四步：预测、算损失、反向传播、更新参数。
            optimizer.zero_grad()                 # 清空上一次留下的梯度。
            scores = model(batch_features)        # 1. 前向传播。
            loss = criterion(scores, batch_labels)  # 2. 计算交叉熵损失。
            if not torch.isfinite(loss):
                raise RuntimeError("损失出现 NaN/无穷大，请检查学习率和输入。")
            loss.backward()                       # 3. 计算各个参数的梯度。
            optimizer.step()                      # 4. 按梯度更新参数。

        # 每轮结束后，用同一个当前模型检查完整训练集和验证集。
        # 因而这里记录的训练损失是全训练集平均值，不是最后一个 batch。
        train_loss, train_accuracy, _ = evaluate(model, x_train, y_train, criterion)
        val_loss, val_accuracy, _ = evaluate(model, x_val, y_val, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_accuracy": train_accuracy, "val_loss": val_loss,
                        "val_accuracy": val_accuracy})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 20 == 0:
            print(f"lr={learning_rate:g}  epoch={epoch:3d}/{EPOCHS}  "
                  f"训练loss={train_loss:.4f}  验证loss={val_loss:.4f}  "
                  f"验证准确率={val_accuracy:.2%}", flush=True)

    model.load_state_dict(best_state)
    rate_tag = f"{learning_rate:g}".replace(".", "p")
    checkpoint_path = run_dir / f"lr_{rate_tag}_best.pt"
    torch.save({"state_dict": best_state, "config": config, "best_epoch": best_epoch},
               checkpoint_path)
    history_path = run_dir / f"lr_{rate_tag}_history.csv"
    with history_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result = {"config": config, "initial_val_loss": initial_loss,
              "initial_val_accuracy": initial_accuracy, "best_epoch": best_epoch,
              "best_val_loss": best_val_loss,
              "best_val_accuracy": history[best_epoch - 1]["val_accuracy"],
              "elapsed_seconds": time.perf_counter() - started,
              "checkpoint_file": checkpoint_path.name, "history_file": history_path.name}
    return model, result


def main():
    if EPOCHS < 1 or BATCH_SIZE < 1 or not LEARNING_RATES:
        raise ValueError("训练轮数、批大小和学习率列表不能为空或小于 1。")
    if len(HIDDEN_SIZES) != 2 or any(size < 1 for size in HIDDEN_SIZES):
        raise ValueError("HIDDEN_SIZES 需要包含两个正整数。")
    if any(rate <= 0 for rate in LEARNING_RATES) or len(set(LEARNING_RATES)) != len(LEARNING_RATES):
        raise ValueError("学习率必须大于 0，且不能重复。")
    torch.set_num_threads(2)
    reset_random_seed()
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "outputs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    digits = load_digits()
    raw_features = digits.data.astype(np.float32)
    labels = digits.target.astype(np.int64)
    # 数据说明明确像素取值 0～16，所以除以固定常数 16，无需用全数据计算统计量。
    features = raw_features / 16.0
    all_indices = np.arange(len(labels))
    train_indices, heldout_indices = train_test_split(
        all_indices, test_size=0.30, random_state=SEED, stratify=labels)
    val_indices, test_indices = train_test_split(
        heldout_indices, test_size=0.50, random_state=SEED,
        stratify=labels[heldout_indices])
    # 记录并检查三个集合不交叉，完整覆盖数据。
    assert not set(train_indices) & set(val_indices)
    assert not set(train_indices) & set(test_indices)
    assert not set(val_indices) & set(test_indices)
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(labels)
    np.savez_compressed(run_dir / "data_and_splits.npz", raw_features=raw_features,
                        labels=labels, train_indices=train_indices,
                        val_indices=val_indices, test_indices=test_indices)
    x_train = torch.from_numpy(features[train_indices])
    y_train = torch.from_numpy(labels[train_indices])
    x_val = torch.from_numpy(features[val_indices])
    y_val = torch.from_numpy(labels[val_indices])
    print(f"数据划分：训练 {len(train_indices)}，验证 {len(val_indices)}，测试 {len(test_indices)}")
    print("开始比较学习率；选择模型时只使用验证集。", flush=True)

    models = []
    experiments = []
    for learning_rate in LEARNING_RATES:
        model, result = train_one_experiment(
            x_train, y_train, x_val, y_val, learning_rate, run_dir)
        models.append(model)
        experiments.append(result)

    selected_index = min(range(len(experiments)),
                         key=lambda index: experiments[index]["best_val_loss"])
    selected_model = models[selected_index]
    selected = experiments[selected_index]
    # 所有候选比较结束后，只对选定模型做最终测试。测试结果不参与选择。
    x_test = torch.from_numpy(features[test_indices])
    y_test = torch.from_numpy(labels[test_indices])
    test_loss, _, predictions = evaluate(selected_model, x_test, y_test, nn.CrossEntropyLoss())
    probabilities = torch.softmax(selected_model(x_test).detach(), dim=1).numpy()
    correct = int((predictions == y_test.numpy()).sum())
    majority_class = int(np.bincount(labels[train_indices], minlength=10).argmax())
    with (run_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset_index", "true_label", "predicted_label", "probability", "correct"])
        for position, dataset_index in enumerate(test_indices):
            predicted = int(predictions[position])
            writer.writerow([int(dataset_index), int(labels[dataset_index]), predicted,
                             float(probabilities[position, predicted]),
                             int(predicted == labels[dataset_index])])

    summary = {
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "execution_note": "本记录由训练脚本自动生成；实际操作人及亲手完成情况在学习记录中登记。",
        "execution_operator": "unspecified",
        "dataset": "sklearn.datasets.load_digits", "dataset_size": len(labels),
        "dataset_sha256": hashlib.sha256(raw_features.tobytes() + labels.tobytes()).hexdigest(),
        "split_sizes": {"train": len(train_indices), "validation": len(val_indices), "test": len(test_indices)},
        "class_counts": {name: np.bincount(labels[indices], minlength=10).tolist()
                         for name, indices in [("train", train_indices), ("validation", val_indices), ("test", test_indices)]},
        "unique_image_count": int(np.unique(raw_features, axis=0).shape[0]),
        "experiments": experiments, "selected_experiment_index": selected_index,
        "selection_rule": "选择验证交叉熵最低的实验及其轮次；测试集不参与选择。",
        "test": {"loss": test_loss, "accuracy": float(accuracy_score(y_test.numpy(), predictions)),
                 "macro_f1": float(f1_score(y_test.numpy(), predictions, average="macro")),
                 "correct": correct, "total": len(test_indices),
                 "majority_class": majority_class,
                 "majority_baseline_accuracy": float(np.mean(labels[test_indices] == majority_class)),
                 "confusion_matrix": confusion_matrix(y_test.numpy(), predictions, labels=list(range(10))).tolist(),
                 "classification_report": classification_report(y_test.numpy(), predictions,
                     labels=list(range(10)), output_dict=True, zero_division=0)},
        "versions": {"python": __import__("platform").python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "limitations": ["仅单次固定随机划分与单一随机种子。", "未按书写者分组，不代表跨书写者泛化评估。",
                        "适用本数据分布的 8×8 图片，不能直接保证识别手机照片或任意手写输入。",
                        "Softmax 数值未经置信度校准。"],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # 所有核心结果成功写入后，才把本次实验标记为最近一次成功实验。
    from experiment_report import create_report
    create_report(run_dir)
    (ROOT / "outputs" / "latest_run.txt").write_text(run_dir.name, encoding="utf-8")
    print(f"\n最终测试：{correct}/{len(test_indices)}，准确率 {correct / len(test_indices):.2%}")
    print(f"选中学习率 {selected['config']['learning_rate']}，第 {selected['best_epoch']} 轮。")
    print("结果目录：", run_dir)


if __name__ == "__main__":
    main()
