# week2：手写数字十分类

姓名：从国晗

本周作业设计一个多分类任务：将 8×8 灰度手写数字图片识别为 0～9。使用公开 Digits 数据集和 PyTorch 多层感知机，完成训练、验证选择、测试评估及错误分析。

完整说明见 [实验报告](实验报告.md)，核心训练代码为 [作业01.py](作业01.py)。

## 实验结果

各组均比较学习率 0.1 与 0.01，按最低验证交叉熵选取权重；固定随机种子 42，三组使用相同的训练/验证/测试划分 1257/270/270。

| 组别 | 隐藏层 | 每候选轮数 | 选中轮次 | 验证损失 | 测试结果 |
| --- | --- | --- | --- | --- | --- |
| A | 32、16 | 20 | 16 | 0.2206 | 254/270，94.07% |
| B | 16、8 | 20 | 19 | 0.4348 | 237/270，87.78% |
| C | 16、8 | 100 | 50 | 0.2233 | 253/270，93.70% |

最终按验证损失选择 A 组；测试成绩没有参与选择。A 与 C 测试仅差一张，不能据此声称存在稳定优势。默认配置和 `outputs/latest_run.txt` 均对应 A 组。

## 运行

建议 Python 3.11，以下 PowerShell 命令在本目录执行。需要安装环境时：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements_lock.txt
```

直接检查已保存的权重和结果，不必重新训练：

```powershell
.\.venv\Scripts\python.exe -X utf8 verify_experiment.py
.\.venv\Scripts\python.exe -X utf8 体验预测.py
```

重新训练 A 组（会生成新的结果目录，并更新最近一次运行指针）：

```powershell
.\.venv\Scripts\python.exe -X utf8 作业01.py
```

B 组将 `HIDDEN_SIZES` 改为 `[16, 8]`，保持 `EPOCHS = 20`；C 组保留小网络并改为 `EPOCHS = 100`。每次运行会从固定随机初始化重新训练两个学习率候选。C 组执行全部 100 轮，再恢复第 50 轮最佳权重。

核对指定历史运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 verify_experiment.py 20261022_173951_144399
.\.venv\Scripts\python.exe -X utf8 verify_experiment.py 20261022_174606_893925
```

## 文件说明

- `作业01.py`：数据处理、MLP、训练循环、验证选择和测试。
- `体验预测.py`：加载最近一次成功运行的权重，预测测试集图片；修改 `TEST_POSITION` 可查看其他样本。
- `experiment_report.py`：新训练结束后生成图表和逐次运行参考报告。
- `verify_experiment.py`：核对划分、最佳验证轮次、权重恢复、测试预测及混淆矩阵。
- `最终模型.pt`、`模型选择说明.json`：交付的 A 组模型及选择依据。
- `outputs/`：A/B/C 的原始数值记录、两学习率候选权重、数据和划分快照。
- `assets/`、`实验汇总.json`：实验报告配图、错误案例和汇总。
- `requirements_lock.txt`：实际实验环境版本。CPU 环境即可运行。

输出文件名中的学习率使用 `0p1`、`0p01` 表示 0.1、0.01，避免在文件主名中使用额外的点；只调整提交副本的文件名和对应引用，不改变权重及历史数值。

## 学习参与说明

代码框架、环境配置和报告整理有 AI 助手协助。本人按指导修改数据查看、归一化、网络宽度及轮数，亲手运行实验和预测，并提出猜测、分析结果。具体参与及局限在实验报告中说明。
