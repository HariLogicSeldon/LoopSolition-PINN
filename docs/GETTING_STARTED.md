# 运行指南

## 环境准备

先在仓库根目录执行 README 的虚拟环境和依赖安装命令。项目未保存原训练环境的 Python/依赖版本，因此不承诺任意最新依赖组合可直接复现历史结果。成功运行后请保存 `python --version` 和 `python -m pip freeze` 输出。

在已激活的环境中检查关键依赖：

```bash
python -c "import torch, lightning.pytorch, pinnstorch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
python -m pip check
```

## 设备与训练配置

参数直接定义在各实验的 `train.py` 中，没有命令行参数解析器。

| 实验 | N0 | N_b | N_f | max_epochs | early_stopping / patience |
| --- | --- | --- | --- | --- | --- |
| ShortPulseFunc | 50 | 50 | 20,000 | 5,000 | False / 1,000 |
| TwoSolition | 50 | 50 | 20,000 | 20,000 | False / 1,000 |
| Breather | 50 | 50 | 20,000 | 16,000 | False / 2,500 |
| RogueWave | 50 | 500 | 20,000 | 16,000 | True / 20,000 |

四组实验默认 `device = 'mps'`，Trainer 使用 `devices=-1`。若当前 Lightning 版本或设备不接受该组合，在 `pl.Trainer(...)` 中改为 `devices=1`；CPU 设置 `device='cpu'`，NVIDIA GPU 设置 `device='gpu'`。同时更新 `hyperparams` 中记录的 `devices`，使日志与实际运行一致。设备切换后的兼容性需要实际验证。

首次调试可临时降低 `N_f`、`N0`、`N_b` 和 `max_epochs`，或设置 Trainer 的 `fast_dev_run=True`。这仅用于检查执行链路，不代表模型收敛。正式实验前恢复参数，记录改动。

## 执行训练

从仓库根目录选择一条命令运行：

```bash
(cd ShortPulseFunc && PYTHONPATH="..${PYTHONPATH:+:$PYTHONPATH}" python -m ShortPulseFunc.train)
(cd TwoSolition && PYTHONPATH="..${PYTHONPATH:+:$PYTHONPATH}" python -m TwoSolition.train)
(cd Breather && PYTHONPATH="..${PYTHONPATH:+:$PYTHONPATH}" python -m Breather.train)
(cd RogueWave && PYTHONPATH="..${PYTHONPATH:+:$PYTHONPATH}" python -m RogueWave.train)
```

切换目录是因为数据读取固定为 `./data`，设置 `PYTHONPATH` 则是为了支持 `from Breather.utilities...` 等导入。直接从根目录执行 `python Breather/train.py` 不能保证这两项要求同时满足。

## 查看输出

每个实验目录内的 `lightning_logs/csv_logs/version_*/` 保存指标及可能的检查点，`lightning_logs/tb_logs/version_*/` 保存 TensorBoard 事件和预测导出。两类 logger 的版本号可能不同，应根据本次实际生成的目录关联。

```bash
tensorboard --logdir Breather/lightning_logs/tb_logs
```

预测绘图脚本输出 `rex_q.pdf`、`heatmap.pdf`、`q_3d_plot.pdf` 和 `solution_data.csv`。CSV 坐标排列有待核对，见[数据文档](DATA_AND_REPRODUCIBILITY.md)。

根目录运行 `python read.py` 会汇总 `RogueWave/lightning_logs/csv_logs` 中的验证误差，并保存 `RogueWave/figures/error_metrics.pdf`。需要先有相应训练日志；其他实验需修改脚本中的 `base_dir` 与 `output_dir`。

`Breather/reader.py` 是临时检查工具，读取相对路径 `lightning_logs/tb_logs/version_1/solution_data.csv`，运行前需在 Breather 目录内确认并调整版本号。

## 常见问题

- 找不到实验包：使用上述带 `PYTHONPATH` 的命令。
- 找不到 `.mat`：确认工作目录为对应实验目录，必要时按数据文档重新生成。
- MPS 不可用或 `devices` 报错：检查 `torch.backends.mps.is_available()`，按设备配置一节调整。
- 训练无动态进度条：脚本明确设置了 `enable_progress_bar=False`，可查看 CSV 或 TensorBoard。
- 绘图时报张量不能转 NumPy：绘图代码直接调用 `.numpy()`；需检查预测张量是否位于 CPU、是否仍带梯度，必要时在代码中进行 `detach().cpu()` 转换。
- 无图形界面的环境：可设置 `MPLBACKEND=Agg`；如 PINNs-Torch 的绘图样式触发 LaTeX 错误，检查 traceback 和实际 Matplotlib 样式设置。
