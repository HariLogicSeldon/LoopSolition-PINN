# LoopSolition-PINN

基于 PyTorch、Lightning 和 PINNs-Torch 的物理信息神经网络实验项目，用于研究短脉冲相关的环孤子、双孤子、呼吸子与怪波解。每组实验包含解析数据生成、PDE 约束训练和预测可视化。

仓库沿用已有 `Solition` 拼写，以保持路径和导入兼容。当前为研究脚本集合，尚无统一命令行入口或经过验证的依赖锁文件。

## 实验目录

| 目录 | 实验 | 数据文件 | 默认最大 epochs |
| --- | --- | --- | --- |
| `ShortPulseFunc/` | 单环孤子实验 | `data/LoopSolition.mat` | 5,000 |
| `TwoSolition/` | 双孤子实验 | `data/MultiSolition.mat` | 20,000 |
| `Breather/` | 呼吸子实验 | `data/Breather.mat` | 16,000 |
| `RogueWave/` | 怪波实验 | `data/RogueWave.mat` | 16,000 |

每个目录中的 `train.py` 定义采样、网络、PDE 残差和训练流程；`utilities/data_generate.py` 生成解析数据，`utilities/plot.py` 输出预测对比图和 CSV。`TwoSolition/utilities/net.py` 是自定义网络实验，当前训练入口使用的是 `pinnstorch.models.FCN`。

## 开始使用

在仓库根目录创建独立 Python 环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 根据源码导入整理，未锁定版本，首次安装后的训练兼容性需要验证。现有训练脚本默认使用 Apple Silicon 的 `mps`；其他设备需要先按[运行指南](docs/GETTING_STARTED.md)调整配置。

仓库保留四份 `.mat` 输入数据，可直接用于实验。以下命令从仓库根目录运行，在子进程中切换到实验目录，并将仓库根目录加入模块搜索路径：

```bash
(cd Breather && PYTHONPATH="..${PYTHONPATH:+:$PYTHONPATH}" python -m Breather.train)
```

把命令中的两处 `Breather` 同时替换为其他实验目录即可。训练时间取决于设备和配置；完整运行会进行训练、验证、预测并导出图表。

## 文档

- [环境安装、训练、日志与故障排查](docs/GETTING_STARTED.md)
- [模型结构、PDE 残差与当前限制](docs/ARCHITECTURE.md)
- [数据格式、生成方法与复现记录](docs/DATA_AND_REPRODUCIBILITY.md)
- [协作与提交约定](CONTRIBUTING.md)
- [变更记录](CHANGELOG.md)

## 版本管理与验证范围

源码、文档和四份输入 `.mat` 纳入 Git；训练日志、检查点、生成图表、缓存与 IDE 设置保留在本地并忽略。已有提交历史可能仍包含历史实验输出，新增忽略规则不会删除历史对象。

2026-09-07 的整理验证覆盖 Python 语法，以及四份输入数据的键、维度和有限值。当前系统 Python 缺少 PyTorch、Lightning、PINNs-Torch 和 TensorBoard，未执行完整训练，也未验证数值精度或收敛效果。已知实现差异见模型文档。

本项目按私有研究仓库管理，尚未指定开源许可证；第三方依赖遵循各自的许可证。
