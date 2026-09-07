# 数据与复现

## 输入数据

四份输入 `.mat` 均由对应 `utilities/data_generate.py` 中的解析公式生成，并随源码纳入版本管理。以下为 2026-09-07 对当前文件的读取结果：

| 实验 | y 范围 / 数量 | t 范围 / 数量 | xx、qq 形状 |
| --- | --- | --- | --- |
| ShortPulseFunc | [-1.5, 1.5] / 256 | [-1.5, 1.5] / 256 | (256, 256) |
| TwoSolition | [-1.5, 1.5] / 256 | [-1.5, 1.5] / 256 | (256, 256) |
| Breather | [-6, 6] / 1024 | [-1.5, 1.5] / 256 | (256, 1024) |
| RogueWave | [-10, 10] / 256 | [-20, 20] / 256 | (256, 256) |

键 `y`、`t` 为 float64 行向量；`xx`、`qq` 为 complex128 数组，轴顺序是 `(t, y)`。当前四份文件均为有限值。训练读取后转置为 `(y, t)`。生成器的部分注释仍写着旧区间，以上表格和实际 `np.linspace` 参数为准。

## 重新生成

以下命令从仓库根目录执行，会覆盖对应现有 `.mat`，并显示解析解图形：

```bash
python ShortPulseFunc/utilities/data_generate.py
python TwoSolition/utilities/data_generate.py
python Breather/utilities/data_generate.py
python RogueWave/utilities/data_generate.py
```

若希望保留旧实验输入，请先复制数据再运行。无界面环境可在命令前加 `MPLBACKEND=Agg`。生成器在导入时也会执行计算和写文件，不应作为无副作用的库模块导入。

## 预测 CSV

各绘图脚本导出 `y`、`t`，以及 `u/v/p/q` 的 `_pred`、`_exact` 列，另有 `qq_pred_modulus` 和 `qq_exact_modulus`。

当前实现将场数组 reshape 为 `(len(y), len(t))` 后按默认顺序展平，但坐标使用 `tile(y, len(t))` 和 `repeat(t, len(y))`。这与 `(y, t)` 数组按行展平的坐标顺序不一致。使用 CSV 做空间定位或二次计算前，需要核验 PINNs-Torch 预测排列，并同步修正坐标或场数组顺序；不能仅凭文件已生成认定每行坐标准确。本次整理未修改绘图算法。

## 实验记录建议

每次正式实验应在独立记录中保存以下信息：

- Git commit、未提交差异和输入文件 SHA-256。
- Python 版本、依赖版本列表、系统与设备信息。
- 各采样数量、网络层、epochs、提前停止参数、实际优化器与学习率。
- 随机种子及确定性设置；现有代码没有统一种子。
- CSV/TensorBoard 的实际版本目录、终止 epoch 和验证指标。
- 误差定义、参考解析解与相应图表，避免将不同网格或参数下结果直接混用。

在仓库根目录可记录：

```bash
git rev-parse HEAD
git diff --stat
python --version
python -m pip freeze
shasum -a 256 */data/*.mat
```

训练输出默认忽略，必要时单独归档并关联 commit。Git 历史中已存在的日志仍会随历史保留；忽略规则仅控制后续跟踪。
