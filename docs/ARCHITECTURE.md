# 模型与代码结构

## 处理流程

解析公式生成 `(y, t, xx, qq)` → `.mat` → `PointCloudData` → 初始条件、Dirichlet 边界条件及内部采样 → FCN → 自动微分计算 PDE 残差 → Lightning 训练、验证及预测 → CSV/PDF。

四个训练入口均使用 `pinnstorch.models.FCN`，网络为 `[2, 20, 20, 20, 20, 20, 20, 20, 20, 4]`：输入为 `y, t`，八个宽度为 20 的隐藏层，输出为 `u, v, p, q`。损失配置为 `mse`；优化器参数由安装的 PINNs-Torch 实现决定，源码没有显式固定学习率。创建的 `MultiStepLR` 变量未作为启用的 scheduler 传入模型。

## 变量约定

`u, v` 对应复值 `xx` 的实部、虚部；`p, q` 对应复值 `qq` 的实部、虚部。模型额外输出 `xx = sqrt(u² + v²)` 与 `qq = sqrt(p² + q²)`，这里输出字段表示模长，不是原始复数组。Breather 的读取逻辑将 `v` 显式设为零，与其他三个实验不同。

## 源码中的残差

以下为当前实现的逐项记录，不构成对解析公式或 PDE 正确性的证明。下标表示偏导数。

```text
f_u = u_yt + 0.5 * ∂y(p² + q²)
f_v = v_yt
f_p = p_yt - p*u_y + q*v_y
f_q = q_yt - p*v_y - q*u_y
```

Breather、RogueWave、TwoSolition 使用上述表达式。ShortPulseFunc 的最后一项是：

```text
f_q = q_yt - q*u_y - p*v_y
```

两种最后一项在代数上相同；目录之间仍有数据处理、采样参数和训练设置差异，应分别记录。

## 当前限制

- 未设置统一随机种子，采样和初始化不保证重复运行一致。
- 多数入口在模块顶层读取数据并创建对象；ShortPulseFunc 还在顶层启动训练。不要通过导入训练模块进行轻量检查。
- 输入路径依赖当前工作目录，默认设备为 MPS。
- `TwoSolition/utilities/net.py` 中的 `DIYFCN` 当前未接入训练。
- 预测 CSV 的坐标与展平顺序存在潜在错配，详见数据文档；用于后续分析前应验证。
- 未建立数值回归测试、依赖版本锁或四组实验的统一精度基线。

本次仓库整理保持原有实验算法不变，以上事项留作后续单独验证与改进。
