# 第三方来源与许可

本项目原创代码、文档与项目生成的示例数据采用根目录的 [MIT License](LICENSE)。下列第三方内容保留其原有许可；根目录许可不取代第三方版权、许可条件或署名。

## PINNs-Torch 派生代码

`TwoSolition/utilities/net.py` 中的 `DIYFCN` 与 PINNs-Torch 的 `FCN` 实现高度一致，包括初始化、归一化和输出组织。根据源码比对，将其作为上游派生代码保留 BSD-3-Clause 许可。当前本地版本将类名改为 `DIYFCN`，并使用 GELU 激活函数；当前训练入口未使用该自定义网络。

- 上游项目：[rezaakb/pinns-torch](https://github.com/rezaakb/pinns-torch)
- 对照源码：[pinnstorch/models/net/neural_net.py](https://github.com/rezaakb/pinns-torch/blob/main/pinnstorch/models/net/neural_net.py)
- 上游版权：Copyright (c) 2023, Reza Akbarian Bafghi
- 上游许可：[BSD 3-Clause License](https://github.com/rezaakb/pinns-torch/blob/main/LICENSE)
- 随仓库分发的许可全文：[PINNs-Torch-BSD-3-Clause.txt](LICENSES/PINNs-Torch-BSD-3-Clause.txt)

来源和许可于 2026-09-07 核对；原始引入 commit 未记录，上述链接指向上游 main，内容可能继续更新。

## 外部依赖

`requirements.txt` 声明的 PyTorch、Lightning、PINNs-Torch、NumPy、SciPy、Matplotlib、pandas、seaborn 和 TensorBoard 由安装环境提供，遵循各自版本随附的许可证。本项目的 MIT License 不对这些依赖重新授权。分发包含这些库的环境、容器或二进制产物时，应一并保留相应许可证及声明。
