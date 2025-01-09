from typing import Dict

import torch
import numpy as np
import lightning.pytorch as pl

import pinnstorch
from LS_Sep.utilities.plot import plot_loop_solition


# 定义网格(从数据中)
def read_data_fn(root_path: str):
    """Read and preprocess the data from the specified root path."""
    data = pinnstorch.utils.load_data(root_path, "LoopSolition.mat")
    y = data["y"].T  # Spatial variable
    t = data["t"].T  # Time variable

    u = data["xx"].real.T  # Solution u(y, t)
    v = data["xx"].imag.T  # Solution v(y, t)
    p = data["qq"].real.T  # Solution p(y, t)
    q = data["qq"].imag.T  # Solution q(y, t)

    xx = np.sqrt(u ** 2 + v ** 2)
    qq = np.sqrt(p ** 2 + q ** 2)
    return pinnstorch.data.PointCloudData(
        spatial=[y], time=[t], solution={"u": u, "v": v, "p": p, "q": q, "xx":xx, "qq":qq},
    )


mesh = pinnstorch.data.PointCloud(root_dir='LS_Sep/data/',
                                  read_data_fn=read_data_fn)

# 定义训练数据集
# 初始条件
N0 = 50
in_c = pinnstorch.data.InitialCondition(
    mesh=mesh,
    num_sample=N0,
    solution=['u', 'v', 'p', 'q']
)

# 边值条件
N_b = 50
dr_b = pinnstorch.data.DirichletBoundaryCondition(
    mesh=mesh,
    num_sample=N_b,
    solution=['u', 'v', 'p', 'q']
)

# 采样点和解
N_f = 10000
me_s = pinnstorch.data.MeshSampler(
    mesh=mesh,
    num_sample=N_f,
    collection_points=['f_u', 'f_v', 'f_p', 'f_q']
)

# 验证集
val_s = pinnstorch.data.MeshSampler(
    mesh=mesh,
    solution=['u', 'v', 'p', 'q','xx','qq']
)

# 定义NN
net = pinnstorch.models.FCN(
    layers=[2, 20, 20, 20, 20, 20, 20, 20, 20, 4],
    output_names=['u', 'v', 'p', 'q'],
    lb=mesh.lb,
    ub=mesh.ub
)


# output_fn
def output_fn(outputs: Dict[str, torch.Tensor], y: torch.Tensor, t: torch.Tensor):
    """Define output transformations by separating real and imaginary parts."""
    outputs["xx"] = torch.sqrt(outputs["u"] ** 2 + outputs["v"] ** 2)

    outputs["qq"] = torch.sqrt(outputs["p"] ** 2 + outputs["q"] ** 2)

    return outputs


# pde_fn
def pde_fn(outputs: Dict[str, torch.Tensor],
           y: torch.Tensor,
           t: torch.Tensor):
    """Define the PDE residuals for real and imaginary parts."""
    # Gradients for xx (实部和虚部分开计算)
    u_y, u_t = pinnstorch.utils.gradient(outputs["u"], [y, t])
    v_y, v_t = pinnstorch.utils.gradient(outputs["v"], [y, t])

    u_yt = pinnstorch.utils.gradient(u_y, t)[0]
    v_yt = pinnstorch.utils.gradient(v_y, t)[0]

    # Gradients for qq (实部和虚部分开计算)
    p_y, p_t = pinnstorch.utils.gradient(outputs["p"], [y, t])
    q_y, q_t = pinnstorch.utils.gradient(outputs["q"], [y, t])

    p_yt = pinnstorch.utils.gradient(p_y, t)[0]
    q_yt = pinnstorch.utils.gradient(q_y, t)[0]

    # 分离实部和虚部
    outputs["f_u"] = u_yt + 0.5 *  pinnstorch.utils.gradient(outputs["p"]**2+outputs["q"]**2, y)[0]
    outputs["f_v"] = v_yt 
    outputs["f_p"] = p_yt - outputs["p"] * u_y + outputs["q"] * v_y
    outputs["f_q"] = q_yt - outputs["q"] * u_y - outputs["p"] * v_y

    return outputs


# 数据管理 PINNDataModule
train_datasets = [me_s, in_c, dr_b]
val_dataset = val_s
datamodule = pinnstorch.data.PINNDataModule(
    train_datasets=train_datasets,
    val_dataset=val_dataset,
    pred_dataset=val_s
)

# 模型管理 PINNModule
model = pinnstorch.models.PINNModule(
    net=net,
    pde_fn=pde_fn,
    output_fn=output_fn,
    loss_fn='mse'  # 均方误差适用于实部和虚部
)

# 训练
trainer = pl.Trainer(accelerator='mps', devices=1, max_epochs=1000)
trainer.fit(model=model, datamodule=datamodule)

# 验证
trainer.validate(model=model, datamodule=datamodule)

# 绘图
preds_list = trainer.predict(model=model, datamodule=datamodule)
preds_dict = pinnstorch.utils.fix_predictions(preds_list)



plot_loop_solition(
    mesh=mesh,
    preds=preds_dict,
    file_name='out'
)