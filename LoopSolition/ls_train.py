from typing import Dict

import torch
import numpy as np
import lightning.pytorch as pl

import pinnstorch
from plot import plot_loop_solition


#定义网格(从数据中)
def read_data_fn(root_path: str):
    """Read and preprocess the data from the specified root path."""
    data = pinnstorch.utils.load_data(root_path, "LoopSolition.mat")
    y = data["y"].T  # Spatial variable
    t = data["t"].T # Time variable

    xx = data["xx"].T  # Solution xx(y, t)
    qq = data["qq"].T # Solution qq(y, t)
    return pinnstorch.data.PointCloudData(
        spatial=[y], time=[t], solution={"xx": xx, "qq": qq}
    )


mesh = pinnstorch.data.PointCloud(root_dir='./data',
                                  read_data_fn=read_data_fn)

#定义训练数据集
##初始条件(定义函数计算初始条件)
N0 = 50
in_c = pinnstorch.data.InitialCondition(mesh = mesh,
                                        num_sample = N0,
                                        solution = ['xx', 'qq'])

# ##周期性条件
# N_b = 50
# pe_b = pinnstorch.data.PeriodicBoundaryCondition(mesh=mesh,
#                                                  num_sample=N_b,
#                                                  derivative_order=0,
#                                                  solution=['xx', 'qq'])
#collection points and solutions
N_f = 10000
me_s = pinnstorch.data.MeshSampler(mesh=mesh,
                                   num_sample=N_f,
                                   collection_points=['f_xx', 'f_qq'])

#定义验证集合
val_s = pinnstorch.data.MeshSampler(mesh=mesh,
                                    solution=['xx', 'qq'])

#定义NN
net = pinnstorch.models.FCN(layers=[2, 20,20,20,20,20,20,20,20, 2],
                            output_names=['xx', 'qq'],
                            lb=mesh.lb,
                            ub=mesh.ub)


#output_fn
def output_fn(outputs: Dict[str, torch.Tensor], y: torch.Tensor, t: torch.Tensor):
    """Define output transformations."""
    # Compute derived outputs if needed (e.g., transformations).
    outputs["Re_xx"] = torch.real(outputs["xx"])
    outputs["abs_qq"] = torch.abs(outputs["qq"])
    return outputs


#pde_fn
def pde_fn(outputs: Dict[str, torch.Tensor],
           y: torch.Tensor,
           t: torch.Tensor):
    """Define the PDE residuals."""
    # Gradients for xx
    xx_y, xx_t = pinnstorch.utils.gradient(outputs["xx"], [y, t])
    xx_yt = pinnstorch.utils.gradient(xx_y, t)[0]

    # Gradients for qq
    qq_y, qq_t = pinnstorch.utils.gradient(outputs["qq"], [y, t])
    qq_yt = pinnstorch.utils.gradient(qq_y, y)[0]

    # Residuals for the PDE system
    outputs["f_xx"] = xx_yt + 0.5 * pinnstorch.utils.gradient(outputs["qq"] * torch.conj(outputs["qq"]), y)[0]
    outputs["f_qq"] = qq_yt - outputs["qq"] * xx_y

    return outputs


#数据管理 PINNDataModule
train_datasets = [me_s, in_c]
val_dataset = val_s
datamodule = pinnstorch.data.PINNDataModule(train_datasets=[me_s, in_c],
                                            val_dataset=val_dataset,
                                            pred_dataset=val_s)

#模型管理PINNModule
model = pinnstorch.models.PINNModule(net=net,
                                     pde_fn=pde_fn,
                                     output_fn=output_fn,
                                     loss_fn='mse')

#训练
trainer = pl.Trainer(accelerator='mps', devices=1,max_epochs=1000)
trainer.fit(model=model, datamodule=datamodule)
#验证
trainer.validate(model=model, datamodule=datamodule)
#绘图

preds_list = trainer.predict(model=model, datamodule=datamodule)
preds_dict = pinnstorch.utils.fix_predictions(preds_list)

plot_loop_solition(mesh=mesh,
                  preds=preds_dict,
                  file_name='out')
