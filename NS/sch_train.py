from typing import Dict

import torch
import numpy as np
import lightning.pytorch as pl

import pinnstorch


#定义网格(从数据中)
def read_data_fn(root_path):
    """Read and preprocess data from the specified root path.

    :param root_path: The root directory containing the data.
    :return: Processed data will be used in PointCloud class.
    """

    data = pinnstorch.utils.load_data(root_path, "NLS.mat")

    x = data["x"].T  # N x 1
    t = data["tt"].T  # T x 1

    exact = data["uu"]
    exact_u = np.real(exact)  # N x T
    exact_v = np.imag(exact)  # N x T
    exact_h = np.sqrt(exact_u ** 2 + exact_v ** 2)  # N x T

    return pinnstorch.data.PointCloudData(
        spatial=[x], time=[t], solution={"u": exact_u, "v": exact_v, "h": exact_h}
    )


mesh = pinnstorch.data.PointCloud(root_dir='NS/data/',
                                  read_data_fn=read_data_fn)

#定义训练数据集
##初始条件(定义函数计算初始条件)
N0 = 50
in_c = pinnstorch.data.InitialCondition(mesh = mesh,
                                        num_sample = N0,
                                        solution = ['u', 'v'])
##周期性条件
N_b = 50
pe_b = pinnstorch.data.PeriodicBoundaryCondition(mesh=mesh,
                                                 num_sample=N_b,
                                                 derivative_order=1,
                                                 solution=['u','v'])
##collection points and solutions
N_f = 20000
me_s = pinnstorch.data.MeshSampler(mesh=mesh,
                                   num_sample=N_f,
                                   collection_points = ['f_v','f_u'])

#定义验证集合
val_s = pinnstorch.data.MeshSampler(mesh = mesh,
                                    solution = ['u', 'v', 'h'])

#定义NN
net = pinnstorch.models.FCN(layers = [2, 100, 100, 100, 100, 2],
                            output_names=['u', 'v'],
                            lb = mesh.lb,
                            ub = mesh.ub)
#output_fn
def output_fn(outputs: Dict[str, torch.Tensor],
              x: torch.Tensor,
              t: torch.Tensor):
    """Define `output_fn` function that will be applied to outputs of net."""

    outputs["h"] = torch.sqrt(outputs["u"] ** 2 + outputs["v"] ** 2)

    return outputs
#pde_fn
def pde_fn(outputs: Dict[str, torch.Tensor],
           x: torch.Tensor,
           t: torch.Tensor):
    """Define the partial differential equations (PDEs)."""
    u_x, u_t = pinnstorch.utils.gradient(outputs["u"], [x, t])
    v_x, v_t = pinnstorch.utils.gradient(outputs["v"], [x, t])

    u_xx = pinnstorch.utils.gradient(u_x, x)[0]
    v_xx = pinnstorch.utils.gradient(v_x, x)[0]

    outputs["f_u"] = u_t + 0.5 * v_xx + (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["v"]
    outputs["f_v"] = v_t - 0.5 * u_xx - (outputs["u"] ** 2 + outputs["v"] ** 2) * outputs["u"]

    return outputs

#数据管理 PINNDataModule
train_datasets = [me_s, in_c, pe_b]
val_dataset = val_s
datamodule = pinnstorch.data.PINNDataModule(train_datasets = [me_s, in_c, pe_b],
                                            val_dataset = val_dataset,
                                            pred_dataset = val_s)

#模型管理PINNModule
model = pinnstorch.models.PINNModule(net = net,
                                     pde_fn = pde_fn,
                                     output_fn = output_fn,
                                     loss_fn = 'mse')

#训练
trainer = pl.Trainer(accelerator='mps', devices=1,
                                     max_epochs=1000)
trainer.fit(model=model, datamodule=datamodule)
#验证
trainer.validate(model=model, datamodule=datamodule)
#绘图

preds_list = trainer.predict(model=model, datamodule=datamodule)
preds_dict = pinnstorch.utils.fix_predictions(preds_list)

pinnstorch.utils.plot_schrodinger(mesh=mesh,
                                  preds=preds_dict,
                                  train_datasets=train_datasets,
                                  val_dataset=val_dataset,
                                  file_name='out')