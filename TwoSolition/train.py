from typing import Dict

import torch
import numpy as np
import lightning.pytorch as pl
import pinnstorch
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from TwoSolition.utilities.plot import plot_loop_solition
from lightning.pytorch.callbacks import EarlyStopping


# Define training hyperparameters
max_epochs = 20000  # Maximum number of epochs
device = 'mps'    # Training device
early_stopping,patience = False,1000  # Early stopping settings
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 4]  # Neural network architecture
N0 = 50      # Number of initial condition samples
N_b = 50     # Number of boundary condition samples
N_f = 20000  # Number of interior points

def read_data_fn(root_path: str):
    """Read and preprocess data
    
    Args:
        root_path: Path to data file
    
    Returns:
        PointCloudData object containing preprocessed data
    """
    data = pinnstorch.utils.load_data(root_path, "MultiSolition.mat")
    y = data["y"].T  # Spatial variable
    t = data["t"].T  # Time variable
    
    # Extract real and imaginary parts
    # noise_level = 0.001
    u = data["xx"].real.T
    v = data["xx"].imag.T
    p = data["qq"].real.T
    q = data["qq"].imag.T
    
    # Calculate modulus
    xx = np.sqrt(u ** 2 + v ** 2)
    qq = np.sqrt(p ** 2 + q ** 2)
    
    return pinnstorch.data.PointCloudData(
        spatial=[y], time=[t], 
        solution={"u": u, "v": v, "p": p, "q": q, "xx": xx, "qq": qq},
    )

# Create mesh object
mesh = pinnstorch.data.PointCloud(root_dir="./data",
                                read_data_fn=read_data_fn)

# Define data samplers
in_c = pinnstorch.data.InitialCondition(  # Initial conditions
    mesh=mesh,
    num_sample=N0,
    solution=['u', 'v', 'p', 'q']
)

dr_b = pinnstorch.data.DirichletBoundaryCondition(  # Boundary conditions
    mesh=mesh,
    num_sample=N_b,
    solution=['u', 'v', 'p', 'q']
)

me_s = pinnstorch.data.MeshSampler(  # Interior points
    mesh=mesh,
    num_sample=N_f,
    collection_points=['f_u', 'f_v', 'f_p', 'f_q']
)

val_s = pinnstorch.data.MeshSampler(  # Validation set
    mesh=mesh,
    solution=['u', 'v', 'p', 'q', 'xx', 'qq']
)

def output_fn(outputs: Dict[str, torch.Tensor], y: torch.Tensor, t: torch.Tensor):
    """Define output transformation function
    
    Args:
        outputs: Model output dictionary
        y: Spatial coordinates
        t: Time coordinates
    
    Returns:
        Updated output dictionary
    """
    outputs["xx"] = torch.sqrt(outputs["u"] ** 2 + outputs["v"] ** 2)
    outputs["qq"] = torch.sqrt(outputs["p"] ** 2 + outputs["q"] ** 2)
    return outputs

def pde_fn(outputs: Dict[str, torch.Tensor], y: torch.Tensor, t: torch.Tensor):
    """Define PDE residual function
    
    Args:
        outputs: Model output dictionary
        y: Spatial coordinates
        t: Time coordinates
    
    Returns:
        Dictionary containing PDE residuals
    """
    # Calculate x gradients
    u_y, u_t = pinnstorch.utils.gradient(outputs["u"], [y, t])
    v_y, v_t = pinnstorch.utils.gradient(outputs["v"], [y, t])
    u_yt = pinnstorch.utils.gradient(u_y, t)[0]
    v_yt = pinnstorch.utils.gradient(v_y, t)[0]
    
    # Calculate q gradients
    p_y, p_t = pinnstorch.utils.gradient(outputs["p"], [y, t])
    q_y, q_t = pinnstorch.utils.gradient(outputs["q"], [y, t])
    p_yt = pinnstorch.utils.gradient(p_y, t)[0]
    q_yt = pinnstorch.utils.gradient(q_y, t)[0]
    
    # Calculate PDE residuals
    outputs["f_u"] = u_yt + 0.5 * pinnstorch.utils.gradient(outputs["p"] ** 2 + outputs["q"] ** 2, y)[0]
    outputs["f_v"] = v_yt
    outputs["f_p"] = p_yt - outputs["p"] * u_y + outputs["q"] * v_y
    outputs["f_q"] = q_yt - outputs["p"] * v_y - outputs["q"] * u_y
    
    return outputs

# Create neural network model
net = pinnstorch.models.FCN(
    layers=layers,
    output_names=['u', 'v', 'p', 'q'],
    lb=mesh.lb,
    ub=mesh.ub
)

# Set up data module
train_datasets = [me_s, in_c, dr_b]
val_dataset = val_s
datamodule = pinnstorch.data.PINNDataModule(
    train_datasets=train_datasets,
    val_dataset=val_dataset,
    pred_dataset=val_s
)

# Create PINN model
scheduler = torch.optim.lr_scheduler.MultiStepLR
model = pinnstorch.models.PINNModule(
    net=net,
    pde_fn=pde_fn,
    output_fn=output_fn,
    loss_fn='mse',
    scheduler= scheduler
)

# Record hyperparameters
hyperparams = {
    "network": {
        "layers": layers,
        "lb": mesh.lb.tolist(),
        "ub": mesh.ub.tolist(),
    },
    "sampling_points": {
        "N0": N0,
        "N_b": N_b,
        "N_f": N_f,
    },
    "training": {
        "max_epochs": max_epochs,
        "accelerator": device,
        "devices": -1,
        "early_stop": early_stopping,
        "patience": patience
    }
}
if __name__ == "__main__":
    # Set up early stopping callback
    early_stop_callback = [EarlyStopping(monitor='val/error_qq', patience=patience,stopping_threshold=0.04)] if early_stopping else None

    # Set up loggers
    csv_logger = CSVLogger(save_dir="lightning_logs", name="csv_logs")
    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name="tb_logs")
    tb_logger.log_hyperparams(hyperparams)

    logger = [csv_logger, tb_logger]

    # Create trainer and start training
    trainer = pl.Trainer(
        accelerator=device,
        devices=-1,
        max_epochs=max_epochs,
        enable_progress_bar=False,
        logger=logger,
        callbacks=early_stop_callback,
        # fast_dev_run=1
    )

    # Train model
    trainer.fit(model=model, datamodule=datamodule)

    # Validate model
    trainer.validate(model=model, datamodule=datamodule)

    # Make predictions and plot results
    preds_list = trainer.predict(model=model, datamodule=datamodule)
    preds_dict = pinnstorch.utils.fix_predictions(preds_list)
    plot_loop_solition(
        mesh=mesh,
        preds=preds_dict,
        logger=logger[1]
    )
