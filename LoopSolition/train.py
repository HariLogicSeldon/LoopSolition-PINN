from typing import Any, Dict, Optional
import torch
from omegaconf import DictConfig
import hydra
import pinnstorch

def read_data_fn(root_path: str):
    """Read and preprocess the data from the specified root path."""
    data = pinnstorch.utils.load_data(root_path, "LoopSolition.mat")
    y = data["y"]  # Spatial variable
    t = data["t"]  # Time variable
    xx = data["xx"]  # Solution xx(y, t)
    qq = data["qq"]  # Solution qq(y, t)
    return pinnstorch.data.PointCloudData(
        spatial=[y], time=[t], solution={"xx": xx, "qq": qq}
    )

def output_fn(outputs: Dict[str, torch.Tensor], y: torch.Tensor, t: torch.Tensor):
    """Define output transformations."""
    # Compute derived outputs if needed (e.g., transformations).
    outputs["Re_xx"] = torch.real(outputs["xx"])
    outputs["abs_qq"] = torch.abs(outputs["qq"])
    return outputs

def pde_fn(outputs: Dict[str, torch.Tensor],
           y: torch.Tensor,
           t: torch.Tensor,
           extra_variables: Dict[str, torch.Tensor]):
    """Define the PDE residuals."""
    # Gradients for xx
    xx_y, xx_t = pinnstorch.utils.gradient(outputs["xx"], [y, t])
    xx_yy = pinnstorch.utils.gradient(xx_y, y)[0]

    # Gradients for qq
    qq_y, qq_t = pinnstorch.utils.gradient(outputs["qq"], [y, t])
    qq_yy = pinnstorch.utils.gradient(qq_y, y)[0]

    # Residuals for the PDE system
    outputs["f_xx"] = xx_yy + 0.5 * pinnstorch.utils.gradient(outputs["qq"] * torch.conj(outputs["qq"]), y)[0]
    outputs["f_qq"] = qq_t - outputs["qq"] * xx_y

    return outputs

@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training."""
    # Apply extra utilities
    pinnstorch.utils.extras(cfg)

    # Train the model
    metric_dict, _ = pinnstorch.train(
        cfg,
        read_data_fn=read_data_fn,
        pde_fn=pde_fn,
        output_fn=output_fn
    )

    # Retrieve the optimized metric value
    metric_value = pinnstorch.utils.get_metric_value(
        metric_dict=metric_dict,
        metric_names=cfg.get("optimized_metric")
    )

    return metric_value

if __name__ == "__main__":
    main()