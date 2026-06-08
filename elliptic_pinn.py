import os
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Global setup
# ============================================================

DTYPE = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_tensor(arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(arr, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def second_derivative(y: torch.Tensor, x: torch.Tensor, dim: int) -> torch.Tensor:
    grad_y = gradient(y, x)
    return torch.autograd.grad(
        grad_y[:, dim:dim + 1],
        x,
        grad_outputs=torch.ones_like(grad_y[:, dim:dim + 1]),
        create_graph=True,
        retain_graph=True,
    )[0][:, dim:dim + 1]


def relative_l2_error(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.linalg.norm(pred.reshape(-1) - truth.reshape(-1))
    den = np.linalg.norm(truth.reshape(-1)) + 1e-12
    return float(num / den)


# ============================================================
# Network backbone
# ============================================================

class MLPBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 4,
        activation: str = "tanh",
    ):
        super().__init__()

        if activation == "tanh":
            act = nn.Tanh
        elif activation == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]

        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Elliptic benchmark
#
#     -Delta u + u = f      in Omega = (0,1)^2
#      u = 0               on boundary
#
# Exact solution used only for evaluation:
#
#     u(x,y) = sin(pi x) sin(pi y)
#
# Forcing:
#
#     f(x,y) = (2 pi^2 + 1) sin(pi x) sin(pi y)
#
# PDE-derived structure:
#
#     1. zero Dirichlet boundary
#     2. positivity from the minimum principle
#     3. full square symmetry:
#        x <-> y,
#        x -> 1-x,
#        y -> 1-y,
#        and all compositions of these.
# ============================================================

class EllipticProblem:
    name = "elliptic"
    input_dim = 2

    def normalize_inputs(self, z: torch.Tensor) -> torch.Tensor:
        return 2.0 * z - 1.0

    @staticmethod
    def u_exact_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_solution(self, z: np.ndarray) -> np.ndarray:
        x = z[:, 0]
        y = z[:, 1]
        return self.u_exact_xy(x, y)[:, None]

    def forcing_numpy(self, z: np.ndarray) -> np.ndarray:
        x = z[:, 0]
        y = z[:, 1]
        f = (2.0 * np.pi**2 + 1.0) * np.sin(np.pi * x) * np.sin(np.pi * y)
        return f[:, None]

    def forcing_torch(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        y = z[:, 1:2]
        return (2.0 * torch.pi**2 + 1.0) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    def sample_interior(self, n: int) -> np.ndarray:
        return np.random.rand(n, 2)

    def sample_boundary(self, n: int) -> np.ndarray:
        n_side = n // 4

        y_left = np.random.rand(n_side, 1)
        y_right = np.random.rand(n_side, 1)
        x_bottom = np.random.rand(n_side, 1)
        x_top = np.random.rand(n - 3 * n_side, 1)

        left = np.hstack([np.zeros_like(y_left), y_left])
        right = np.hstack([np.ones_like(y_right), y_right])
        bottom = np.hstack([x_bottom, np.zeros_like(x_bottom)])
        top = np.hstack([x_top, np.ones_like(x_top)])

        return np.vstack([left, right, bottom, top])

    def sample_observations(self, n: int) -> np.ndarray:
        return self.sample_interior(n)

    def heldout_residual_points(self, n: int = 4096) -> np.ndarray:
        return self.sample_interior(n)

    def evaluation_grid(self, nx: int = 161, ny: int = 161) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        xs = np.linspace(0.0, 1.0, nx)
        ys = np.linspace(0.0, 1.0, ny)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        Z = np.column_stack([X.ravel(), Y.ravel()])
        return Z, {"x": xs, "y": ys, "X": X, "Y": Y}

    def extra_metrics(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        grid_meta: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        xs = grid_meta["x"]
        ys = grid_meta["y"]

        U_pred = pred.reshape(len(ys), len(xs))
        U_true = truth.reshape(len(ys), len(xs))

        boundary_mask = np.zeros_like(U_pred, dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True

        boundary_vals = U_pred[boundary_mask]
        boundary_l2_abs = float(np.sqrt(np.mean(boundary_vals**2)))
        boundary_max_abs = float(np.max(np.abs(boundary_vals)))

        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        uy_pred, ux_pred = np.gradient(U_pred, dy, dx, edge_order=2)
        uy_true, ux_true = np.gradient(U_true, dy, dx, edge_order=2)

        h1_semi_error = np.sqrt(
            np.sum((ux_pred - ux_true) ** 2 + (uy_pred - uy_true) ** 2) * dx * dy
        )
        h1_semi_ref = np.sqrt(
            np.sum(ux_true**2 + uy_true**2) * dx * dy
        ) + 1e-12

        negative_part = np.maximum(-U_pred, 0.0)
        frac_negative = float(np.mean(U_pred < 0.0))
        max_negativity_violation = float(np.max(negative_part))
        mean_negativity_violation = float(np.mean(negative_part))

        # Full square symmetry diagnostics on a square grid.
        # Axis conventions:
        # U_pred[j, i] corresponds to y_j, x_i.
        swap_xy_error = U_pred - U_pred.T
        reflect_x_error = U_pred - U_pred[:, ::-1]
        reflect_y_error = U_pred - U_pred[::-1, :]
        rotate_180_error = U_pred - U_pred[::-1, ::-1]

        all_sym_errors = np.stack(
            [
                swap_xy_error,
                reflect_x_error,
                reflect_y_error,
                rotate_180_error,
            ],
            axis=0,
        )

        symmetry_l2 = float(np.sqrt(np.mean(all_sym_errors**2)))
        symmetry_max_abs = float(np.max(np.abs(all_sym_errors)))

        return {
            "boundary_l2_abs": boundary_l2_abs,
            "boundary_max_abs": boundary_max_abs,
            "h1_semi_rel": float(h1_semi_error / h1_semi_ref),
            "min_pred": float(np.min(U_pred)),
            "max_pred": float(np.max(U_pred)),
            "frac_negative": frac_negative,
            "max_negativity_violation": max_negativity_violation,
            "mean_negativity_violation": mean_negativity_violation,
            "symmetry_l2": symmetry_l2,
            "symmetry_max_abs": symmetry_max_abs,
            "swap_xy_symmetry_max_abs": float(np.max(np.abs(swap_xy_error))),
            "reflect_x_symmetry_max_abs": float(np.max(np.abs(reflect_x_error))),
            "reflect_y_symmetry_max_abs": float(np.max(np.abs(reflect_y_error))),
            "rotate_180_symmetry_max_abs": float(np.max(np.abs(rotate_180_error))),
        }

    def plot_prediction(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        grid_meta: Dict[str, np.ndarray],
        outdir: str,
        tag: str,
    ) -> None:
        xs = grid_meta["x"]
        ys = grid_meta["y"]
        X = grid_meta["X"]
        Y = grid_meta["Y"]

        U_pred = pred.reshape(len(ys), len(xs))
        U_true = truth.reshape(len(ys), len(xs))
        U_err = U_pred - U_true
        U_abs_err = np.abs(U_err)
        U_neg = np.maximum(-U_pred, 0.0)

        swap_xy_error = np.abs(U_pred - U_pred.T)
        reflect_x_error = np.abs(U_pred - U_pred[:, ::-1])
        reflect_y_error = np.abs(U_pred - U_pred[::-1, :])
        rotate_180_error = np.abs(U_pred - U_pred[::-1, ::-1])
        U_sym = np.maximum.reduce(
            [swap_xy_error, reflect_x_error, reflect_y_error, rotate_180_error]
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_true, levels=40, cmap="jet")
        ax.set_title("Exact Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_exact.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_pred, levels=40, cmap="jet")
        ax.set_title("Predicted Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_prediction.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_err, levels=40, cmap="jet")
        ax.set_title("Informed Elliptic Error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_error.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_abs_err, levels=40, cmap="jet")
        ax.set_title("Absolute Error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_abs_error.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_neg, levels=40, cmap="jet")
        ax.set_title("Negative Part of Prediction")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_negative_part.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_sym, levels=40, cmap="jet")
        ax.set_title("Max Full-Symmetry Error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_full_symmetry_error.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Hard zero-boundary + hard positivity + full square symmetry
#
# The square and benchmark data are invariant under the 8 maps:
#
#   (x,y)
#   (y,x)
#   (1-x,y)
#   (x,1-y)
#   (1-x,1-y)
#   (1-y,1-x)
#   (y,1-x)
#   (1-y,x)
#
# Define:
#
#   S_theta(x,y) = average of N_theta over all 8 transformed inputs
#
# and then:
#
#   u_theta(x,y)
#     = x(1-x)y(1-y) softplus(S_theta(x,y)).
#
# This guarantees:
#   u_theta = 0 on boundary
#   u_theta >= 0 in Omega
#   u_theta respects all square symmetries
# ============================================================

class EllipticModel(nn.Module):
    def __init__(self, backbone: nn.Module, problem: EllipticProblem):
        super().__init__()
        self.backbone = backbone
        self.problem = problem

    def square_symmetry_transforms(self, z: torch.Tensor) -> List[torch.Tensor]:
        x = z[:, 0:1]
        y = z[:, 1:2]

        return [
            torch.cat([x, y], dim=1),                    # identity
            torch.cat([y, x], dim=1),                    # swap
            torch.cat([1.0 - x, y], dim=1),              # reflect x
            torch.cat([x, 1.0 - y], dim=1),              # reflect y
            torch.cat([1.0 - x, 1.0 - y], dim=1),        # rotate 180
            torch.cat([1.0 - y, 1.0 - x], dim=1),        # swap + rotate 180
            torch.cat([y, 1.0 - x], dim=1),              # rotate 90
            torch.cat([1.0 - y, x], dim=1),              # rotate 270
        ]

    def forward(self, z_physical: torch.Tensor) -> torch.Tensor:
        x = z_physical[:, 0:1]
        y = z_physical[:, 1:2]

        transformed_inputs = self.square_symmetry_transforms(z_physical)

        raw_sum = 0.0
        for z_sym in transformed_inputs:
            raw_sum = raw_sum + self.backbone(self.problem.normalize_inputs(z_sym))

        symmetric_raw = raw_sum / len(transformed_inputs)

        envelope = x * (1.0 - x) * y * (1.0 - y)

        return envelope * F.softplus(symmetric_raw)


# ============================================================
# Training config
# ============================================================

@dataclass
class TrainConfig:
    seed: int = 0
    width: int = 64
    depth: int = 4
    activation: str = "tanh"
    epochs: int = 4000
    lr: float = 1e-3

    n_energy: int = 4096
    n_data: int = 0

    lambda_energy: float = 1.0
    lambda_data: float = 1.0

    eval_every: int = 100
    heldout_residual_n: int = 4096

    outdir: str = "elliptic_runs"


# ============================================================
# Trainer
# ============================================================

class EllipticTrainer:
    def __init__(self, problem: EllipticProblem, config: TrainConfig):
        self.problem = problem
        self.config = config

        self.run_dir = os.path.join(
            config.outdir,
            f"{problem.name}_ndata{config.n_data}_seed{config.seed}",
        )
        ensure_dir(self.run_dir)

        set_seed(config.seed)

        backbone = MLPBackbone(
            in_dim=problem.input_dim,
            out_dim=1,
            width=config.width,
            depth=config.depth,
            activation=config.activation,
        ).to(DEVICE)

        self.model = EllipticModel(backbone, problem).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "total_loss": [],
            "loss_energy": [],
            "loss_data": [],
            "heldout_residual_mse": [],
            "rel_l2_test": [],
            "boundary_l2_abs": [],
            "boundary_max_abs": [],
            "h1_semi_rel": [],
            "min_pred": [],
            "max_pred": [],
            "frac_negative": [],
            "max_negativity_violation": [],
            "mean_negativity_violation": [],
            "symmetry_l2": [],
            "symmetry_max_abs": [],
            "swap_xy_symmetry_max_abs": [],
            "reflect_x_symmetry_max_abs": [],
            "reflect_y_symmetry_max_abs": [],
            "rotate_180_symmetry_max_abs": [],
        }

    def model_u(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

    def energy_density(self, z: torch.Tensor) -> torch.Tensor:
        z = z.clone().detach().requires_grad_(True)

        u = self.model_u(z)
        grad_u = gradient(u, z)
        grad_sq = torch.sum(grad_u**2, dim=1, keepdim=True)

        f = self.problem.forcing_torch(z)

        return 0.5 * grad_sq + 0.5 * u**2 - f * u

    def energy_loss(self, n: int) -> torch.Tensor:
        z_np = self.problem.sample_interior(n)
        z = to_tensor(z_np, requires_grad=True)
        edens = self.energy_density(z)

        return torch.mean(edens)

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        """Strong residual for diagnostics only, not training."""
        z = z.clone().detach().requires_grad_(True)

        u = self.model_u(z)
        uxx = second_derivative(u, z, 0)
        uyy = second_derivative(u, z, 1)
        f = self.problem.forcing_torch(z)

        return -uxx - uyy + u - f

    def heldout_residual_mse(self) -> float:
        z_np = self.problem.heldout_residual_points(self.config.heldout_residual_n)
        z = to_tensor(z_np, requires_grad=True)
        res = self.residual(z)
        return float(torch.mean(res**2).detach().cpu().item())

    def _loss_terms(self) -> Dict[str, torch.Tensor]:
        cfg = self.config

        terms: Dict[str, torch.Tensor] = {}
        terms["loss_energy"] = self.energy_loss(cfg.n_energy)

        if cfg.n_data > 0:
            z_data_np = self.problem.sample_observations(cfg.n_data)
            z_data = to_tensor(z_data_np)
            u_true = to_tensor(self.problem.exact_solution(z_data_np))
            u_pred = self.model_u(z_data)
            terms["loss_data"] = torch.mean((u_pred - u_true) ** 2)
        else:
            terms["loss_data"] = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        terms["total_loss"] = (
            cfg.lambda_energy * terms["loss_energy"]
            + cfg.lambda_data * terms["loss_data"]
        )

        return terms

    @torch.no_grad()
    def predict(self, z_np: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        self.model.eval()
        preds = []

        for i in range(0, len(z_np), batch_size):
            batch = to_tensor(z_np[i:i + batch_size])
            pred = self.model_u(batch).detach().cpu().numpy()
            preds.append(pred)

        return np.vstack(preds)

    def evaluate_common_metrics(
        self,
    ) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        z_grid, meta = self.problem.evaluation_grid()
        pred = self.predict(z_grid)
        truth = self.problem.exact_solution(z_grid)

        rel_l2 = relative_l2_error(pred, truth)
        extras = self.problem.extra_metrics(pred, truth, meta)

        return rel_l2, extras, pred, truth, meta

    def train(self) -> Dict[str, float]:
        cfg = self.config
        t0 = time.time()

        pbar = trange(
            1,
            cfg.epochs + 1,
            desc="Training elliptic Ritz hard-BC + positivity + full symmetry model",
            dynamic_ncols=True,
        )

        final_metrics: Dict[str, float] = {}

        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            terms = self._loss_terms()
            terms["total_loss"].backward()
            self.optimizer.step()

            if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                rel_l2, extras, _, _, _ = self.evaluate_common_metrics()
                heldout_res_mse = self.heldout_residual_mse()

                self.history["epoch"].append(epoch)
                self.history["total_loss"].append(float(terms["total_loss"].detach().cpu().item()))
                self.history["loss_energy"].append(float(terms["loss_energy"].detach().cpu().item()))
                self.history["loss_data"].append(float(terms["loss_data"].detach().cpu().item()))
                self.history["heldout_residual_mse"].append(heldout_res_mse)
                self.history["rel_l2_test"].append(rel_l2)

                for k in [
                    "boundary_l2_abs",
                    "boundary_max_abs",
                    "h1_semi_rel",
                    "min_pred",
                    "max_pred",
                    "frac_negative",
                    "max_negativity_violation",
                    "mean_negativity_violation",
                    "symmetry_l2",
                    "symmetry_max_abs",
                    "swap_xy_symmetry_max_abs",
                    "reflect_x_symmetry_max_abs",
                    "reflect_y_symmetry_max_abs",
                    "rotate_180_symmetry_max_abs",
                ]:
                    self.history[k].append(extras[k])

                pbar.set_postfix_str(
                    f"loss={self.history['total_loss'][-1]:.3e} | "
                    f"energy={self.history['loss_energy'][-1]:.3e} | "
                    f"testL2={rel_l2:.3e} | "
                    f"H1={extras['h1_semi_rel']:.3e} | "
                    f"min={extras['min_pred']:.3e} | "
                    f"sym={extras['symmetry_max_abs']:.3e} | "
                    f"heldoutRes={heldout_res_mse:.3e}"
                )

                final_metrics = {
                    "rel_l2_test": rel_l2,
                    "heldout_residual_mse": heldout_res_mse,
                    **extras,
                }

        elapsed = time.time() - t0
        final_metrics["train_seconds"] = elapsed
        final_metrics["epochs"] = cfg.epochs

        self._save_outputs(final_metrics)
        return final_metrics

    def _save_outputs(self, final_metrics: Dict[str, float]) -> None:
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)

        with open(os.path.join(self.run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        with open(os.path.join(self.run_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)

        z_grid, meta = self.problem.evaluation_grid()
        pred = self.predict(z_grid)
        truth = self.problem.exact_solution(z_grid)

        self.problem.plot_prediction(pred, truth, meta, self.run_dir, tag=self.problem.name)
        self._plot_history()

        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "model_state_dict.pt"))

    def _plot_history(self) -> None:
        epochs = np.array(self.history["epoch"])
        if len(epochs) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ["total_loss", "loss_energy", "loss_data"]:
            vals = np.array(self.history[key])
            if np.any(vals != 0):
                ax.plot(epochs, vals, label=key)

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training losses: elliptic Ritz hard-BC + positivity + full symmetry")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "loss_history.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["heldout_residual_mse"], label="Validation Residual MSE")
        ax.plot(epochs, self.history["rel_l2_test"], label="Relative L2 Error")
        ax.plot(epochs, self.history["total_loss"], label="Total Loss")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title(f"Convergence metrics: informed {self.problem.name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "convergence_metrics.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for key in [
            "min_pred",
            "max_pred",
            "boundary_max_abs",
            "max_negativity_violation",
            "frac_negative",
            "symmetry_max_abs",
        ]:
            vals = np.array(self.history[key])
            ax.plot(epochs, vals, label=key)

        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Diagnostic")
        ax.set_title("Boundary, positivity, and full-symmetry diagnostics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "constraint_diagnostics.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Convenience runner
# ============================================================

def default_config(
    seed: int = 0,
    n_data: int = 0,
    outdir: str = "elliptic_runs",
) -> TrainConfig:
    return TrainConfig(
        seed=seed,
        width=64,
        depth=4,
        activation="tanh",
        epochs=4000,
        lr=1e-3,
        n_energy=4096,
        n_data=n_data,
        lambda_energy=1.0,
        lambda_data=1.0,
        eval_every=100,
        heldout_residual_n=4096,
        outdir=outdir,
    )


def run_single(
    seed: int = 0,
    n_data: int = 0,
    outdir: str = "elliptic_runs",
) -> Dict[str, float]:
    problem = EllipticProblem()
    cfg = default_config(seed=seed, n_data=n_data, outdir=outdir)
    trainer = EllipticTrainer(problem, cfg)

    final_metrics = trainer.train()

    print("\nFinal metrics:")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")

    print(f"Saved outputs to: {trainer.run_dir}")

    return final_metrics


if __name__ == "__main__":
    SEED = 0
    N_DATA = 0
    OUTDIR = "elliptic_runs"

    run_single(seed=SEED, n_data=N_DATA, outdir=OUTDIR)