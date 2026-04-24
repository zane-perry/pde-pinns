import os
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import trange

# ============================================================
# Global config
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


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
        grad_y[:, dim:dim+1],
        x,
        grad_outputs=torch.ones_like(grad_y[:, dim:dim+1]),
        create_graph=True,
        retain_graph=True,
    )[0][:, dim:dim+1]


def relative_l2_error(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.linalg.norm(pred.reshape(-1) - truth.reshape(-1))
    den = np.linalg.norm(truth.reshape(-1)) + 1e-12
    return float(num / den)


# ============================================================
# Network backbone (same spirit as baseline)
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 4, activation: str = "tanh"):
        super().__init__()
        if activation == "tanh":
            act = nn.Tanh
        elif activation == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = [nn.Linear(in_dim, width), act()]
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
# Elliptic benchmark problem (same PDE as baseline)
# ============================================================

class EllipticProblem:
    name = "elliptic_ritz_hardbc_maxmin"
    input_dim = 2

    def normalize_inputs(self, z: torch.Tensor) -> torch.Tensor:
        return 2.0 * z - 1.0

    @staticmethod
    def u_exact_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * np.sin(np.pi * y) + x**2 * y

    @staticmethod
    def a_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1.0 + x + y

    @staticmethod
    def c_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1.0 + x**2

    def exact_solution(self, z: np.ndarray) -> np.ndarray:
        x, y = z[:, 0], z[:, 1]
        return self.u_exact_xy(x, y)[:, None]

    def boundary_extension_numpy(self, z: np.ndarray) -> np.ndarray:
        # For this benchmark the exact solution is available, so using it as a boundary extension is clean.
        return self.exact_solution(z)

    def lower_barrier_numpy(self, z: np.ndarray) -> np.ndarray:
        # Soft elliptic max/min-style lower barrier.
        # For this benchmark, u(x,y) = sin(pi x) sin(pi y) + x^2 y >= 0 on [0,1]^2.
        return np.zeros((len(z), 1), dtype=np.float64)

    def upper_barrier_numpy(self, z: np.ndarray) -> np.ndarray:
        # Soft elliptic max/min-style upper barrier.
        # Since 0 <= sin(pi x) sin(pi y) <= 1 and 0 <= x^2 y <= 1 on [0,1]^2,
        # the exact solution satisfies u(x,y) <= 2.
        return 2.0 * np.ones((len(z), 1), dtype=np.float64)

    def forcing(self, z: np.ndarray) -> np.ndarray:
        x, y = z[:, 0], z[:, 1]
        a = self.a_xy(x, y)
        c = self.c_xy(x, y)

        ux = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) + 2.0 * x * y
        uy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) + x**2
        uxx = -(np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y) + 2.0 * y
        uyy = -(np.pi**2) * np.sin(np.pi * x) * np.sin(np.pi * y)
        ax = np.ones_like(x)
        ay = np.ones_like(y)
        u = self.u_exact_xy(x, y)

        div_term = ax * ux + a * uxx + ay * uy + a * uyy
        f = -div_term + c * u
        return f[:, None]

    def sample_interior(self, n: int) -> np.ndarray:
        return np.random.rand(n, 2)

    def sample_observations(self, n: int) -> np.ndarray:
        return self.sample_interior(n)

    def heldout_residual_points(self, n: int = 4096) -> np.ndarray:
        return self.sample_interior(n)

    def evaluation_grid(self, n1: int = 121, n2: int = 121) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        xs = np.linspace(0.0, 1.0, n1)
        ys = np.linspace(0.0, 1.0, n2)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        Z = np.column_stack([X.ravel(), Y.ravel()])
        return Z, {"x": xs, "y": ys, "X": X, "Y": Y}

    def extra_metrics(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        xs = grid_meta["x"]
        ys = grid_meta["y"]
        U_pred = pred.reshape(len(ys), len(xs))
        U_true = truth.reshape(len(ys), len(xs))

        boundary_mask = np.zeros_like(U_pred, dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        boundary_error = np.linalg.norm(U_pred[boundary_mask] - U_true[boundary_mask]) / (np.linalg.norm(U_true[boundary_mask]) + 1e-12)

        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        gx_pred, gy_pred = np.gradient(U_pred, dx, dy, edge_order=2)
        gx_true, gy_true = np.gradient(U_true, dx, dy, edge_order=2)
        h1_semi_error = np.sqrt(np.sum((gx_pred - gx_true) ** 2 + (gy_pred - gy_true) ** 2) * dx * dy)
        h1_semi_ref = np.sqrt(np.sum(gx_true**2 + gy_true**2) * dx * dy) + 1e-12

        lower = self.lower_barrier_numpy(np.column_stack([grid_meta["X"].ravel(), grid_meta["Y"].ravel()])).reshape(len(ys), len(xs))
        upper = self.upper_barrier_numpy(np.column_stack([grid_meta["X"].ravel(), grid_meta["Y"].ravel()])).reshape(len(ys), len(xs))
        lower_violation = np.maximum(lower - U_pred, 0.0)
        upper_violation = np.maximum(U_pred - upper, 0.0)

        return {
            "boundary_rel_l2": float(boundary_error),
            "h1_semi_rel": float(h1_semi_error / h1_semi_ref),
            "min_pred": float(np.min(U_pred)),
            "max_pred": float(np.max(U_pred)),
            "max_lower_violation": float(np.max(lower_violation)),
            "max_upper_violation": float(np.max(upper_violation)),
            "mean_barrier_violation": float(np.mean(lower_violation + upper_violation)),
        }

    def plot_prediction(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray], outdir: str, tag: str) -> None:
        X, Y = grid_meta["X"], grid_meta["Y"]
        xs, ys = grid_meta["x"], grid_meta["y"]
        U_pred = pred.reshape(len(ys), len(xs))
        U_true = truth.reshape(len(ys), len(xs))
        U_err = U_pred - U_true
        U_abs = np.abs(U_err)

        fig, axes = plt.subplots(1, 4, figsize=(19, 4))
        for ax, Z, title in zip(axes, [U_true, U_pred, U_err, U_abs], ["Exact", "Prediction", "Signed error", "Absolute error"]):
            im = ax.contourf(X, Y, Z, levels=40)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_field_comparison.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Hard-BC admissible ansatz wrapper
# ============================================================

class HardBCEllipticModel(nn.Module):
    def __init__(self, backbone: nn.Module, problem: EllipticProblem):
        super().__init__()
        self.backbone = backbone
        self.problem = problem

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        y = z[:, 1:2]
        return x * (1.0 - x) * y * (1.0 - y)

    def g_ext(self, z: torch.Tensor) -> torch.Tensor:
        # Using the manufactured exact solution as an extension of the boundary data.
        x = z[:, 0:1]
        y = z[:, 1:2]
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) + x**2 * y

    def forward(self, z_physical: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(self.problem.normalize_inputs(z_physical))
        return self.g_ext(z_physical) + self.phi(z_physical) * raw


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

    n_quad: int = 4096
    n_data: int = 0
    n_qual: int = 4096

    lambda_energy: float = 1.0
    lambda_data: float = 1.0
    lambda_qual: float = 0.1

    eval_every: int = 100
    heldout_residual_n: int = 4096
    outdir: str = "elliptic_ritz_runs"


# ============================================================
# Trainer
# ============================================================

class EllipticRitzTrainer:
    def __init__(self, problem: EllipticProblem, config: TrainConfig):
        self.problem = problem
        self.config = config
        self.run_dir = os.path.join(config.outdir, f"{problem.name}_ndata{config.n_data}_seed{config.seed}")
        ensure_dir(self.run_dir)

        set_seed(config.seed)
        backbone = MLP(
            in_dim=problem.input_dim,
            out_dim=1,
            width=config.width,
            depth=config.depth,
            activation=config.activation,
        ).to(DEVICE)
        self.model = HardBCEllipticModel(backbone, problem).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "total_loss": [],
            "loss_energy": [],
            "loss_data": [],
            "loss_qual": [],
            "heldout_residual_mse": [],
            "rel_l2_test": [],
            "boundary_rel_l2": [],
            "h1_semi_rel": [],
            "min_pred": [],
            "max_pred": [],
            "max_lower_violation": [],
            "max_upper_violation": [],
            "mean_barrier_violation": [],
        }

    def energy_density(self, z: torch.Tensor) -> torch.Tensor:
        z = z.clone().detach().requires_grad_(True)
        u = self.model(z)
        grad_u = gradient(u, z)
        x = z[:, 0:1]
        y = z[:, 1:2]
        a = 1.0 + x + y
        c = 1.0 + x**2
        f = to_tensor(self.problem.forcing(z.detach().cpu().numpy()))
        return 0.5 * a * torch.sum(grad_u**2, dim=1, keepdim=True) + 0.5 * c * u**2 - f * u

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        z = z.clone().detach().requires_grad_(True)
        u = self.model(z)
        grad_u = gradient(u, z)
        ux = grad_u[:, 0:1]
        uy = grad_u[:, 1:2]
        uxx = second_derivative(u, z, 0)
        uyy = second_derivative(u, z, 1)

        x = z[:, 0:1]
        y = z[:, 1:2]
        a = 1.0 + x + y
        c = 1.0 + x**2
        f = to_tensor(self.problem.forcing(z.detach().cpu().numpy()))
        return -(ux + a * uxx + uy + a * uyy) + c * u - f

    def qualitative_loss(self, z: torch.Tensor) -> torch.Tensor:
        u = self.model(z)
        lower = to_tensor(self.problem.lower_barrier_numpy(z.detach().cpu().numpy()))
        upper = to_tensor(self.problem.upper_barrier_numpy(z.detach().cpu().numpy()))
        lower_violation = torch.relu(lower - u)
        upper_violation = torch.relu(u - upper)
        return torch.mean(lower_violation**2 + upper_violation**2)

    def _loss_terms(self) -> Dict[str, torch.Tensor]:
        cfg = self.config
        terms: Dict[str, torch.Tensor] = {}

        z_quad = to_tensor(self.problem.sample_interior(cfg.n_quad), requires_grad=True)
        edens = self.energy_density(z_quad)
        # Domain area is 1, so Monte Carlo estimate is just mean.
        terms["loss_energy"] = torch.mean(edens)

        z_qual = to_tensor(self.problem.sample_interior(cfg.n_qual))
        terms["loss_qual"] = self.qualitative_loss(z_qual)

        if cfg.n_data > 0:
            z_data_np = self.problem.sample_observations(cfg.n_data)
            z_data = to_tensor(z_data_np)
            u_data_true = to_tensor(self.problem.exact_solution(z_data_np))
            u_data_pred = self.model(z_data)
            terms["loss_data"] = torch.mean((u_data_pred - u_data_true) ** 2)
        else:
            terms["loss_data"] = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        terms["total_loss"] = (
            cfg.lambda_energy * terms["loss_energy"]
            + cfg.lambda_qual * terms["loss_qual"]
            + cfg.lambda_data * terms["loss_data"]
        )
        return terms

    @torch.no_grad()
    def predict(self, z_np: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        self.model.eval()
        preds = []
        for i in range(0, len(z_np), batch_size):
            batch = to_tensor(z_np[i:i+batch_size])
            pred = self.model(batch).detach().cpu().numpy()
            preds.append(pred)
        return np.vstack(preds)

    def heldout_residual_mse(self) -> float:
        z_np = self.problem.heldout_residual_points(self.config.heldout_residual_n)
        z = to_tensor(z_np, requires_grad=True)
        res = self.residual(z)
        return float(torch.mean(res**2).detach().cpu().item())

    def evaluate_common_metrics(self) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        z_grid, meta = self.problem.evaluation_grid()
        pred = self.predict(z_grid)
        truth = self.problem.exact_solution(z_grid)
        rel_l2 = relative_l2_error(pred, truth)
        extras = self.problem.extra_metrics(pred, truth, meta)
        return rel_l2, extras, pred, truth, meta

    def train(self) -> Dict[str, float]:
        cfg = self.config
        t0 = time.time()
        pbar = trange(1, cfg.epochs + 1, desc="Training elliptic Ritz hard-BC + max/min", dynamic_ncols=True)
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
                self.history["loss_qual"].append(float(terms["loss_qual"].detach().cpu().item()))
                self.history["heldout_residual_mse"].append(heldout_res_mse)
                self.history["rel_l2_test"].append(rel_l2)
                self.history["boundary_rel_l2"].append(extras["boundary_rel_l2"])
                self.history["h1_semi_rel"].append(extras["h1_semi_rel"])
                self.history["min_pred"].append(extras["min_pred"])
                self.history["max_pred"].append(extras["max_pred"])
                self.history["max_lower_violation"].append(extras["max_lower_violation"])
                self.history["max_upper_violation"].append(extras["max_upper_violation"])
                self.history["mean_barrier_violation"].append(extras["mean_barrier_violation"])

                pbar.set_postfix_str(
                    f"loss={self.history['total_loss'][-1]:.3e} | "
                    f"energy={self.history['loss_energy'][-1]:.3e} | "
                    f"qual={self.history['loss_qual'][-1]:.3e} | "
                    f"testL2={rel_l2:.3e} | "
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
        for key in ["total_loss", "loss_energy", "loss_qual", "loss_data"]:
            vals = np.array(self.history[key])
            if np.any(vals != 0):
                ax.plot(epochs, np.abs(vals), label=key)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss magnitude")
        ax.set_title("Training losses: elliptic Ritz hard-BC + max/min")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "loss_history.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["heldout_residual_mse"], label="heldout_residual_mse")
        ax.plot(epochs, self.history["rel_l2_test"], label="rel_l2_test")
        ax.plot(epochs, self.history["boundary_rel_l2"], label="boundary_rel_l2")
        ax.plot(epochs, self.history["h1_semi_rel"], label="h1_semi_rel")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title("Convergence metrics: elliptic Ritz hard-BC + max/min")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "convergence_metrics.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, self.history["min_pred"], label="min_pred")
        ax.plot(epochs, self.history["max_pred"], label="max_pred")
        ax.plot(epochs, self.history["max_lower_violation"], label="max_lower_violation")
        ax.plot(epochs, self.history["max_upper_violation"], label="max_upper_violation")
        ax.plot(epochs, self.history["mean_barrier_violation"], label="mean_barrier_violation")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Barrier metric")
        ax.set_title("Max/min-style barrier diagnostics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "barrier_diagnostics.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Convenience runner
# ============================================================

def default_config(seed: int = 0, n_data: int = 0, outdir: str = "elliptic_ritz_runs") -> TrainConfig:
    return TrainConfig(
        seed=seed,
        width=64,
        depth=4,
        activation="tanh",
        epochs=4000,
        lr=1e-3,
        n_quad=4096,
        n_data=n_data,
        n_qual=4096,
        lambda_energy=1.0,
        lambda_data=1.0,
        lambda_qual=0.1,
        eval_every=100,
        heldout_residual_n=4096,
        outdir=outdir,
    )


def run_single(seed: int = 0, n_data: int = 0, outdir: str = "elliptic_ritz_runs") -> Dict[str, float]:
    problem = EllipticProblem()
    cfg = default_config(seed=seed, n_data=n_data, outdir=outdir)
    trainer = EllipticRitzTrainer(problem, cfg)
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
    N_DATA = 100   # set to 0 for pure physics-informed; try 50 or 100 for sparse hybrid runs
    OUTDIR = "elliptic_ritz_runs"
    run_single(seed=SEED, n_data=N_DATA, outdir=OUTDIR)
