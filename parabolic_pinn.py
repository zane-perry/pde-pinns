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
# Parabolic benchmark
# u_t - kappa u_xx + beta u_x = f
# exact: u(x,t) = exp(-t) sin(pi x) + t x (1-x)
# ============================================================

class ParabolicProblem:
    name = "parabolic"
    input_dim = 2
    has_boundary = True
    has_initial = True

    def __init__(self, kappa: float = 1.0, beta: float = 2.0, T: float = 1.0):
        self.kappa = kappa
        self.beta = beta
        self.T = T

    def normalize_inputs(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        t = z[:, 1:2]
        x_scaled = 2.0 * x - 1.0
        t_scaled = 2.0 * (t / self.T) - 1.0
        return torch.cat([x_scaled, t_scaled], dim=1)

    @staticmethod
    def u_exact_xt(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.exp(-t) * np.sin(np.pi * x) + t * x * (1.0 - x)

    def exact_solution(self, z: np.ndarray) -> np.ndarray:
        x, t = z[:, 0], z[:, 1]
        return self.u_exact_xt(x, t)[:, None]

    def forcing(self, z: np.ndarray) -> np.ndarray:
        x, t = z[:, 0], z[:, 1]

        ut = -np.exp(-t) * np.sin(np.pi * x) + x * (1.0 - x)
        ux = np.exp(-t) * np.pi * np.cos(np.pi * x) + t * (1.0 - 2.0 * x)
        uxx = -np.exp(-t) * (np.pi ** 2) * np.sin(np.pi * x) - 2.0 * t

        f = ut - self.kappa * uxx + self.beta * ux
        return f[:, None]

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x)[:, None]

    def lower_barrier_numpy(self, z: np.ndarray) -> np.ndarray:
        # Explicit subsolution:
        # underline u(x,t) = exp(-t) sin(pi x)
        x, t = z[:, 0], z[:, 1]
        return (np.exp(-t) * np.sin(np.pi * x))[:, None]

    def upper_barrier_numpy(self, z: np.ndarray) -> np.ndarray:
        # Explicit supersolution:
        # overline u(x,t) = exp(-t) sin(pi x) + 2 t x(1-x)
        x, t = z[:, 0], z[:, 1]
        return (np.exp(-t) * np.sin(np.pi * x) + 2.0 * t * x * (1.0 - x))[:, None]

    def sample_interior(self, n: int) -> np.ndarray:
        x = np.random.rand(n, 1)
        t = self.T * np.random.rand(n, 1)
        return np.hstack([x, t])

    def sample_observations(self, n: int) -> np.ndarray:
        return self.sample_interior(n)

    def heldout_residual_points(self, n: int = 4096) -> np.ndarray:
        return self.sample_interior(n)

    def evaluation_grid(self, nx: int = 241, nt: int = 201) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        xs = np.linspace(0.0, 1.0, nx)
        ts = np.linspace(0.0, self.T, nt)
        X, Tm = np.meshgrid(xs, ts, indexing="xy")
        Z = np.column_stack([X.ravel(), Tm.ravel()])
        return Z, {"x": xs, "t": ts, "X": X, "T": Tm}

    def extra_metrics(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        xs = grid_meta["x"]
        ts = grid_meta["t"]
        U_pred = pred.reshape(len(ts), len(xs))
        U_true = truth.reshape(len(ts), len(xs))

        # Boundary / initial exactness diagnostics
        ic_true = U_true[0, :]
        ic_pred = U_pred[0, :]
        ic_max_abs = float(np.max(np.abs(ic_pred - ic_true)))

        bc_left_max_abs = float(np.max(np.abs(U_pred[:, 0])))
        bc_right_max_abs = float(np.max(np.abs(U_pred[:, -1])))
        bc_max_abs = max(bc_left_max_abs, bc_right_max_abs)

        # Final-time error
        final_rel_l2 = relative_l2_error(U_pred[-1, :], U_true[-1, :])

        # Barrier diagnostics
        Z = np.column_stack([grid_meta["X"].ravel(), grid_meta["T"].ravel()])
        lower = self.lower_barrier_numpy(Z).reshape(len(ts), len(xs))
        upper = self.upper_barrier_numpy(Z).reshape(len(ts), len(xs))
        lower_violation = np.maximum(lower - U_pred, 0.0)
        upper_violation = np.maximum(U_pred - upper, 0.0)

        negative_part = np.maximum(-U_pred, 0.0)
        frac_negative = float(np.mean(U_pred < 0.0))

        # Time-dependent minimum
        min_over_x_vs_t = np.min(U_pred, axis=1)

        return {
            "ic_max_abs": ic_max_abs,
            "bc_max_abs": bc_max_abs,
            "final_time_rel_l2": float(final_rel_l2),
            "min_pred": float(np.min(U_pred)),
            "max_pred": float(np.max(U_pred)),
            "frac_negative": frac_negative,
            "neg_part_l2": float(np.sqrt(np.mean(negative_part ** 2))),
            "max_lower_violation": float(np.max(lower_violation)),
            "max_upper_violation": float(np.max(upper_violation)),
            "mean_barrier_violation": float(np.mean(lower_violation + upper_violation)),
            "min_over_x_vs_t_min": float(np.min(min_over_x_vs_t)),
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
        ts = grid_meta["t"]
        X, Tm = grid_meta["X"], grid_meta["T"]
        U_pred = pred.reshape(len(ts), len(xs))
        U_true = truth.reshape(len(ts), len(xs))
        U_err = U_pred - U_true

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_err, levels=40, cmap="jet")
        ax.set_title("Informed Parabolic Error")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_error.png"), dpi=160)
        plt.close(fig)

        # Time slices
        fig, ax = plt.subplots(figsize=(7, 4))
        idxs = [0, len(ts) // 2, len(ts) - 1]
        for idx in idxs:
            ax.plot(xs, U_true[idx, :], label=f"Exact t={ts[idx]:.2f}")
            ax.plot(xs, U_pred[idx, :], "--", label=f"Pred t={ts[idx]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title("Selected time slices")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_time_slices.png"), dpi=160)
        plt.close(fig)

        # Barrier diagnostics over time
        min_over_x = np.min(U_pred, axis=1)
        Z = np.column_stack([X.ravel(), Tm.ravel()])
        lower = self.lower_barrier_numpy(Z).reshape(len(ts), len(xs))
        upper = self.upper_barrier_numpy(Z).reshape(len(ts), len(xs))
        lower_min_over_x = np.min(lower, axis=1)
        upper_max_over_x = np.max(upper, axis=1)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts, min_over_x, label=r"$\min_x u_\theta(x,t)$")
        ax.plot(ts, lower_min_over_x, "--", label=r"$\min_x u_-(x,t)$")
        ax.plot(ts, upper_max_over_x, "--", label=r"$\max_x u_+(x,t)$")
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.set_title("Barrier / min diagnostics over time")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_barrier_vs_time.png"), dpi=160)
        plt.close(fig)

        # Negative part heatmap
        fig, ax = plt.subplots(figsize=(7, 4))
        neg = np.maximum(-U_pred, 0.0)
        im = ax.contourf(X, Tm, neg, levels=40, cmap="jet")
        ax.set_title("Negative part of prediction")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_negative_part.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Hard IC/BC + hard comparison-barrier admissible ansatz
#
# lower(x,t) = exp(-t) sin(pi x)
# upper(x,t) = exp(-t) sin(pi x) + 2 t x(1-x)
#
# u_theta(x,t) = lower(x,t) + (upper(x,t) - lower(x,t)) sigmoid(N_theta(x,t))
#
# Since 0 < sigmoid(N) < 1:
# lower <= u_theta <= upper everywhere.
#
# Also upper - lower = 2 t x(1-x), so the learnable correction
# vanishes at t = 0, x = 0, and x = 1. Therefore the IC/BC are
# hard-enforced simultaneously with the barrier bounds.
# ============================================================

class ParabolicModel(nn.Module):
    def __init__(self, backbone: nn.Module, problem: ParabolicProblem):
        super().__init__()
        self.backbone = backbone
        self.problem = problem

    def lower_barrier(self, z_physical: torch.Tensor) -> torch.Tensor:
        x = z_physical[:, 0:1]
        t = z_physical[:, 1:2]
        return torch.exp(-t) * torch.sin(torch.pi * x)

    def upper_barrier(self, z_physical: torch.Tensor) -> torch.Tensor:
        x = z_physical[:, 0:1]
        t = z_physical[:, 1:2]
        return torch.exp(-t) * torch.sin(torch.pi * x) + 2.0 * t * x * (1.0 - x)

    def forward(self, z_physical: torch.Tensor) -> torch.Tensor:
        raw = self.backbone(self.problem.normalize_inputs(z_physical))
        lower = self.lower_barrier(z_physical)
        upper = self.upper_barrier(z_physical)
        return lower + (upper - lower) * torch.sigmoid(raw)


# ============================================================
# Training config
# ============================================================

@dataclass
class TrainConfig:
    seed: int = 0
    width: int = 64
    depth: int = 4
    activation: str = "tanh"
    epochs: int = 5000
    lr: float = 1e-3

    n_res: int = 4096
    n_data: int = 0

    lambda_res: float = 1.0
    lambda_data: float = 1.0

    eval_every: int = 100
    heldout_residual_n: int = 4096
    outdir: str = "parabolic_hardbcic_hardbarrier_runs"


# ============================================================
# Trainer
# ============================================================

class ParabolicTrainer:
    def __init__(self, problem: ParabolicProblem, config: TrainConfig):
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

        self.model = ParabolicModel(backbone, problem).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "total_loss": [],
            "loss_res": [],
            "loss_data": [],
            "heldout_residual_mse": [],
            "rel_l2_test": [],
            "ic_max_abs": [],
            "bc_max_abs": [],
            "final_time_rel_l2": [],
            "min_pred": [],
            "max_pred": [],
            "frac_negative": [],
            "neg_part_l2": [],
            "max_lower_violation": [],
            "max_upper_violation": [],
            "mean_barrier_violation": [],
            "min_over_x_vs_t_min": [],
        }

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        z = z.clone().detach().requires_grad_(True)
        u = self.model(z)
        grad_u = gradient(u, z)
        ux = grad_u[:, 0:1]
        ut = grad_u[:, 1:2]
        uxx = second_derivative(u, z, 0)

        f = to_tensor(self.problem.forcing(z.detach().cpu().numpy()))
        return ut - self.problem.kappa * uxx + self.problem.beta * ux - f

    def _loss_terms(self) -> Dict[str, torch.Tensor]:
        cfg = self.config
        terms: Dict[str, torch.Tensor] = {}

        z_res = to_tensor(self.problem.sample_interior(cfg.n_res), requires_grad=True)
        res = self.residual(z_res)
        terms["loss_res"] = torch.mean(res ** 2)

        if cfg.n_data > 0:
            z_data_np = self.problem.sample_observations(cfg.n_data)
            z_data = to_tensor(z_data_np)
            u_data_true = to_tensor(self.problem.exact_solution(z_data_np))
            u_data_pred = self.model(z_data)
            terms["loss_data"] = torch.mean((u_data_pred - u_data_true) ** 2)
        else:
            terms["loss_data"] = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        terms["total_loss"] = (
            cfg.lambda_res * terms["loss_res"]
            + cfg.lambda_data * terms["loss_data"]
        )
        return terms

    @torch.no_grad()
    def predict(self, z_np: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        self.model.eval()
        preds = []
        for i in range(0, len(z_np), batch_size):
            batch = to_tensor(z_np[i:i + batch_size])
            pred = self.model(batch).detach().cpu().numpy()
            preds.append(pred)
        return np.vstack(preds)

    def heldout_residual_mse(self) -> float:
        z_np = self.problem.heldout_residual_points(self.config.heldout_residual_n)
        z = to_tensor(z_np, requires_grad=True)
        res = self.residual(z)
        return float(torch.mean(res ** 2).detach().cpu().item())

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
        pbar = trange(
            1,
            cfg.epochs + 1,
            desc="Training parabolic hard-IC/BC + hard barrier",
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
                self.history["loss_res"].append(float(terms["loss_res"].detach().cpu().item()))
                self.history["loss_data"].append(float(terms["loss_data"].detach().cpu().item()))
                self.history["heldout_residual_mse"].append(heldout_res_mse)
                self.history["rel_l2_test"].append(rel_l2)

                for k in [
                    "ic_max_abs",
                    "bc_max_abs",
                    "final_time_rel_l2",
                    "min_pred",
                    "max_pred",
                    "frac_negative",
                    "neg_part_l2",
                    "max_lower_violation",
                    "max_upper_violation",
                    "mean_barrier_violation",
                    "min_over_x_vs_t_min",
                ]:
                    self.history[k].append(extras[k])

                pbar.set_postfix_str(
                    f"loss={self.history['total_loss'][-1]:.3e} | "
                    f"res={self.history['loss_res'][-1]:.3e} | "
                    f"testL2={rel_l2:.3e} | "
                    f"heldoutRes={heldout_res_mse:.3e} | "
                    f"barrierViol={extras['mean_barrier_violation']:.3e}"
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
        for key in ["total_loss", "loss_res", "loss_data"]:
            vals = np.array(self.history[key])
            if np.any(vals != 0):
                ax.plot(epochs, vals, label=key)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training losses: parabolic hard-IC/BC + hard barrier")
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
        ax.plot(epochs, self.history["min_pred"], label="min_pred")
        ax.plot(epochs, self.history["max_pred"], label="max_pred")
        ax.plot(epochs, self.history["max_lower_violation"], label="max_lower_violation")
        ax.plot(epochs, self.history["max_upper_violation"], label="max_upper_violation")
        ax.plot(epochs, self.history["frac_negative"], label="frac_negative")
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Barrier / sign diagnostic")
        ax.set_title("Barrier diagnostics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "barrier_diagnostics.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Convenience runner
# ============================================================

def default_config(
    seed: int = 0,
    n_data: int = 0,
    outdir: str = "parabolic_runs",
) -> TrainConfig:
    return TrainConfig(
        seed=seed,
        width=64,
        depth=4,
        activation="tanh",
        epochs=5000,
        lr=1e-3,
        n_res=4096,
        n_data=n_data,
        lambda_res=1.0,
        lambda_data=1.0,
        eval_every=100,
        heldout_residual_n=4096,
        outdir=outdir,
    )


def run_single(
    seed: int = 0,
    n_data: int = 0,
    outdir: str = "parabolic_runs",
) -> Dict[str, float]:
    problem = ParabolicProblem()
    cfg = default_config(seed=seed, n_data=n_data, outdir=outdir)
    trainer = ParabolicTrainer(problem, cfg)
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
    N_DATA = 0   # set to 0 for pure physics-informed; try 50 or 100 for sparse hybrid runs
    OUTDIR = "parabolic_runs"
    run_single(seed=SEED, n_data=N_DATA, outdir=OUTDIR)