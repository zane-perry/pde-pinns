
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


def relative_l2_error(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.linalg.norm(pred.reshape(-1) - truth.reshape(-1))
    den = np.linalg.norm(truth.reshape(-1)) + 1e-12
    return float(num / den)


def relative_l1_error(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.mean(np.abs(pred.reshape(-1) - truth.reshape(-1)))
    den = np.mean(np.abs(truth.reshape(-1))) + 1e-12
    return float(num / den)


# ============================================================
# Network backbone
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
# Hyperbolic benchmark: Burgers entropy shock
# u_t + (u^2/2)_x = 0
# Riemann data: u_L = 2, u_R = 0, shock speed s = 1
# Entropy solution: u(x,t)=2 for x<t, 0 for x>t
# Domain chosen as x in [-1, 2], t in [0, 1]
# ============================================================

class HyperbolicProblem:
    name = "hyperbolic_entropy_pinn"
    input_dim = 2
    has_initial = True

    def __init__(self, x_left: float = -1.0, x_right: float = 2.0, T: float = 1.0):
        self.x_left = x_left
        self.x_right = x_right
        self.T = T
        self.uL = 2.0
        self.uR = 0.0
        self.shock_speed = 1.0

    def normalize_inputs(self, z: torch.Tensor) -> torch.Tensor:
        x = z[:, 0:1]
        t = z[:, 1:2]
        x_scaled = 2.0 * (x - self.x_left) / (self.x_right - self.x_left) - 1.0
        t_scaled = 2.0 * (t / self.T) - 1.0
        return torch.cat([x_scaled, t_scaled], dim=1)

    def exact_solution(self, z: np.ndarray) -> np.ndarray:
        x, t = z[:, 0], z[:, 1]
        u = np.where(x < self.shock_speed * t, self.uL, self.uR)
        return u[:, None].astype(np.float64)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0.0, self.uL, self.uR)[:, None].astype(np.float64)

    def sample_interior(self, n: int) -> np.ndarray:
        x = self.x_left + (self.x_right - self.x_left) * np.random.rand(n, 1)
        t = self.T * np.random.rand(n, 1)
        return np.hstack([x, t])

    def sample_initial(self, n: int) -> np.ndarray:
        x = self.x_left + (self.x_right - self.x_left) * np.random.rand(n, 1)
        t = np.zeros((n, 1), dtype=np.float64)
        return np.hstack([x, t])

    def sample_observations(self, n: int) -> np.ndarray:
        return self.sample_interior(n)

    def heldout_residual_points(self, n: int = 4096) -> np.ndarray:
        # Avoid t=0 exactly for residual diagnostics because of the initial discontinuity.
        x = self.x_left + (self.x_right - self.x_left) * np.random.rand(n, 1)
        t = 1e-3 + (self.T - 1e-3) * np.random.rand(n, 1)
        return np.hstack([x, t])

    def heldout_entropy_points(self, n: int = 4096) -> np.ndarray:
        return self.heldout_residual_points(n)

    def evaluation_grid(self, nx: int = 401, nt: int = 201) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        xs = np.linspace(self.x_left, self.x_right, nx)
        ts = np.linspace(0.0, self.T, nt)
        X, Tm = np.meshgrid(xs, ts, indexing="xy")
        Z = np.column_stack([X.ravel(), Tm.ravel()])
        return Z, {"x": xs, "t": ts, "X": X, "T": Tm}

    def shock_location_true(self, t: np.ndarray) -> np.ndarray:
        return self.shock_speed * t

    def extra_metrics(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        xs = grid_meta["x"]
        ts = grid_meta["t"]
        U_pred = pred.reshape(len(ts), len(xs))
        U_true = truth.reshape(len(ts), len(xs))

        rel_l1 = relative_l1_error(U_pred, U_true)
        final_rel_l1 = relative_l1_error(U_pred[-1, :], U_true[-1, :])
        final_rel_l2 = relative_l2_error(U_pred[-1, :], U_true[-1, :])

        # Initial-condition mismatch on evaluation grid at t=0
        ic_max_abs = float(np.max(np.abs(U_pred[0, :] - U_true[0, :])))

        # Estimate shock location at final time as nearest crossing of level u=1.
        final_profile = U_pred[-1, :]
        threshold = 1.0
        idx = np.argmin(np.abs(final_profile - threshold))
        shock_loc_pred = xs[idx]
        shock_loc_true = self.shock_speed * ts[-1]
        shock_loc_error = float(abs(shock_loc_pred - shock_loc_true))

        # Approximate shock thickness at final time: width of region with 0.1 <= u <= 1.9
        mask = (final_profile >= 0.1) & (final_profile <= 1.9)
        if np.any(mask):
            shock_width = float(xs[mask][-1] - xs[mask][0])
        else:
            shock_width = 0.0

        # Overshoot / undershoot relative to [0, 2]
        overshoot = float(max(np.max(U_pred) - self.uL, 0.0))
        undershoot = float(max(-np.min(U_pred), 0.0))

        # Entropy violation indicator on evaluation grid using quadratic entropy
        # eta(u)=u^2/2, q(u)=u^3/3, should satisfy eta_t + q_x <= 0
        dt = ts[1] - ts[0]
        dx = xs[1] - xs[0]
        eta = 0.5 * U_pred**2
        q = (U_pred**3) / 3.0
        eta_t = np.gradient(eta, dt, axis=0, edge_order=1)
        q_x = np.gradient(q, dx, axis=1, edge_order=1)
        ent_expr = eta_t + q_x
        pos_entropy_violation = np.maximum(ent_expr, 0.0)

        return {
            "rel_l1_test": float(rel_l1),
            "final_time_rel_l1": float(final_rel_l1),
            "final_time_rel_l2": float(final_rel_l2),
            "ic_max_abs": ic_max_abs,
            "shock_location_error_final": shock_loc_error,
            "shock_width_final": shock_width,
            "overshoot": overshoot,
            "undershoot": undershoot,
            "mean_entropy_violation_eval": float(np.mean(pos_entropy_violation)),
            "max_entropy_violation_eval": float(np.max(pos_entropy_violation)),
            "min_pred": float(np.min(U_pred)),
            "max_pred": float(np.max(U_pred)),
        }

    def plot_prediction(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray], outdir: str, tag: str) -> None:
        xs = grid_meta["x"]
        ts = grid_meta["t"]
        X, Tm = grid_meta["X"], grid_meta["T"]
        U_pred = pred.reshape(len(ts), len(xs))
        U_true = truth.reshape(len(ts), len(xs))
        U_err = U_pred - U_true
        U_abs = np.abs(U_err)

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        for ax, Z, title in zip(
            axes,
            [U_true, U_pred, U_err, U_abs],
            ["Exact", "Prediction", "Signed error", "Absolute error"],
        ):
            im = ax.contourf(X, Tm, Z, levels=40)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_field_comparison.png"), dpi=160)
        plt.close(fig)

        # Selected time slices
        fig, ax = plt.subplots(figsize=(7, 4))
        idxs = [0, len(ts) // 2, len(ts) - 1]
        for idx in idxs:
            ax.plot(xs, U_true[idx, :], label=f"Exact t={ts[idx]:.2f}")
            ax.plot(xs, U_pred[idx, :], "--", label=f"Pred t={ts[idx]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title("Shock profiles at selected times")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_time_slices.png"), dpi=160)
        plt.close(fig)

        # Entropy violation heatmap
        dt = ts[1] - ts[0]
        dx = xs[1] - xs[0]
        eta = 0.5 * U_pred**2
        q = (U_pred**3) / 3.0
        eta_t = np.gradient(eta, dt, axis=0, edge_order=1)
        q_x = np.gradient(q, dx, axis=1, edge_order=1)
        ent_expr = eta_t + q_x
        pos_entropy_violation = np.maximum(ent_expr, 0.0)

        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.contourf(X, Tm, pos_entropy_violation, levels=40)
        ax.set_title("Positive entropy violation")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_entropy_violation.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Training config
# ============================================================

@dataclass
class TrainConfig:
    seed: int = 0
    width: int = 64
    depth: int = 4
    activation: str = "tanh"
    epochs: int = 6000
    lr: float = 1e-3

    n_res: int = 4096
    n_ent: int = 4096
    n_ic: int = 1024
    n_data: int = 0

    lambda_res: float = 1.0
    lambda_ent: float = 0.1
    lambda_ic: float = 5.0
    lambda_data: float = 1.0

    eval_every: int = 100
    heldout_residual_n: int = 4096
    heldout_entropy_n: int = 4096
    outdir: str = "hyperbolic_entropy_runs"


# ============================================================
# Trainer
# ============================================================

class HyperbolicEntropyTrainer:
    def __init__(self, problem: HyperbolicProblem, config: TrainConfig):
        self.problem = problem
        self.config = config
        self.run_dir = os.path.join(config.outdir, f"{problem.name}_ndata{config.n_data}_seed{config.seed}")
        ensure_dir(self.run_dir)

        set_seed(config.seed)
        self.model = MLP(
            in_dim=problem.input_dim,
            out_dim=1,
            width=config.width,
            depth=config.depth,
            activation=config.activation,
        ).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "total_loss": [],
            "loss_res": [],
            "loss_ent": [],
            "loss_ic": [],
            "loss_data": [],
            "heldout_residual_mse": [],
            "heldout_entropy_violation": [],
            "rel_l2_test": [],
            "rel_l1_test": [],
            "final_time_rel_l1": [],
            "shock_location_error_final": [],
            "shock_width_final": [],
            "overshoot": [],
            "undershoot": [],
            "mean_entropy_violation_eval": [],
            "max_entropy_violation_eval": [],
            "ic_max_abs": [],
            "min_pred": [],
            "max_pred": [],
        }

    def model_u(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(self.problem.normalize_inputs(z))

    def residual(self, z: torch.Tensor) -> torch.Tensor:
        # Conservative-form residual: u_t + (u^2/2)_x
        z = z.clone().detach().requires_grad_(True)
        u = self.model_u(z)
        flux = 0.5 * u**2
        ut = gradient(u, z)[:, 1:2]
        fx = gradient(flux, z)[:, 0:1]
        return ut + fx

    def entropy_expression(self, z: torch.Tensor) -> torch.Tensor:
        # Quadratic entropy: eta(u)=u^2/2, q(u)=u^3/3
        z = z.clone().detach().requires_grad_(True)
        u = self.model_u(z)
        eta = 0.5 * u**2
        q = (u**3) / 3.0
        eta_t = gradient(eta, z)[:, 1:2]
        q_x = gradient(q, z)[:, 0:1]
        return eta_t + q_x

    def entropy_loss(self, z: torch.Tensor) -> torch.Tensor:
        expr = self.entropy_expression(z)
        return torch.mean(torch.relu(expr) ** 2)

    def _loss_terms(self) -> Dict[str, torch.Tensor]:
        cfg = self.config
        terms: Dict[str, torch.Tensor] = {}

        z_res = to_tensor(self.problem.sample_interior(cfg.n_res), requires_grad=True)
        terms["loss_res"] = torch.mean(self.residual(z_res) ** 2)

        z_ent = to_tensor(self.problem.sample_interior(cfg.n_ent), requires_grad=True)
        terms["loss_ent"] = self.entropy_loss(z_ent)

        z_ic_np = self.problem.sample_initial(cfg.n_ic)
        z_ic = to_tensor(z_ic_np)
        u_ic_true = to_tensor(self.problem.exact_solution(z_ic_np))
        u_ic_pred = self.model_u(z_ic)
        terms["loss_ic"] = torch.mean((u_ic_pred - u_ic_true) ** 2)

        if cfg.n_data > 0:
            z_data_np = self.problem.sample_observations(cfg.n_data)
            z_data = to_tensor(z_data_np)
            u_data_true = to_tensor(self.problem.exact_solution(z_data_np))
            u_data_pred = self.model_u(z_data)
            terms["loss_data"] = torch.mean((u_data_pred - u_data_true) ** 2)
        else:
            terms["loss_data"] = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        terms["total_loss"] = (
            cfg.lambda_res * terms["loss_res"]
            + cfg.lambda_ent * terms["loss_ent"]
            + cfg.lambda_ic * terms["loss_ic"]
            + cfg.lambda_data * terms["loss_data"]
        )
        return terms

    @torch.no_grad()
    def predict(self, z_np: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        self.model.eval()
        preds = []
        for i in range(0, len(z_np), batch_size):
            batch = to_tensor(z_np[i:i+batch_size])
            pred = self.model_u(batch).detach().cpu().numpy()
            preds.append(pred)
        return np.vstack(preds)

    def heldout_residual_mse(self) -> float:
        z_np = self.problem.heldout_residual_points(self.config.heldout_residual_n)
        z = to_tensor(z_np, requires_grad=True)
        res = self.residual(z)
        return float(torch.mean(res**2).detach().cpu().item())

    def heldout_entropy_violation(self) -> float:
        z_np = self.problem.heldout_entropy_points(self.config.heldout_entropy_n)
        z = to_tensor(z_np, requires_grad=True)
        expr = self.entropy_expression(z)
        return float(torch.mean(torch.relu(expr) ** 2).detach().cpu().item())

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
        pbar = trange(1, cfg.epochs + 1, desc="Training hyperbolic entropy PINN", dynamic_ncols=True)
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
                heldout_ent = self.heldout_entropy_violation()

                self.history["epoch"].append(epoch)
                self.history["total_loss"].append(float(terms["total_loss"].detach().cpu().item()))
                self.history["loss_res"].append(float(terms["loss_res"].detach().cpu().item()))
                self.history["loss_ent"].append(float(terms["loss_ent"].detach().cpu().item()))
                self.history["loss_ic"].append(float(terms["loss_ic"].detach().cpu().item()))
                self.history["loss_data"].append(float(terms["loss_data"].detach().cpu().item()))
                self.history["heldout_residual_mse"].append(heldout_res_mse)
                self.history["heldout_entropy_violation"].append(heldout_ent)
                self.history["rel_l2_test"].append(rel_l2)

                for k in [
                    "rel_l1_test",
                    "final_time_rel_l1",
                    "shock_location_error_final",
                    "shock_width_final",
                    "overshoot",
                    "undershoot",
                    "mean_entropy_violation_eval",
                    "max_entropy_violation_eval",
                    "ic_max_abs",
                    "min_pred",
                    "max_pred",
                ]:
                    self.history[k].append(extras[k])

                pbar.set_postfix_str(
                    f"loss={self.history['total_loss'][-1]:.3e} | "
                    f"res={self.history['loss_res'][-1]:.3e} | "
                    f"ent={self.history['loss_ent'][-1]:.3e} | "
                    f"testL1={self.history['rel_l1_test'][-1]:.3e} | "
                    f"heldoutEnt={heldout_ent:.3e}"
                )

                final_metrics = {
                    "rel_l2_test": rel_l2,
                    "heldout_residual_mse": heldout_res_mse,
                    "heldout_entropy_violation": heldout_ent,
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
        for key in ["total_loss", "loss_res", "loss_ent", "loss_ic", "loss_data"]:
            vals = np.array(self.history[key])
            if np.any(vals != 0):
                ax.plot(epochs, vals, label=key)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training losses: hyperbolic entropy PINN")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "loss_history.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ["heldout_residual_mse", "heldout_entropy_violation", "rel_l1_test", "final_time_rel_l1", "shock_location_error_final"]:
            vals = np.array(self.history[key])
            ax.plot(epochs, vals, label=key)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_title("Convergence metrics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "convergence_metrics.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ["overshoot", "undershoot", "mean_entropy_violation_eval", "max_entropy_violation_eval", "shock_width_final"]:
            vals = np.array(self.history[key])
            ax.plot(epochs, vals, label=key)
        ax.set_yscale("symlog", linthresh=1e-10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Shock / entropy diagnostic")
        ax.set_title("Shock and entropy diagnostics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "shock_entropy_diagnostics.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Convenience runner
# ============================================================

def default_config(seed: int = 0, n_data: int = 0, outdir: str = "hyperbolic_entropy_runs") -> TrainConfig:
    return TrainConfig(
        seed=seed,
        width=64,
        depth=4,
        activation="tanh",
        epochs=6000,
        lr=1e-3,
        n_res=4096,
        n_ent=4096,
        n_ic=1024,
        n_data=n_data,
        lambda_res=1.0,
        lambda_ent=0.1,
        lambda_ic=5.0,
        lambda_data=1.0,
        eval_every=100,
        heldout_residual_n=4096,
        heldout_entropy_n=4096,
        outdir=outdir,
    )


def run_single(seed: int = 0, n_data: int = 0, outdir: str = "hyperbolic_entropy_runs") -> Dict[str, float]:
    problem = HyperbolicProblem()
    cfg = default_config(seed=seed, n_data=n_data, outdir=outdir)
    trainer = HyperbolicEntropyTrainer(problem, cfg)
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
    OUTDIR = "hyperbolic_entropy_runs"
    run_single(seed=SEED, n_data=N_DATA, outdir=OUTDIR)
