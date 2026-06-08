import os
import math
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import trange

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def to_tensor(x: np.ndarray | List[List[float]] | List[float], requires_grad: bool = False) -> Tensor:
    return torch.tensor(x, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)


def relative_l2_error(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.linalg.norm(pred - truth)
    den = np.linalg.norm(truth) + 1e-12
    return float(num / den)


def relative_l1_error(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.linalg.norm(pred - truth, ord=1)
    den = np.linalg.norm(truth, ord=1) + 1e-12
    return float(num / den)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def gradient(y: Tensor, x: Tensor) -> Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def second_derivative(y: Tensor, x: Tensor, component: int) -> Tensor:
    grad_y = gradient(y, x)
    dyi = grad_y[:, component:component + 1]
    grad2 = torch.autograd.grad(
        dyi,
        x,
        grad_outputs=torch.ones_like(dyi),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad2[:, component:component + 1]


# ============================================================
# Network
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, width: int = 64, depth: int = 4, activation: str = "tanh"):
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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ============================================================
# Problem base class
# ============================================================

class PDEProblem:
    name: str = "base"
    input_dim: int = 1
    has_boundary: bool = True
    has_initial: bool = False

    def normalize_inputs(self, z: Tensor) -> Tensor:
        raise NotImplementedError

    def exact_solution(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def residual(self, model: nn.Module, z: Tensor) -> Tensor:
        raise NotImplementedError

    def boundary_values(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def initial_values(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_interior(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def sample_boundary(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def sample_initial(self, n: int) -> np.ndarray:
        raise NotImplementedError

    def sample_observations(self, n: int) -> np.ndarray:
        return self.sample_interior(n)

    def evaluation_grid(self, n1: int = 101, n2: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        raise NotImplementedError

    def heldout_residual_points(self, n: int = 4096) -> np.ndarray:
        return self.sample_interior(n)

    def extra_metrics(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        return {}

    def plot_prediction(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray], outdir: str, tag: str) -> None:
        raise NotImplementedError


# ============================================================
# Elliptic benchmark: reaction-diffusion Dirichlet problem
#
# Baseline version:
#   - Strong residual loss
#   - Soft boundary-condition loss
#
# PDE:
#   -Delta u + u = f,  Omega = (0,1)^2
#
# Homogeneous Dirichlet boundary:
#   u = 0 on boundary
#
# Exact solution used for evaluation:
#   u(x,y) = sin(pi x) sin(pi y)
#
# Forcing:
#   f(x,y) = (2*pi^2 + 1) sin(pi x) sin(pi y)
#
# This benchmark is compatible with the minimum-principle story:
#   f >= 0 and u = 0 on boundary imply u >= 0 in the domain.
# ============================================================

class EllipticProblem(PDEProblem):
    name = "elliptic"
    input_dim = 2
    has_boundary = True
    has_initial = False

    def normalize_inputs(self, z: Tensor) -> Tensor:
        # Domain already [0,1]^2 -> scale to [-1,1]^2.
        return 2.0 * z - 1.0

    @staticmethod
    def u_exact_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_solution(self, z: np.ndarray) -> np.ndarray:
        x, y = z[:, 0], z[:, 1]
        return self.u_exact_xy(x, y)[:, None]

    def forcing(self, z: np.ndarray) -> np.ndarray:
        x, y = z[:, 0], z[:, 1]
        f = (2.0 * np.pi**2 + 1.0) * np.sin(np.pi * x) * np.sin(np.pi * y)
        return f[:, None]

    def residual(self, model: nn.Module, z: Tensor) -> Tensor:
        z = z.clone().detach().requires_grad_(True)

        u = model(self.normalize_inputs(z))
        uxx = second_derivative(u, z, 0)
        uyy = second_derivative(u, z, 1)

        f = to_tensor(self.forcing(z.detach().cpu().numpy()))
        return -uxx - uyy + u - f

    def boundary_values(self, z: np.ndarray) -> np.ndarray:
        # Homogeneous Dirichlet boundary values.
        return np.zeros((len(z), 1), dtype=np.float32)

    def initial_values(self, z: np.ndarray) -> np.ndarray:
        raise RuntimeError("Elliptic problem has no initial condition.")

    def sample_interior(self, n: int) -> np.ndarray:
        return np.random.rand(n, 2)

    def sample_boundary(self, n: int) -> np.ndarray:
        n_side = n // 4

        y1 = np.random.rand(n_side, 1)
        y2 = np.random.rand(n_side, 1)
        x3 = np.random.rand(n_side, 1)
        x4 = np.random.rand(n - 3 * n_side, 1)

        b1 = np.hstack([np.zeros_like(y1), y1])
        b2 = np.hstack([np.ones_like(y2), y2])
        b3 = np.hstack([x3, np.zeros_like(x3)])
        b4 = np.hstack([x4, np.ones_like(x4)])

        return np.vstack([b1, b2, b3, b4])

    def sample_initial(self, n: int) -> np.ndarray:
        raise RuntimeError("Elliptic problem has no initial condition.")

    def evaluation_grid(self, n1: int = 121, n2: Optional[int] = 121) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
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

        # Boundary is zero, so absolute boundary diagnostics are more meaningful
        # than relative boundary error.
        boundary_vals = U_pred[boundary_mask]
        boundary_l2_abs = np.sqrt(np.mean(boundary_vals**2))
        boundary_max_abs = np.max(np.abs(boundary_vals))

        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        # np.gradient axis order is y first, x second for array shape [ny, nx].
        uy_pred, ux_pred = np.gradient(U_pred, dy, dx, edge_order=2)
        uy_true, ux_true = np.gradient(U_true, dy, dx, edge_order=2)

        h1_semi_error = np.sqrt(
            np.sum((ux_pred - ux_true) ** 2 + (uy_pred - uy_true) ** 2) * dx * dy
        )
        h1_semi_ref = np.sqrt(
            np.sum(ux_true**2 + uy_true**2) * dx * dy
        ) + 1e-12

        negative_part = np.maximum(-U_pred, 0.0)

        return {
            "boundary_l2_abs": float(boundary_l2_abs),
            "boundary_max_abs": float(boundary_max_abs),
            "h1_semi_rel": float(h1_semi_error / h1_semi_ref),
            "min_pred": float(np.min(U_pred)),
            "max_pred": float(np.max(U_pred)),
            "frac_negative": float(np.mean(U_pred < 0.0)),
            "neg_part_l2": float(np.sqrt(np.mean(negative_part**2))),
            "max_negativity_violation": float(np.max(negative_part)),
        }

    def plot_prediction(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray], outdir: str, tag: str) -> None:
        xs = grid_meta["x"]
        ys = grid_meta["y"]
        X, Y = grid_meta["X"], grid_meta["Y"]

        U_pred = pred.reshape(len(ys), len(xs))
        U_true = truth.reshape(len(ys), len(xs))
        U_err = U_pred - U_true
        U_neg = np.maximum(-U_pred, 0.0)

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
        ax.set_title(f"Uninformed {self.name} Error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_error.png"), dpi=160)
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


# ============================================================
# Parabolic benchmark: 1D advection-diffusion
# ============================================================

class ParabolicProblem(PDEProblem):
    name = "parabolic"
    input_dim = 2
    has_boundary = True
    has_initial = True

    def __init__(self, kappa: float = 1.0, beta: float = 2.0, T: float = 1.0):
        self.kappa = kappa
        self.beta = beta
        self.T = T

    def normalize_inputs(self, z: Tensor) -> Tensor:
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
        uxx = -np.exp(-t) * (np.pi**2) * np.sin(np.pi * x) - 2.0 * t
        f = ut - self.kappa * uxx + self.beta * ux
        return f[:, None]

    def residual(self, model: nn.Module, z: Tensor) -> Tensor:
        z = z.clone().detach().requires_grad_(True)
        u = model(self.normalize_inputs(z))
        grad_u = gradient(u, z)
        ux = grad_u[:, 0:1]
        ut = grad_u[:, 1:2]
        uxx = second_derivative(u, z, 0)
        f = to_tensor(self.forcing(z.detach().cpu().numpy()))
        return ut - self.kappa * uxx + self.beta * ux - f

    def boundary_values(self, z: np.ndarray) -> np.ndarray:
        return self.exact_solution(z)

    def initial_values(self, z: np.ndarray) -> np.ndarray:
        return self.exact_solution(z)

    def sample_interior(self, n: int) -> np.ndarray:
        x = np.random.rand(n, 1)
        t = self.T * np.random.rand(n, 1)
        return np.hstack([x, t])

    def sample_boundary(self, n: int) -> np.ndarray:
        n_half = n // 2
        t1 = self.T * np.random.rand(n_half, 1)
        t2 = self.T * np.random.rand(n - n_half, 1)
        b1 = np.hstack([np.zeros_like(t1), t1])
        b2 = np.hstack([np.ones_like(t2), t2])
        return np.vstack([b1, b2])

    def sample_initial(self, n: int) -> np.ndarray:
        x = np.random.rand(n, 1)
        t = np.zeros((n, 1))
        return np.hstack([x, t])

    def evaluation_grid(self, n1: int = 161, n2: Optional[int] = 121) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        xs = np.linspace(0.0, 1.0, n1)
        ts = np.linspace(0.0, self.T, n2)
        X, Tm = np.meshgrid(xs, ts, indexing="xy")
        Z = np.column_stack([X.ravel(), Tm.ravel()])
        return Z, {"x": xs, "t": ts, "X": X, "T": Tm}

    def extra_metrics(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        xs = grid_meta["x"]
        ts = grid_meta["t"]
        U_pred = pred.reshape(len(ts), len(xs))
        U_true = truth.reshape(len(ts), len(xs))
        final_time_rel_l2 = np.linalg.norm(U_pred[-1, :] - U_true[-1, :]) / (np.linalg.norm(U_true[-1, :]) + 1e-12)
        initial_rel_l2 = np.linalg.norm(U_pred[0, :] - U_true[0, :]) / (np.linalg.norm(U_true[0, :]) + 1e-12)
        max_time_rel_l2 = max(
            np.linalg.norm(U_pred[i, :] - U_true[i, :]) / (np.linalg.norm(U_true[i, :]) + 1e-12)
            for i in range(len(ts))
        )
        return {
            "initial_rel_l2": float(initial_rel_l2),
            "final_time_rel_l2": float(final_time_rel_l2),
            "max_time_slice_rel_l2": float(max_time_rel_l2),
        }

    def plot_prediction(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray], outdir: str, tag: str) -> None:
        xs = grid_meta["x"]
        ts = grid_meta["t"]
        X, Tm = grid_meta["X"], grid_meta["T"]
        U_pred = pred.reshape(len(ts), len(xs))
        U_true = truth.reshape(len(ts), len(xs))
        U_err = U_pred - U_true

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_true, levels=40, cmap="jet")
        ax.set_title("Exact Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_exact.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_err, levels=40, cmap="jet")
        ax.set_title(f"Uninformed {self.name} Error")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_error.png"), dpi=160)
        plt.close(fig)

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


# ============================================================
# Hyperbolic benchmark: Burgers entropy shock solution
# ============================================================

class HyperbolicProblem(PDEProblem):
    name = "hyperbolic"
    input_dim = 2
    has_boundary = True
    has_initial = True

    def __init__(self, x_left: float = -1.0, x_right: float = 1.0, T: float = 0.75, uL: float = 2.0, uR: float = 0.0):
        self.x_left = x_left
        self.x_right = x_right
        self.T = T
        self.uL = uL
        self.uR = uR
        self.s = 0.5 * (uL + uR)

    def normalize_inputs(self, z: Tensor) -> Tensor:
        x = z[:, 0:1]
        t = z[:, 1:2]
        x_scaled = 2.0 * (x - self.x_left) / (self.x_right - self.x_left) - 1.0
        t_scaled = 2.0 * (t / self.T) - 1.0
        return torch.cat([x_scaled, t_scaled], dim=1)

    def exact_solution(self, z: np.ndarray) -> np.ndarray:
        x, t = z[:, 0], z[:, 1]
        out = np.where(x < self.s * t, self.uL, self.uR)
        return out[:, None].astype(np.float32)

    def residual(self, model: nn.Module, z: Tensor) -> Tensor:
        z = z.clone().detach().requires_grad_(True)
        u = model(self.normalize_inputs(z))
        grad_u = gradient(u, z)
        ux = grad_u[:, 0:1]
        ut = grad_u[:, 1:2]
        return ut + u * ux

    def boundary_values(self, z: np.ndarray) -> np.ndarray:
        return self.exact_solution(z)

    def initial_values(self, z: np.ndarray) -> np.ndarray:
        return self.exact_solution(z)

    def sample_interior(self, n: int) -> np.ndarray:
        x = self.x_left + (self.x_right - self.x_left) * np.random.rand(n, 1)
        t = self.T * np.random.rand(n, 1)
        return np.hstack([x, t])

    def sample_boundary(self, n: int) -> np.ndarray:
        n_half = n // 2
        t1 = self.T * np.random.rand(n_half, 1)
        t2 = self.T * np.random.rand(n - n_half, 1)
        b1 = np.hstack([self.x_left * np.ones_like(t1), t1])
        b2 = np.hstack([self.x_right * np.ones_like(t2), t2])
        return np.vstack([b1, b2])

    def sample_initial(self, n: int) -> np.ndarray:
        x = self.x_left + (self.x_right - self.x_left) * np.random.rand(n, 1)
        t = np.zeros((n, 1))
        return np.hstack([x, t])

    def evaluation_grid(self, n1: int = 401, n2: Optional[int] = 121) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        xs = np.linspace(self.x_left, self.x_right, n1)
        ts = np.linspace(0.0, self.T, n2)
        X, Tm = np.meshgrid(xs, ts, indexing="xy")
        Z = np.column_stack([X.ravel(), Tm.ravel()])
        return Z, {"x": xs, "t": ts, "X": X, "T": Tm}

    def extra_metrics(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        xs = grid_meta["x"]
        ts = grid_meta["t"]
        U_pred = pred.reshape(len(ts), len(xs))
        U_true = truth.reshape(len(ts), len(xs))
        final_pred = U_pred[-1, :]
        final_true = U_true[-1, :]

        thresh = 0.5 * (self.uL + self.uR)
        pred_idx = int(np.argmin(np.abs(final_pred - thresh)))
        shock_loc_pred = xs[pred_idx]
        shock_loc_true = self.s * self.T

        transition_width = float(
            np.sum(
                (final_pred > self.uR + 0.1 * (self.uL - self.uR))
                & (final_pred < self.uL - 0.1 * (self.uL - self.uR))
            ) * (xs[1] - xs[0])
        )

        osc_measure = float(
            np.mean(
                np.maximum(final_pred - self.uL, 0.0)
                + np.maximum(self.uR - final_pred, 0.0)
            )
        )

        return {
            "rel_l1": relative_l1_error(pred, truth),
            "shock_location_error": float(abs(shock_loc_pred - shock_loc_true)),
            "transition_width": transition_width,
            "overshoot_undershoot_mean": osc_measure,
        }

    def plot_prediction(self, pred: np.ndarray, truth: np.ndarray, grid_meta: Dict[str, np.ndarray], outdir: str, tag: str) -> None:
        xs = grid_meta["x"]
        ts = grid_meta["t"]
        X, Tm = grid_meta["X"], grid_meta["T"]
        U_pred = pred.reshape(len(ts), len(xs))
        U_true = truth.reshape(len(ts), len(xs))
        U_err = U_pred - U_true

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_true, levels=40, cmap="jet")
        ax.set_title("Exact Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_exact.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_err, levels=40, cmap="jet")
        ax.set_title(f"Uninformed {self.name} Error")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{tag}_error.png"), dpi=160)
        plt.close(fig)

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
        fig.savefig(os.path.join(outdir, f"{tag}_shock_profiles.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Config and training
# ============================================================

@dataclass
class TrainConfig:
    seed: int = 0
    width: int = 64
    depth: int = 4
    activation: str = "tanh"
    epochs: int = 5000
    lr: float = 1e-3

    n_res: int = 2048
    n_bc: int = 512
    n_ic: int = 512
    n_data: int = 0

    lambda_res: float = 1.0
    lambda_bc: float = 1.0
    lambda_ic: float = 1.0
    lambda_data: float = 1.0

    eval_every: int = 100
    heldout_residual_n: int = 4096
    outdir: str = "baseline_runs"
    problem_name: str = "elliptic"


class PINNTrainer:
    def __init__(self, problem: PDEProblem, config: TrainConfig):
        self.problem = problem
        self.config = config
        self.run_dir = os.path.join(
            config.outdir,
            f"{problem.name}_ndata{config.n_data}_seed{config.seed}"
        )
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
            "loss_bc": [],
            "loss_ic": [],
            "loss_data": [],
            "heldout_residual_mse": [],
            "rel_l2_test": [],
        }

    def _loss_terms(self) -> Dict[str, Tensor]:
        cfg = self.config
        terms: Dict[str, Tensor] = {}

        z_res = to_tensor(self.problem.sample_interior(cfg.n_res), requires_grad=True)
        res = self.problem.residual(self.model, z_res)
        terms["loss_res"] = torch.mean(res**2)

        if self.problem.has_boundary and cfg.n_bc > 0:
            z_bc_np = self.problem.sample_boundary(cfg.n_bc)
            z_bc = to_tensor(z_bc_np)
            u_bc_true = to_tensor(self.problem.boundary_values(z_bc_np))
            u_bc_pred = self.model(self.problem.normalize_inputs(z_bc))
            terms["loss_bc"] = torch.mean((u_bc_pred - u_bc_true) ** 2)
        else:
            terms["loss_bc"] = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        if self.problem.has_initial and cfg.n_ic > 0:
            z_ic_np = self.problem.sample_initial(cfg.n_ic)
            z_ic = to_tensor(z_ic_np)
            u_ic_true = to_tensor(self.problem.initial_values(z_ic_np))
            u_ic_pred = self.model(self.problem.normalize_inputs(z_ic))
            terms["loss_ic"] = torch.mean((u_ic_pred - u_ic_true) ** 2)
        else:
            terms["loss_ic"] = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        if cfg.n_data > 0:
            z_data_np = self.problem.sample_observations(cfg.n_data)
            z_data = to_tensor(z_data_np)
            u_data_true = to_tensor(self.problem.exact_solution(z_data_np))
            u_data_pred = self.model(self.problem.normalize_inputs(z_data))
            terms["loss_data"] = torch.mean((u_data_pred - u_data_true) ** 2)
        else:
            terms["loss_data"] = torch.tensor(0.0, dtype=DTYPE, device=DEVICE)

        total = (
            cfg.lambda_res * terms["loss_res"]
            + cfg.lambda_bc * terms["loss_bc"]
            + cfg.lambda_ic * terms["loss_ic"]
            + cfg.lambda_data * terms["loss_data"]
        )

        terms["total_loss"] = total
        return terms

    @torch.no_grad()
    def predict(self, z_np: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        self.model.eval()
        preds = []

        for i in range(0, len(z_np), batch_size):
            batch = to_tensor(z_np[i:i + batch_size])
            pred = self.model(self.problem.normalize_inputs(batch)).detach().cpu().numpy()
            preds.append(pred)

        return np.vstack(preds)

    def heldout_residual_mse(self) -> float:
        z_np = self.problem.heldout_residual_points(self.config.heldout_residual_n)
        z = to_tensor(z_np, requires_grad=True)
        res = self.problem.residual(self.model, z)
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

        pbar = trange(
            1,
            cfg.epochs + 1,
            desc=f"Training {self.problem.name}",
            dynamic_ncols=True
        )

        final_metrics: Dict[str, float] = {}

        for epoch in pbar:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            terms = self._loss_terms()
            terms["total_loss"].backward()
            self.optimizer.step()

            if epoch == 1 or epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
                rel_l2, extras, pred, truth, meta = self.evaluate_common_metrics()
                heldout_res_mse = self.heldout_residual_mse()

                self.history["epoch"].append(epoch)
                self.history["total_loss"].append(float(terms["total_loss"].detach().cpu().item()))
                self.history["loss_res"].append(float(terms["loss_res"].detach().cpu().item()))
                self.history["loss_bc"].append(float(terms["loss_bc"].detach().cpu().item()))
                self.history["loss_ic"].append(float(terms["loss_ic"].detach().cpu().item()))
                self.history["loss_data"].append(float(terms["loss_data"].detach().cpu().item()))
                self.history["heldout_residual_mse"].append(heldout_res_mse)
                self.history["rel_l2_test"].append(rel_l2)

                desc = (
                    f"loss={self.history['total_loss'][-1]:.3e} | "
                    f"res={self.history['loss_res'][-1]:.3e} | "
                    f"testL2={rel_l2:.3e} | heldoutRes={heldout_res_mse:.3e}"
                )
                pbar.set_postfix_str(desc)

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
        self._save_model()

    def _save_model(self) -> None:
        torch.save(self.model.state_dict(), os.path.join(self.run_dir, "model_state_dict.pt"))

    def _plot_history(self) -> None:
        epochs = np.array(self.history["epoch"])

        if len(epochs) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ["total_loss", "loss_res", "loss_bc", "loss_ic", "loss_data"]:
            vals = np.array(self.history[key])
            if np.any(vals > 0):
                ax.plot(epochs, vals, label=key)

        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training losses: {self.problem.name}")
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
        ax.set_title(f"Convergence metrics: uninformed {self.problem.name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.run_dir, "convergence_metrics.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Problem factory and convenience runners
# ============================================================

def make_problem(name: str) -> PDEProblem:
    name = name.lower()

    if name == "elliptic":
        return EllipticProblem()

    if name == "parabolic":
        return ParabolicProblem(kappa=1.0, beta=2.0, T=1.0)

    if name == "hyperbolic":
        return HyperbolicProblem(x_left=-1.0, x_right=1.0, T=0.75, uL=2.0, uR=0.0)

    raise ValueError(f"Unknown problem name: {name}")


def default_config(problem_name: str) -> TrainConfig:
    cfg = TrainConfig(problem_name=problem_name)

    if problem_name == "elliptic":
        cfg.n_res = 4096
        cfg.n_bc = 1024
        cfg.n_ic = 0
        cfg.epochs = 4000
        cfg.eval_every = 100

    elif problem_name == "parabolic":
        cfg.n_res = 4096
        cfg.n_bc = 1024
        cfg.n_ic = 1024
        cfg.epochs = 5000
        cfg.eval_every = 100

    elif problem_name == "hyperbolic":
        cfg.n_res = 4096
        cfg.n_bc = 1024
        cfg.n_ic = 2048
        cfg.epochs = 6000
        cfg.eval_every = 100
        cfg.lambda_ic = 2.0

    return cfg


def run_single(problem_name: str = "elliptic", seed: int = 0, n_data: int = 0, outdir: str = "baseline_runs") -> Dict[str, float]:
    problem = make_problem(problem_name)
    cfg = default_config(problem_name)
    cfg.problem_name = problem_name
    cfg.seed = seed
    cfg.n_data = n_data
    cfg.outdir = outdir

    trainer = PINNTrainer(problem, cfg)
    metrics = trainer.train()

    print("\nFinal metrics")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:>28s}: {v:.6e}" if isinstance(v, (int, float)) else f"{k}: {v}")

    print(f"Saved outputs to: {trainer.run_dir}\n")
    return metrics


def run_all_baselines(seed: int = 0, n_data: int = 0, outdir: str = "baseline_runs") -> Dict[str, Dict[str, float]]:
    results = {}

    for name in ["elliptic", "parabolic", "hyperbolic"]:
        print(f"\n{'=' * 80}\nRunning baseline for {name}\n{'=' * 80}")
        results[name] = run_single(problem_name=name, seed=seed, n_data=n_data, outdir=outdir)

    return results


if __name__ == "__main__":
    # --------------------------------------------------------
    # Quick-start options
    # --------------------------------------------------------
    # Change the values below if you want to switch problems or
    # add matched observation data later.
    PROBLEM = "elliptic"          # "elliptic", "parabolic", or "hyperbolic"
    SEED = 0
    N_DATA = 0                    # 0 = pure physics-informed baseline
    OUTDIR = "baseline_runs"

    # Run one problem at a time first. After this is working,
    # you can call run_all_baselines(...) if you want all three.
    #run_single(problem_name=PROBLEM, seed=SEED, n_data=N_DATA, outdir=OUTDIR)

    # Or uncomment this to run all three:
    run_all_baselines(seed=SEED, n_data=N_DATA, outdir=OUTDIR)