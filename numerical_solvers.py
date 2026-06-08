import os
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def relative_l2_error(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.linalg.norm(pred.reshape(-1) - truth.reshape(-1))
    den = np.linalg.norm(truth.reshape(-1)) + 1e-12
    return float(num / den)


def relative_l1_error(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.mean(np.abs(pred.reshape(-1) - truth.reshape(-1)))
    den = np.mean(np.abs(truth.reshape(-1))) + 1e-12
    return float(num / den)


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def solve_linear_system(A, b: np.ndarray) -> np.ndarray:
    if SCIPY_AVAILABLE and sp.issparse(A):
        return spla.spsolve(A, b)
    if SCIPY_AVAILABLE:
        return spla.spsolve(sp.csr_matrix(A), b)
    return np.linalg.solve(np.asarray(A), b)


# ============================================================
# Config
# ============================================================

@dataclass
class NumericalConfig:
    seed: int = 0

    # Elliptic grid
    elliptic_nx: int = 121
    elliptic_ny: int = 121

    # Parabolic grid
    parabolic_nx: int = 161
    parabolic_nt: int = 121

    # Hyperbolic grid
    hyperbolic_nx: int = 401
    hyperbolic_nt: int = 121
    hyperbolic_cfl: float = 0.45

    outdir: str = "numerical_solver_runs"


# ============================================================
# Elliptic numerical solver
#
# New elliptic benchmark:
#
#   -Delta u + u = f,    Omega = (0,1)^2
#   u = 0               on boundary
#
# Exact solution:
#
#   u(x,y) = sin(pi x) sin(pi y)
#
# Forcing:
#
#   f(x,y) = (2 pi^2 + 1) sin(pi x) sin(pi y)
#
# Classical method:
#
#   5-point finite-difference stencil + sparse linear solve.
# ============================================================

class EllipticNumericalSolver:
    name = "elliptic"
    classification = "Elliptic"

    def __init__(self, nx: int = 121, ny: int = 121):
        self.nx = nx
        self.ny = ny

    @staticmethod
    def exact_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    @staticmethod
    def forcing_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (2.0 * np.pi**2 + 1.0) * np.sin(np.pi * x) * np.sin(np.pi * y)

    def solve(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        xs = np.linspace(0.0, 1.0, self.nx)
        ys = np.linspace(0.0, 1.0, self.ny)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]

        X, Y = np.meshgrid(xs, ys, indexing="xy")
        U_true = self.exact_xy(X, Y)

        nx_i = self.nx - 2
        ny_i = self.ny - 2
        n_unknowns = nx_i * ny_i

        def idx(i: int, j: int) -> int:
            # interior i=1..nx-2, j=1..ny-2
            return (j - 1) * nx_i + (i - 1)

        rows, cols, data = [], [], []
        b = np.zeros(n_unknowns, dtype=np.float64)

        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                k = idx(i, j)
                x = xs[i]
                y = ys[j]

                diag = 2.0 / dx**2 + 2.0 / dy**2 + 1.0
                rows.append(k)
                cols.append(k)
                data.append(diag)

                # left/right neighbors
                for ii, coeff in [(i - 1, -1.0 / dx**2), (i + 1, -1.0 / dx**2)]:
                    if 1 <= ii <= self.nx - 2:
                        rows.append(k)
                        cols.append(idx(ii, j))
                        data.append(coeff)
                    # boundary is zero, so no RHS correction

                # down/up neighbors
                for jj, coeff in [(j - 1, -1.0 / dy**2), (j + 1, -1.0 / dy**2)]:
                    if 1 <= jj <= self.ny - 2:
                        rows.append(k)
                        cols.append(idx(i, jj))
                        data.append(coeff)
                    # boundary is zero, so no RHS correction

                b[k] = self.forcing_xy(x, y)

        if SCIPY_AVAILABLE:
            A = sp.csr_matrix((data, (rows, cols)), shape=(n_unknowns, n_unknowns))
        else:
            A = np.zeros((n_unknowns, n_unknowns), dtype=np.float64)
            for r, c, v in zip(rows, cols, data):
                A[r, c] += v

        u_vec = solve_linear_system(A, b)

        U_pred = np.zeros_like(U_true)
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                U_pred[j, i] = u_vec[idx(i, j)]

        meta = {"x": xs, "y": ys, "X": X, "Y": Y, "dx": dx, "dy": dy}
        return U_pred, U_true, meta

    def residual_mse(self, U: np.ndarray, meta: Dict[str, np.ndarray]) -> float:
        xs, ys = meta["x"], meta["y"]
        dx, dy = meta["dx"], meta["dy"]
        X, Y = meta["X"], meta["Y"]

        Uxx = (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2
        Uyy = (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dy**2
        f = self.forcing_xy(X[1:-1, 1:-1], Y[1:-1, 1:-1])

        res = -Uxx - Uyy + U[1:-1, 1:-1] - f
        return float(np.mean(res**2))

    def metrics(self, U_pred: np.ndarray, U_true: np.ndarray, meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        xs, ys = meta["x"], meta["y"]
        dx, dy = meta["dx"], meta["dy"]

        boundary_mask = np.zeros_like(U_pred, dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True

        boundary_vals = U_pred[boundary_mask]

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
            "rel_l2_test": relative_l2_error(U_pred, U_true),
            "heldout_residual_mse": self.residual_mse(U_pred, meta),
            "boundary_l2_abs": float(np.sqrt(np.mean(boundary_vals**2))),
            "boundary_max_abs": float(np.max(np.abs(boundary_vals))),
            "h1_semi_rel": float(h1_semi_error / h1_semi_ref),
            "min_pred": float(np.min(U_pred)),
            "max_pred": float(np.max(U_pred)),
            "frac_negative": float(np.mean(U_pred < 0.0)),
            "neg_part_l2": float(np.sqrt(np.mean(negative_part**2))),
            "max_negativity_violation": float(np.max(negative_part)),
        }

    def plot(self, U_pred: np.ndarray, U_true: np.ndarray, meta: Dict[str, np.ndarray], outdir: str) -> None:
        X, Y = meta["X"], meta["Y"]
        U_err = U_pred - U_true

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_true, levels=40, cmap="jet")
        ax.set_title("Exact Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "elliptic_exact.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_pred, levels=40, cmap="jet")
        ax.set_title("Numerical Elliptic Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "elliptic_prediction.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Y, U_err, levels=40, cmap="jet")
        ax.set_title("Numerical Elliptic Error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "elliptic_error.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Parabolic numerical solver
#
# Benchmark:
#
#   u_t - kappa u_xx + beta u_x = f
#   u(0,t)=u(1,t)=0
#   u(x,0)=sin(pi x)
#
# Exact:
#
#   u(x,t)=exp(-t) sin(pi x) + t x(1-x)
#
# Classical method:
#
#   implicit Euler in time
#   centered diffusion
#   upwind advection for beta > 0
# ============================================================

class ParabolicNumericalSolver:
    name = "parabolic"
    classification = "Parabolic"

    def __init__(self, nx: int = 161, nt: int = 121, kappa: float = 1.0, beta: float = 2.0, T: float = 1.0):
        self.nx = nx
        self.nt = nt
        self.kappa = kappa
        self.beta = beta
        self.T = T

    @staticmethod
    def exact_xt(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.exp(-t) * np.sin(np.pi * x) + t * x * (1.0 - x)

    def forcing_xt(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        ut = -np.exp(-t) * np.sin(np.pi * x) + x * (1.0 - x)
        ux = np.exp(-t) * np.pi * np.cos(np.pi * x) + t * (1.0 - 2.0 * x)
        uxx = -np.exp(-t) * (np.pi**2) * np.sin(np.pi * x) - 2.0 * t
        return ut - self.kappa * uxx + self.beta * ux

    def solve(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        xs = np.linspace(0.0, 1.0, self.nx)
        ts = np.linspace(0.0, self.T, self.nt)
        dx = xs[1] - xs[0]
        dt = ts[1] - ts[0]

        X, Tm = np.meshgrid(xs, ts, indexing="xy")
        U_true = self.exact_xt(X, Tm)

        U = np.zeros_like(U_true)
        U[0, :] = np.sin(np.pi * xs)

        n_i = self.nx - 2
        r = self.kappa * dt / dx**2
        adv = self.beta * dt / dx

        # LHS:
        # u^{n+1} - dt*kappa Dxx u^{n+1} + dt*beta Dx_upwind u^{n+1}
        lower = (-r - adv) * np.ones(n_i - 1)
        diag = (1.0 + 2.0 * r + adv) * np.ones(n_i)
        upper = (-r) * np.ones(n_i - 1)

        if SCIPY_AVAILABLE:
            A = sp.diags([lower, diag, upper], offsets=[-1, 0, 1], format="csr")
        else:
            A = np.diag(diag)
            A += np.diag(lower, k=-1)
            A += np.diag(upper, k=1)

        for n in range(self.nt - 1):
            t_next = ts[n + 1]
            rhs = U[n, 1:-1] + dt * self.forcing_xt(xs[1:-1], t_next)

            # Boundary values are zero for this benchmark, so no RHS correction needed.
            U[n + 1, 0] = 0.0
            U[n + 1, -1] = 0.0
            U[n + 1, 1:-1] = solve_linear_system(A, rhs)

        meta = {"x": xs, "t": ts, "X": X, "T": Tm, "dx": dx, "dt": dt}
        return U, U_true, meta

    def residual_mse(self, U: np.ndarray, meta: Dict[str, np.ndarray]) -> float:
        xs, ts = meta["x"], meta["t"]
        dx, dt = meta["dx"], meta["dt"]

        # Use interior space-time points away from t=0 and t=T.
        Ut = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2.0 * dt)
        Ux = (U[1:-1, 2:] - U[1:-1, :-2]) / (2.0 * dx)
        Uxx = (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dx**2

        X_int, T_int = np.meshgrid(xs[1:-1], ts[1:-1], indexing="xy")
        f = self.forcing_xt(X_int, T_int)

        res = Ut - self.kappa * Uxx + self.beta * Ux - f
        return float(np.mean(res**2))

    def metrics(self, U_pred: np.ndarray, U_true: np.ndarray, meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        xs, ts = meta["x"], meta["t"]

        ic_max_abs = float(np.max(np.abs(U_pred[0, :] - U_true[0, :])))
        bc_max_abs = float(
            max(
                np.max(np.abs(U_pred[:, 0] - U_true[:, 0])),
                np.max(np.abs(U_pred[:, -1] - U_true[:, -1])),
            )
        )

        final_time_rel_l2 = relative_l2_error(U_pred[-1, :], U_true[-1, :])
        initial_rel_l2 = relative_l2_error(U_pred[0, :], U_true[0, :])

        max_time_slice_rel_l2 = max(
            relative_l2_error(U_pred[i, :], U_true[i, :])
            for i in range(len(ts))
        )

        return {
            "rel_l2_test": relative_l2_error(U_pred, U_true),
            "heldout_residual_mse": self.residual_mse(U_pred, meta),
            "ic_max_abs": ic_max_abs,
            "bc_max_abs": bc_max_abs,
            "initial_rel_l2": float(initial_rel_l2),
            "final_time_rel_l2": float(final_time_rel_l2),
            "max_time_slice_rel_l2": float(max_time_slice_rel_l2),
            "min_pred": float(np.min(U_pred)),
            "max_pred": float(np.max(U_pred)),
        }

    def plot(self, U_pred: np.ndarray, U_true: np.ndarray, meta: Dict[str, np.ndarray], outdir: str) -> None:
        X, Tm = meta["X"], meta["T"]
        xs, ts = meta["x"], meta["t"]
        U_err = U_pred - U_true

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_true, levels=40, cmap="jet")
        ax.set_title("Exact Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "parabolic_exact.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_pred, levels=40, cmap="jet")
        ax.set_title("Numerical Parabolic Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "parabolic_prediction.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_err, levels=40, cmap="jet")
        ax.set_title("Numerical Parabolic Error")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "parabolic_error.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        idxs = [0, len(ts) // 2, len(ts) - 1]
        for idx in idxs:
            ax.plot(xs, U_true[idx, :], label=f"Exact t={ts[idx]:.2f}")
            ax.plot(xs, U_pred[idx, :], "--", label=f"Numerical t={ts[idx]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title("Numerical Parabolic Time Slices")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "parabolic_time_slices.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Hyperbolic numerical solver
#
# Burgers:
#
#   u_t + (u^2/2)_x = 0
#
# Riemann data:
#
#   u_L = 2, u_R = 0
#
# Entropy shock speed:
#
#   s = (f(uL)-f(uR))/(uL-uR) = 1
#
# Classical method:
#
#   finite-volume Rusanov / local Lax-Friedrichs flux.
# ============================================================

class HyperbolicNumericalSolver:
    name = "hyperbolic"
    classification = "Hyperbolic"

    def __init__(
        self,
        nx: int = 401,
        nt: int = 121,
        x_left: float = -1.0,
        x_right: float = 1.0,
        T: float = 0.75,
        uL: float = 2.0,
        uR: float = 0.0,
        cfl: float = 0.45,
    ):
        self.nx = nx
        self.nt = nt
        self.x_left = x_left
        self.x_right = x_right
        self.T = T
        self.uL = uL
        self.uR = uR
        self.cfl = cfl
        self.s = 0.5 * (uL + uR)

    @staticmethod
    def flux(u: np.ndarray) -> np.ndarray:
        return 0.5 * u**2

    def exact_xt(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.where(x < self.s * t, self.uL, self.uR)

    def rusanov_flux(self, u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
        f_left = self.flux(u_left)
        f_right = self.flux(u_right)
        alpha = np.maximum(np.abs(u_left), np.abs(u_right))
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)

    def solve(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        dx = (self.x_right - self.x_left) / self.nx
        xs = self.x_left + (np.arange(self.nx) + 0.5) * dx
        ts = np.linspace(0.0, self.T, self.nt)

        X, Tm = np.meshgrid(xs, ts, indexing="xy")
        U_true = self.exact_xt(X, Tm)

        U_out = np.zeros_like(U_true)
        u = np.where(xs < 0.0, self.uL, self.uR).astype(np.float64)
        U_out[0, :] = u

        t = 0.0
        out_idx = 1

        while out_idx < self.nt:
            target_t = ts[out_idx]
            max_speed = max(np.max(np.abs(u)), 1e-12)
            dt = self.cfl * dx / max_speed
            dt = min(dt, target_t - t)

            # Ghost cells with fixed far-field states.
            u_ext = np.empty(self.nx + 2, dtype=np.float64)
            u_ext[0] = self.uL
            u_ext[1:-1] = u
            u_ext[-1] = self.uR

            F_half = self.rusanov_flux(u_ext[:-1], u_ext[1:])
            u = u - (dt / dx) * (F_half[1:] - F_half[:-1])

            t += dt

            if abs(t - target_t) < 1e-14:
                U_out[out_idx, :] = u
                out_idx += 1

        meta = {"x": xs, "t": ts, "X": X, "T": Tm, "dx": dx}
        return U_out, U_true, meta

    def entropy_violation_grid(self, U: np.ndarray, meta: Dict[str, np.ndarray]) -> Tuple[float, float]:
        xs, ts = meta["x"], meta["t"]
        dx = xs[1] - xs[0]
        dt = ts[1] - ts[0]

        eta = 0.5 * U**2
        q = (U**3) / 3.0

        eta_t = np.gradient(eta, dt, axis=0, edge_order=1)
        q_x = np.gradient(q, dx, axis=1, edge_order=1)

        expr = eta_t + q_x
        pos = np.maximum(expr, 0.0)

        return float(np.mean(pos)), float(np.max(pos))

    def metrics(self, U_pred: np.ndarray, U_true: np.ndarray, meta: Dict[str, np.ndarray]) -> Dict[str, float]:
        xs, ts = meta["x"], meta["t"]

        rel_l1 = relative_l1_error(U_pred, U_true)
        rel_l2 = relative_l2_error(U_pred, U_true)
        final_rel_l1 = relative_l1_error(U_pred[-1, :], U_true[-1, :])
        final_rel_l2 = relative_l2_error(U_pred[-1, :], U_true[-1, :])

        final_profile = U_pred[-1, :]
        threshold = 0.5 * (self.uL + self.uR)

        idx = int(np.argmin(np.abs(final_profile - threshold)))
        shock_loc_pred = xs[idx]
        shock_loc_true = self.s * ts[-1]
        shock_location_error = float(abs(shock_loc_pred - shock_loc_true))

        mask = (
            (final_profile > self.uR + 0.1 * (self.uL - self.uR))
            & (final_profile < self.uL - 0.1 * (self.uL - self.uR))
        )
        if np.any(mask):
            shock_width = float(xs[mask][-1] - xs[mask][0])
        else:
            shock_width = 0.0

        overshoot = float(max(np.max(U_pred) - self.uL, 0.0))
        undershoot = float(max(self.uR - np.min(U_pred), 0.0))
        mean_ent, max_ent = self.entropy_violation_grid(U_pred, meta)

        return {
            "rel_l2_test": rel_l2,
            "rel_l1_test": rel_l1,
            "final_time_rel_l1": final_rel_l1,
            "final_time_rel_l2": final_rel_l2,
            "shock_location_error_final": shock_location_error,
            "shock_width_final": shock_width,
            "overshoot": overshoot,
            "undershoot": undershoot,
            "mean_entropy_violation_eval": mean_ent,
            "max_entropy_violation_eval": max_ent,
            "min_pred": float(np.min(U_pred)),
            "max_pred": float(np.max(U_pred)),
        }

    def plot(self, U_pred: np.ndarray, U_true: np.ndarray, meta: Dict[str, np.ndarray], outdir: str) -> None:
        X, Tm = meta["X"], meta["T"]
        xs, ts = meta["x"], meta["t"]
        U_err = U_pred - U_true

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_true, levels=40, cmap="jet")
        ax.set_title("Exact Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "hyperbolic_exact.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_pred, levels=40, cmap="jet")
        ax.set_title("Numerical Hyperbolic Solution")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "hyperbolic_prediction.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.contourf(X, Tm, U_err, levels=40, cmap="jet")
        ax.set_title("Numerical Hyperbolic Error")
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "hyperbolic_error.png"), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        idxs = [0, len(ts) // 2, len(ts) - 1]
        for idx in idxs:
            ax.plot(xs, U_true[idx, :], label=f"Exact t={ts[idx]:.2f}")
            ax.plot(xs, U_pred[idx, :], "--", label=f"Numerical t={ts[idx]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title("Numerical Hyperbolic Shock Profiles")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "hyperbolic_shock_profiles.png"), dpi=160)
        plt.close(fig)


# ============================================================
# Runner
# ============================================================

def make_solver(problem_name: str, cfg: NumericalConfig):
    name = problem_name.lower()

    if name == "elliptic":
        return EllipticNumericalSolver(
            nx=cfg.elliptic_nx,
            ny=cfg.elliptic_ny,
        )

    if name == "parabolic":
        return ParabolicNumericalSolver(
            nx=cfg.parabolic_nx,
            nt=cfg.parabolic_nt,
            kappa=1.0,
            beta=2.0,
            T=1.0,
        )

    if name in {"hyperbolic", "burgers", "hyperbolic_burgers_shock"}:
        return HyperbolicNumericalSolver(
            nx=cfg.hyperbolic_nx,
            nt=cfg.hyperbolic_nt,
            x_left=-1.0,
            x_right=1.0,
            T=0.75,
            uL=2.0,
            uR=0.0,
            cfl=cfg.hyperbolic_cfl,
        )

    raise ValueError(f"Unknown problem name: {problem_name}")


def run_single(
    problem_name: str = "elliptic",
    seed: int = 0,
    outdir: str = "numerical_solver_runs",
    config: Optional[NumericalConfig] = None,
) -> Dict[str, float]:
    if config is None:
        config = NumericalConfig(seed=seed, outdir=outdir)

    set_seed(seed)

    solver = make_solver(problem_name, config)

    run_dir = os.path.join(
        config.outdir,
        f"numerical_{solver.name}_seed{config.seed}",
    )
    ensure_dir(run_dir)

    save_json(os.path.join(run_dir, "config.json"), asdict(config))

    t0 = time.time()
    pred, truth, meta = solver.solve()
    elapsed = time.time() - t0

    metrics = solver.metrics(pred, truth, meta)
    metrics["solve_seconds"] = float(elapsed)

    solver.plot(pred, truth, meta, run_dir)

    save_json(os.path.join(run_dir, "final_metrics.json"), metrics)

    print("\nFinal numerical metrics")
    print("-" * 50)
    print(f"Problem: {solver.name}")
    for k, v in metrics.items():
        print(f"{k:>32s}: {v:.6e}" if isinstance(v, (int, float)) else f"{k}: {v}")
    print(f"Saved outputs to: {run_dir}\n")

    return metrics


def run_all_numerical(
    seed: int = 0,
    outdir: str = "numerical_solver_runs",
    config: Optional[NumericalConfig] = None,
) -> Dict[str, Dict[str, float]]:
    if config is None:
        config = NumericalConfig(seed=seed, outdir=outdir)

    results: Dict[str, Dict[str, float]] = {}

    for name in ["elliptic", "parabolic", "hyperbolic"]:
        print(f"\n{'=' * 80}\nRunning numerical solver for {name}\n{'=' * 80}")
        results[name] = run_single(
            problem_name=name,
            seed=seed,
            outdir=outdir,
            config=config,
        )

    summary_path = os.path.join(config.outdir, "numerical_summary_metrics.json")
    ensure_dir(config.outdir)
    save_json(summary_path, results)

    return results


if __name__ == "__main__":
    # Change this to "elliptic", "parabolic", "hyperbolic", or "all".
    PROBLEM = "all"
    SEED = 0
    OUTDIR = "numerical_solver_runs"

    cfg = NumericalConfig(
        seed=SEED,
        elliptic_nx=121,
        elliptic_ny=121,
        parabolic_nx=161,
        parabolic_nt=121,
        hyperbolic_nx=401,
        hyperbolic_nt=121,
        hyperbolic_cfl=0.45,
        outdir=OUTDIR,
    )

    if PROBLEM == "all":
        run_all_numerical(seed=SEED, outdir=OUTDIR, config=cfg)
    else:
        run_single(problem_name=PROBLEM, seed=SEED, outdir=OUTDIR, config=cfg)