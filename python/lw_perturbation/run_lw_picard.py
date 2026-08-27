#!/usr/bin/env python3
"""Direct Lienard-Wiechert Picard iteration benchmark.

This is deliberately not a PIC/grid solver.  It keeps the direct analytic
pair-field model, but replaces the TPS Taylor series in a field-strength
parameter with fixed-point iterations of the retarded trajectory map.

Iteration 0 is the external-field-only trajectory.  Iteration m+1 solves a
full Lorentz push on the requested time grid while evaluating all self-fields
from the complete trajectory history of iteration m.  Optional relaxation
applies the standard damped Picard update

    x_{m+1} = (1 - omega) x_m + omega T[x_m].

The saved moments-by-iteration file is intended to be compared to OPALX in the
same spirit as the K-order TPS plots, except the curve label is now the Picard
iteration count.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from run_lw_test import (
    C_SI,
    EPSILON_0_SI,
    M_E_MEV,
    P_MEV_C_SI,
    POSITION_MEAN_M,
    POSITION_SIGMA_M,
    Q_E_SI,
    beta_from_p,
    gamma_from_p,
    kinetic_energy_mev,
    match_mean_and_rms,
    read_sdds_first_data_row,
    vectorized_unperturbed,
)


DEFAULT_INITIAL_ENERGY_GEV = 1.0e-9
DEFAULT_TOTAL_CHARGE_SI = -1.0e-12
DEFAULT_E_Z_SI = -1.0e6
DEFAULT_N_PARTICLES = 512
DEFAULT_T_END_S = 3.0e-9
DEFAULT_N_OUTPUTS = 801
DEFAULT_ITERATIONS = 4
DEFAULT_RELAXATION = 1.0
DEFAULT_SOFTENING_M = 0.0
DEFAULT_R_MIN_M = 1.0e-12
DEFAULT_RETARDED_ITERATIONS = 3
DEFAULT_SEED = 33
DEFAULT_DISTRIBUTION = "uniform-ellipsoid"
MOMENTUM_RMS_MEV_C = np.array([1.0e-6, 1.0e-6, 1.0e-6])


@dataclass(frozen=True)
class PicardConfig:
    seed: int
    n_particles: int
    total_charge_si: float
    e_z_si: float
    initial_energy_gev: float
    t_end_s: float
    n_outputs: int
    iterations: int
    relaxation: float
    integrator: str
    retarded_iterations: int
    softening_m: float
    r_min_m: float
    distribution: str
    output_dir: str
    match_initial_stat: str | None
    save_all_iterations: bool

    @property
    def q_macro_si(self) -> float:
        return self.total_charge_si / self.n_particles

    @property
    def dpz_dt_mev_c_per_s(self) -> float:
        return (Q_E_SI * self.e_z_si) / P_MEV_C_SI


def parse_args() -> PicardConfig:
    script_dir = Path(__file__).resolve().parent
    default_stat = script_dir / "opalx-sim" / "pert-test-uniformsphere.stat"
    parser = argparse.ArgumentParser(
        description="Run direct Lienard-Wiechert Picard trajectory iterations."
    )
    parser.add_argument("--particles", type=int, default=DEFAULT_N_PARTICLES)
    parser.add_argument("--total-charge", type=float, default=DEFAULT_TOTAL_CHARGE_SI)
    parser.add_argument("--e-z", type=float, default=DEFAULT_E_Z_SI, help="External Ez in V/m.")
    parser.add_argument(
        "--initial-energy-gev",
        type=float,
        default=DEFAULT_INITIAL_ENERGY_GEV,
        help="Initial mean kinetic energy used for the longitudinal reference momentum.",
    )
    parser.add_argument("--t-end", type=float, default=DEFAULT_T_END_S)
    parser.add_argument("--outputs", type=int, default=DEFAULT_N_OUTPUTS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--relaxation", type=float, default=DEFAULT_RELAXATION)
    parser.add_argument("--integrator", choices=("euler", "heun"), default="heun")
    parser.add_argument("--retarded-iterations", type=int, default=DEFAULT_RETARDED_ITERATIONS)
    parser.add_argument(
        "--distribution",
        choices=("uniform-ellipsoid", "gaussian"),
        default=DEFAULT_DISTRIBUTION,
        help="Initial particle sampling model before exact mean/RMS matching.",
    )
    parser.add_argument(
        "--softening",
        type=float,
        default=DEFAULT_SOFTENING_M,
        help=(
            "Optional Plummer-like pair softening in meters.  This is a direct "
            "finite-size regularization, not grid deposition."
        ),
    )
    parser.add_argument("--r-min", type=float, default=DEFAULT_R_MIN_M)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output_picard",
        help="Directory for NPZ, CSV, JSON, and plot outputs.",
    )
    parser.add_argument(
        "--match-initial-stat",
        type=Path,
        default=default_stat if default_stat.exists() else None,
        help=(
            "Optional OPALX .stat file.  The first row supplies mean_x/mean_y/"
            "mean_s and rms_x/rms_y/rms_s plus rms_px/rms_py/rms_ps."
        ),
    )
    parser.add_argument(
        "--no-save-all-iterations",
        action="store_true",
        help="Only save final trajectories in trajectories.npz.",
    )
    args = parser.parse_args()

    if args.particles < 2:
        raise ValueError("--particles must be at least 2")
    if args.outputs < 2:
        raise ValueError("--outputs must be at least 2")
    if args.iterations < 0:
        raise ValueError("--iterations must be nonnegative")
    if not (0.0 < args.relaxation <= 1.0):
        raise ValueError("--relaxation must be in (0, 1]")
    if args.retarded_iterations < 0:
        raise ValueError("--retarded-iterations must be nonnegative")
    if args.softening < 0.0:
        raise ValueError("--softening must be nonnegative")
    if args.r_min < 0.0:
        raise ValueError("--r-min must be nonnegative")
    if args.t_end < 0.0:
        raise ValueError("--t-end must be nonnegative")
    if args.initial_energy_gev < 0.0:
        raise ValueError("--initial-energy-gev must be nonnegative")

    return PicardConfig(
        seed=args.seed,
        n_particles=args.particles,
        total_charge_si=args.total_charge,
        e_z_si=args.e_z,
        initial_energy_gev=args.initial_energy_gev,
        t_end_s=args.t_end,
        n_outputs=args.outputs,
        iterations=args.iterations,
        relaxation=args.relaxation,
        integrator=args.integrator,
        retarded_iterations=args.retarded_iterations,
        softening_m=args.softening,
        r_min_m=args.r_min,
        distribution=args.distribution,
        output_dir=str(args.output_dir),
        match_initial_stat=str(args.match_initial_stat) if args.match_initial_stat else None,
        save_all_iterations=not args.no_save_all_iterations,
    )


def kinetic_energy_gev_to_momentum_mev_c(kinetic_energy_gev: float) -> float:
    kinetic_energy_mev = kinetic_energy_gev * 1.0e3
    return float(np.sqrt(kinetic_energy_mev * (kinetic_energy_mev + 2.0 * M_E_MEV)))


def sample_ellipsoid_qmc(
    mean: np.ndarray,
    radii: np.ndarray,
    n_particles: int,
    seed: int,
) -> np.ndarray:
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    u = sampler.random(n=n_particles)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]

    radius = u1 ** (1.0 / 3.0)
    cos_theta = 1.0 - 2.0 * u2
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
    phi = 2.0 * np.pi * u3

    unit = np.empty((n_particles, 3))
    unit[:, 0] = radius * sin_theta * np.cos(phi)
    unit[:, 1] = radius * sin_theta * np.sin(phi)
    unit[:, 2] = radius * cos_theta
    return mean + radii * unit


def initial_targets(config: PicardConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position_mean = POSITION_MEAN_M.astype(float).copy()
    position_rms = POSITION_SIGMA_M.astype(float).copy()
    momentum_mean = np.array(
        [0.0, 0.0, kinetic_energy_gev_to_momentum_mev_c(config.initial_energy_gev)]
    )
    momentum_rms = MOMENTUM_RMS_MEV_C.astype(float).copy()

    if config.match_initial_stat:
        row = read_sdds_first_data_row(config.match_initial_stat)
        position_mean = np.array(
            [
                row.get("mean_x", position_mean[0]),
                row.get("mean_y", position_mean[1]),
                row.get("mean_s", position_mean[2]),
            ],
            dtype=float,
        )
        position_rms = np.array(
            [
                row.get("rms_x", position_rms[0]),
                row.get("rms_y", position_rms[1]),
                row.get("rms_s", position_rms[2]),
            ],
            dtype=float,
        )
        momentum_rms = M_E_MEV * np.array(
            [
                row.get("rms_px", momentum_rms[0] / M_E_MEV),
                row.get("rms_py", momentum_rms[1] / M_E_MEV),
                row.get("rms_ps", momentum_rms[2] / M_E_MEV),
            ],
            dtype=float,
        )

    return position_mean, position_rms, momentum_mean, momentum_rms


def sample_initial_conditions(config: PicardConfig) -> tuple[np.ndarray, np.ndarray]:
    position_mean, position_rms, momentum_mean, momentum_rms = initial_targets(config)

    if config.distribution == "uniform-ellipsoid":
        # Uniform ellipsoid radii are sqrt(5) times RMS for a filled 3D ellipsoid.
        r0_m = sample_ellipsoid_qmc(
            position_mean,
            np.sqrt(5.0) * position_rms,
            config.n_particles,
            config.seed,
        )
        p0_mev_c = sample_ellipsoid_qmc(
            momentum_mean,
            np.sqrt(5.0) * momentum_rms,
            config.n_particles,
            config.seed + 1,
        )
    elif config.distribution == "gaussian":
        rng = np.random.default_rng(config.seed)
        r0_m = rng.normal(position_mean, position_rms, (config.n_particles, 3))
        p0_mev_c = rng.normal(momentum_mean, momentum_rms, (config.n_particles, 3))
    else:
        raise ValueError(f"Unknown distribution {config.distribution!r}")

    r0_m = match_mean_and_rms(r0_m, position_mean, position_rms, "position")
    p0_mev_c = match_mean_and_rms(p0_mev_c, momentum_mean, momentum_rms, "momentum")
    return r0_m, p0_mev_c


def beta_dot_from_history(p_mev_c: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    beta = beta_from_p(p_mev_c)
    edge_order = 2 if len(times_s) > 2 else 1
    return np.gradient(beta, times_s, axis=0, edge_order=edge_order)


def interpolate_sources(
    values: np.ndarray,
    times_s: np.ndarray,
    t_query_s: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of source histories for all observer-source pairs.

    values has shape (T, N_source, C), t_query_s has shape (N_observer, N_source).
    The returned array has shape (N_observer, N_source, C).
    """
    dt = times_s[1] - times_s[0]
    scaled = (t_query_s - times_s[0]) / dt
    idx = np.floor(scaled).astype(np.int64)
    idx = np.clip(idx, 0, len(times_s) - 2)
    w = np.clip(scaled - idx, 0.0, 1.0)
    source_idx = np.arange(values.shape[1])[np.newaxis, :]
    return (
        (1.0 - w)[..., np.newaxis] * values[idx, source_idx, :]
        + w[..., np.newaxis] * values[idx + 1, source_idx, :]
    )


def retarded_source_state(
    t_obs_s: float,
    r_obs_m: np.ndarray,
    source_r_m: np.ndarray,
    source_p_mev_c: np.ndarray,
    source_beta_dot: np.ndarray,
    times_s: np.ndarray,
    n_fixed_point: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_particles = r_obs_m.shape[0]
    t_high = min(max(t_obs_s, times_s[0]), times_s[-1])
    if t_high <= times_s[0]:
        t_ret = np.full((n_particles, n_particles), times_s[0])
    else:
        t_idx = int(np.clip(round((t_high - times_s[0]) / (times_s[1] - times_s[0])), 0, len(times_s) - 1))
        distance_now = np.linalg.norm(r_obs_m[:, np.newaxis, :] - source_r_m[t_idx][np.newaxis, :, :], axis=-1)
        t_ret = np.clip(t_high - distance_now / C_SI, times_s[0], t_high)

    for _ in range(n_fixed_point):
        r_src = interpolate_sources(source_r_m, times_s, t_ret)
        distance = np.linalg.norm(r_obs_m[:, np.newaxis, :] - r_src, axis=-1)
        t_next = np.clip(t_high - distance / C_SI, times_s[0], t_high)
        if np.max(np.abs(t_next - t_ret)) < 1.0e-15:
            t_ret = t_next
            break
        t_ret = t_next

    return (
        interpolate_sources(source_r_m, times_s, t_ret),
        interpolate_sources(source_p_mev_c, times_s, t_ret),
        interpolate_sources(source_beta_dot, times_s, t_ret),
    )


def direct_lw_self_fields(
    t_obs_s: float,
    r_obs_m: np.ndarray,
    source_r_m: np.ndarray,
    source_p_mev_c: np.ndarray,
    source_beta_dot: np.ndarray,
    times_s: np.ndarray,
    config: PicardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_particles = r_obs_m.shape[0]
    if config.total_charge_si == 0.0 or t_obs_s <= times_s[0]:
        return np.zeros((n_particles, 3)), np.zeros((n_particles, 3))

    r_src, p_src, beta_dot_src = retarded_source_state(
        t_obs_s,
        r_obs_m,
        source_r_m,
        source_p_mev_c,
        source_beta_dot,
        times_s,
        config.retarded_iterations,
    )
    beta_src = beta_from_p(p_src)
    gamma_src = gamma_from_p(p_src)

    r_vec = r_obs_m[:, np.newaxis, :] - r_src
    r_raw = np.linalg.norm(r_vec, axis=-1)
    r_mag = np.sqrt(r_raw * r_raw + config.softening_m * config.softening_m)
    r_mag = np.maximum(r_mag, config.r_min_m)
    n_vec = r_vec / r_mag[..., np.newaxis]

    one_minus_n_beta = 1.0 - np.sum(n_vec * beta_src, axis=-1)
    one_minus_n_beta = np.where(
        np.abs(one_minus_n_beta) < 1.0e-14,
        np.copysign(1.0e-14, one_minus_n_beta),
        one_minus_n_beta,
    )

    n_minus_beta = n_vec - beta_src
    velocity_term = n_minus_beta / (
        gamma_src[..., np.newaxis] ** 2
        * one_minus_n_beta[..., np.newaxis] ** 3
        * r_mag[..., np.newaxis] ** 2
    )
    acceleration_term = np.cross(
        n_vec,
        np.cross(n_minus_beta, beta_dot_src),
    ) / (C_SI * one_minus_n_beta[..., np.newaxis] ** 3 * r_mag[..., np.newaxis])

    e_pair = config.q_macro_si * (velocity_term + acceleration_term) / (
        4.0 * np.pi * EPSILON_0_SI
    )
    b_pair = np.cross(n_vec, e_pair) / C_SI

    self_mask = np.eye(n_particles, dtype=bool)
    e_pair[self_mask] = 0.0
    b_pair[self_mask] = 0.0
    e_pair = np.nan_to_num(e_pair, copy=False)
    b_pair = np.nan_to_num(b_pair, copy=False)
    return np.sum(e_pair, axis=1), np.sum(b_pair, axis=1)


def rhs_from_previous_history(
    t_s: float,
    r_m: np.ndarray,
    p_mev_c: np.ndarray,
    previous_r_m: np.ndarray,
    previous_p_mev_c: np.ndarray,
    previous_beta_dot: np.ndarray,
    times_s: np.ndarray,
    config: PicardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    e_self, b_self = direct_lw_self_fields(
        t_s,
        r_m,
        previous_r_m,
        previous_p_mev_c,
        previous_beta_dot,
        times_s,
        config,
    )
    beta = beta_from_p(p_mev_c)
    velocity_si = beta * C_SI
    electric = e_self.copy()
    electric[:, 2] += config.e_z_si
    force_si = Q_E_SI * (electric + np.cross(velocity_si, b_self))
    dp_dt = force_si / P_MEV_C_SI
    dr_dt = velocity_si
    return dr_dt, dp_dt


def picard_map(
    previous_r_m: np.ndarray,
    previous_p_mev_c: np.ndarray,
    r0_m: np.ndarray,
    p0_mev_c: np.ndarray,
    times_s: np.ndarray,
    config: PicardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_times, n_particles, _ = previous_r_m.shape
    candidate_r = np.empty_like(previous_r_m)
    candidate_p = np.empty_like(previous_p_mev_c)
    candidate_r[0] = r0_m
    candidate_p[0] = p0_mev_c
    previous_beta_dot = beta_dot_from_history(previous_p_mev_c, times_s)

    for t_idx in range(1, n_times):
        t0 = float(times_s[t_idx - 1])
        t1 = float(times_s[t_idx])
        dt = t1 - t0
        r_old = candidate_r[t_idx - 1]
        p_old = candidate_p[t_idx - 1]
        dr0, dp0 = rhs_from_previous_history(
            t0,
            r_old,
            p_old,
            previous_r_m,
            previous_p_mev_c,
            previous_beta_dot,
            times_s,
            config,
        )

        if config.integrator == "euler":
            candidate_r[t_idx] = r_old + dt * dr0
            candidate_p[t_idx] = p_old + dt * dp0
        elif config.integrator == "heun":
            r_pred = r_old + dt * dr0
            p_pred = p_old + dt * dp0
            dr1, dp1 = rhs_from_previous_history(
                t1,
                r_pred,
                p_pred,
                previous_r_m,
                previous_p_mev_c,
                previous_beta_dot,
                times_s,
                config,
            )
            candidate_r[t_idx] = r_old + 0.5 * dt * (dr0 + dr1)
            candidate_p[t_idx] = p_old + 0.5 * dt * (dp0 + dp1)
        else:
            raise ValueError(f"Unknown integrator {config.integrator!r}")

        if not np.isfinite(candidate_r[t_idx]).all() or not np.isfinite(candidate_p[t_idx]).all():
            raise FloatingPointError(f"Non-finite Picard state at output {t_idx}/{n_times - 1}")

        if t_idx % max(1, n_times // 20) == 0 or t_idx == n_times - 1:
            print(f"  time step {t_idx}/{n_times - 1}", end="\r", flush=True)

    print(" " * 40, end="\r", flush=True)
    return candidate_r, candidate_p


def iteration_metrics(
    previous_r_m: np.ndarray,
    previous_p_mev_c: np.ndarray,
    new_r_m: np.ndarray,
    new_p_mev_c: np.ndarray,
) -> dict[str, float]:
    delta_r = new_r_m - previous_r_m
    delta_p = new_p_mev_c - previous_p_mev_c
    return {
        "max_r_rms_change_m": float(np.max(np.sqrt(np.mean(delta_r * delta_r, axis=(1, 2))))),
        "final_r_rms_change_m": float(np.sqrt(np.mean(delta_r[-1] * delta_r[-1]))),
        "max_p_rms_change_mev_c": float(np.max(np.sqrt(np.mean(delta_p * delta_p, axis=(1, 2))))),
        "final_p_rms_change_mev_c": float(np.sqrt(np.mean(delta_p[-1] * delta_p[-1]))),
    }


def moments_for_iteration(
    iteration: int,
    times_s: np.ndarray,
    r_m: np.ndarray,
    p_mev_c: np.ndarray,
) -> list[dict[str, float]]:
    p_bg = p_mev_c / M_E_MEV
    kinetic = kinetic_energy_mev(p_mev_c)
    mean_r = np.mean(r_m, axis=1)
    rms_r = np.std(r_m, axis=1)
    mean_p = np.mean(p_bg, axis=1)
    rms_p = np.std(p_bg, axis=1)
    mean_ke = np.mean(kinetic, axis=1)
    std_ke = np.std(kinetic, axis=1)
    rows: list[dict[str, float]] = []
    for t_idx, t_s in enumerate(times_s):
        rows.append(
            {
                "iteration": iteration,
                "time_s": float(t_s),
                "time_ns": float(t_s * 1.0e9),
                "mean_x_m": float(mean_r[t_idx, 0]),
                "mean_y_m": float(mean_r[t_idx, 1]),
                "mean_z_m": float(mean_r[t_idx, 2]),
                "rms_x_m": float(rms_r[t_idx, 0]),
                "rms_y_m": float(rms_r[t_idx, 1]),
                "rms_z_m": float(rms_r[t_idx, 2]),
                "mean_px_beta_gamma": float(mean_p[t_idx, 0]),
                "mean_py_beta_gamma": float(mean_p[t_idx, 1]),
                "mean_pz_beta_gamma": float(mean_p[t_idx, 2]),
                "rms_px_beta_gamma": float(rms_p[t_idx, 0]),
                "rms_py_beta_gamma": float(rms_p[t_idx, 1]),
                "rms_pz_beta_gamma": float(rms_p[t_idx, 2]),
                "mean_kinetic_energy_MeV": float(mean_ke[t_idx]),
                "energy_spread_MeV": float(std_ke[t_idx]),
            }
        )
    return rows


def write_moments_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No moment rows to write")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_final_moments_csv(path: Path, times_s: np.ndarray, r_m: np.ndarray, p_mev_c: np.ndarray) -> None:
    rows = moments_for_iteration(0, times_s, r_m, p_mev_c)
    rename = {
        "time_s": "time_s",
        "mean_x_m": "mean_x_m",
        "mean_y_m": "mean_y_m",
        "mean_z_m": "mean_z_m",
        "rms_x_m": "rms_x_m",
        "rms_y_m": "rms_y_m",
        "rms_z_m": "rms_z_m",
        "mean_px_beta_gamma": "mean_px_beta_gamma",
        "mean_py_beta_gamma": "mean_py_beta_gamma",
        "mean_pz_beta_gamma": "mean_pz_beta_gamma",
        "rms_px_beta_gamma": "rms_px_beta_gamma",
        "rms_py_beta_gamma": "rms_py_beta_gamma",
        "rms_pz_beta_gamma": "rms_pz_beta_gamma",
        "mean_kinetic_energy_MeV": "mean_kinetic_energy_MeV",
    }
    final_rows = [{dst: row[src] for src, dst in rename.items()} for row in rows]
    write_moments_csv(path, final_rows)


def save_iteration_plots(
    output_dir: Path,
    times_s: np.ndarray,
    r_iterations: list[np.ndarray],
    p_iterations: list[np.ndarray],
    metrics: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ("x", "y", "z")
    times_ns = times_s * 1.0e9

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for iteration, r_m in enumerate(r_iterations):
        rms = np.std(r_m, axis=1)
        for axis, label in enumerate(labels):
            axes[axis].plot(times_ns, rms[:, axis], label=f"iter {iteration}")
            axes[axis].set_title(f"rms_{label}")
            axes[axis].set_xlabel("time [ns]")
            axes[axis].set_ylabel("RMS position [m]")
            axes[axis].grid(True, alpha=0.35)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "picard_rms_positions_by_iteration.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for iteration, p_mev_c in enumerate(p_iterations):
        rms = np.std(p_mev_c / M_E_MEV, axis=1)
        for axis, label in enumerate(labels):
            axes[axis].plot(times_ns, rms[:, axis], label=f"iter {iteration}")
            axes[axis].set_title(f"rms_p{label}")
            axes[axis].set_xlabel("time [ns]")
            axes[axis].set_ylabel("RMS momentum [beta*gamma]")
            axes[axis].grid(True, alpha=0.35)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "picard_rms_momenta_by_iteration.png", dpi=180)
    plt.close(fig)

    if metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        iteration_axis = np.arange(1, len(metrics) + 1)
        ax.semilogy(
            iteration_axis,
            [row["max_r_rms_change_m"] for row in metrics],
            marker="o",
            label="max RMS position change",
        )
        ax.semilogy(
            iteration_axis,
            [row["max_p_rms_change_mev_c"] for row in metrics],
            marker="o",
            label="max RMS momentum change",
        )
        ax.set_xlabel("Picard iteration")
        ax.set_ylabel("RMS update norm")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "picard_convergence.png", dpi=180)
        plt.close(fig)


def run(config: PicardConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    times_s = np.linspace(0.0, config.t_end_s, config.n_outputs)

    print("sampling initial conditions", flush=True)
    r0_m, p0_mev_c = sample_initial_conditions(config)
    print(
        "initial RMS: "
        f"r={np.std(r0_m, axis=0)} m, "
        f"p={np.std(p0_mev_c / M_E_MEV, axis=0)} beta*gamma",
        flush=True,
    )

    print("iteration 0: external-field trajectory", flush=True)
    r_external, p_external = vectorized_unperturbed(
        times_s,
        r0_m,
        p0_mev_c,
        config.dpz_dt_mev_c_per_s,
    )
    r_iterations = [r_external]
    p_iterations = [p_external]
    metrics: list[dict[str, float]] = []

    for iteration in range(1, config.iterations + 1):
        print(f"iteration {iteration}: direct LW Picard map", flush=True)
        candidate_r, candidate_p = picard_map(
            r_iterations[-1],
            p_iterations[-1],
            r0_m,
            p0_mev_c,
            times_s,
            config,
        )
        if config.relaxation < 1.0:
            candidate_r = (1.0 - config.relaxation) * r_iterations[-1] + config.relaxation * candidate_r
            candidate_p = (1.0 - config.relaxation) * p_iterations[-1] + config.relaxation * candidate_p
            candidate_r[0] = r0_m
            candidate_p[0] = p0_mev_c

        update = iteration_metrics(r_iterations[-1], p_iterations[-1], candidate_r, candidate_p)
        metrics.append(update)
        print(
            "  update: "
            f"max_r={update['max_r_rms_change_m']:.3e} m, "
            f"max_p={update['max_p_rms_change_mev_c']:.3e} MeV/c",
            flush=True,
        )
        r_iterations.append(candidate_r)
        p_iterations.append(candidate_p)

    all_moment_rows: list[dict[str, float]] = []
    for iteration, (r_m, p_mev_c) in enumerate(zip(r_iterations, p_iterations, strict=True)):
        all_moment_rows.extend(moments_for_iteration(iteration, times_s, r_m, p_mev_c))
    write_moments_csv(output_dir / "picard_moments_by_iteration.csv", all_moment_rows)
    write_final_moments_csv(output_dir / "moments.csv", times_s, r_iterations[-1], p_iterations[-1])

    save_payload = {
        "times_s": times_s,
        "initial_r_m": r0_m,
        "initial_p_MeVc": p0_mev_c,
        "r_m": r_iterations[-1],
        "p_MeVc": p_iterations[-1],
        "p_beta_gamma": p_iterations[-1] / M_E_MEV,
        "kinetic_energy_MeV": kinetic_energy_mev(p_iterations[-1]),
        "final_r_m": r_iterations[-1],
        "final_p_MeVc": p_iterations[-1],
    }
    if config.save_all_iterations:
        save_payload["r_iterations_m"] = np.stack(r_iterations)
        save_payload["p_iterations_MeVc"] = np.stack(p_iterations)
        save_payload["kinetic_energy_iterations_MeV"] = np.stack(
            [kinetic_energy_mev(p_mev_c) for p_mev_c in p_iterations]
        )
    np.savez_compressed(output_dir / "trajectories.npz", **save_payload)

    with (output_dir / "picard_iteration_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(
            {
                "description": "Direct analytic Lienard-Wiechert Picard iteration benchmark.",
                "config": asdict(config),
                "initial": {
                    "position_mean_m": np.mean(r0_m, axis=0).tolist(),
                    "position_rms_m": np.std(r0_m, axis=0).tolist(),
                    "momentum_mean_MeVc": np.mean(p0_mev_c, axis=0).tolist(),
                    "momentum_rms_beta_gamma": np.std(p0_mev_c / M_E_MEV, axis=0).tolist(),
                },
                "outputs": {
                    "moments_by_iteration": "picard_moments_by_iteration.csv",
                    "final_moments": "moments.csv",
                    "trajectories": "trajectories.npz",
                },
            },
            f,
            indent=2,
        )
    save_iteration_plots(output_dir, times_s, r_iterations, p_iterations, metrics)
    print(f"wrote outputs to {output_dir}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
