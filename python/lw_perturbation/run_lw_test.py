#!/usr/bin/env python3
"""Parallel 3D Lienard-Wiechert perturbation test case.

The default run is intentionally a small but nontrivial reproducible benchmark:
256 Gaussian electron macroparticles, -1 nC total bunch charge, zero mean
initial momentum, 1 eV/c RMS thermal momentum per axis, and a global constant
longitudinal electric field. The direct pair fields use a dynamic finite-size
kernel derived from a virtual 128^3 grid over the bunch RMS extent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


M_E_MEV = 0.511
Q_E_SI = -1.602176634e-19
E_CHARGE_ABS_SI = 1.602176634e-19
EPSILON_0_SI = 8.85418782e-12
C_SI = 299792458.0
P_MEV_C_SI = 1.0e6 * E_CHARGE_ABS_SI / C_SI

DEFAULT_SEED = 33
DEFAULT_N_PARTICLES = 256
DEFAULT_TOTAL_CHARGE_SI = -1.0e-9
DEFAULT_E_Z_SI = -1.0e6
DEFAULT_T_END = 15.0e-9
DEFAULT_N_OUTPUTS = 151
DEFAULT_R_MIN = 1.0e-9
DEFAULT_SMOOTHING_GRID_CELLS = 128
DEFAULT_SMOOTHING_EXTENT_SIGMAS = 3.0
DEFAULT_RTOL = 1.0e-5
DEFAULT_ATOL = 1.0e-10

POSITION_MEAN_M = np.array([0.0, 0.0, 0.0])
POSITION_SIGMA_M = np.array([1.0e-3, 1.0e-3, 1.0e-3])
MOMENTUM_MEAN_MEV_C = np.array([0.0, 0.0, 0.0])
MOMENTUM_SIGMA_MEV_C = np.array([1.0e-6, 1.0e-6, 1.0e-6])


@dataclass(frozen=True)
class RunConfig:
    seed: int
    n_particles: int
    total_charge_si: float
    e_z_si: float
    t_end_s: float
    n_outputs: int
    r_min_m: float
    smoothing_grid_cells: int
    smoothing_extent_sigmas: float
    rtol: float
    atol: float
    workers: int
    output_dir: str

    @property
    def q_macro_si(self) -> float:
        return self.total_charge_si / self.n_particles

    @property
    def e_z_mvm(self) -> float:
        return self.e_z_si / 1.0e6

    @property
    def dpz_dt_mev_c_per_s(self) -> float:
        force_si = Q_E_SI * self.e_z_si
        return force_si / P_MEV_C_SI


def parse_args() -> RunConfig:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run a parallel 3D Lienard-Wiechert perturbation benchmark."
    )
    parser.add_argument("--particles", type=int, default=DEFAULT_N_PARTICLES)
    parser.add_argument("--total-charge", type=float, default=DEFAULT_TOTAL_CHARGE_SI)
    parser.add_argument("--e-z", type=float, default=DEFAULT_E_Z_SI, help="External Ez in V/m.")
    parser.add_argument("--t-end", type=float, default=DEFAULT_T_END, help="End time in seconds.")
    parser.add_argument("--outputs", type=int, default=DEFAULT_N_OUTPUTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--r-min", type=float, default=DEFAULT_R_MIN)
    parser.add_argument(
        "--smoothing-grid-cells",
        type=int,
        default=DEFAULT_SMOOTHING_GRID_CELLS,
        help=(
            "Cells per axis used to derive the dynamic kernel smoothing length. "
            "Use 0 to disable dynamic smoothing."
        ),
    )
    parser.add_argument(
        "--smoothing-extent-sigmas",
        type=float,
        default=DEFAULT_SMOOTHING_EXTENT_SIGMAS,
        help="Half-width of the virtual smoothing grid in bunch RMS units.",
    )
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count. Defaults to min(os.cpu_count(), particles).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output",
        help="Directory for CSV, NPZ, JSON, and plot outputs.",
    )
    args = parser.parse_args()

    if args.particles < 1:
        raise ValueError("--particles must be positive")
    if args.outputs < 2:
        raise ValueError("--outputs must be at least 2")
    if args.t_end < 0:
        raise ValueError("--t-end must be nonnegative")
    if args.r_min < 0:
        raise ValueError("--r-min must be nonnegative")
    if args.smoothing_grid_cells < 0:
        raise ValueError("--smoothing-grid-cells must be nonnegative")
    if args.smoothing_extent_sigmas <= 0:
        raise ValueError("--smoothing-extent-sigmas must be positive")
    workers = args.workers
    if workers is None:
        workers = min(os.cpu_count() or 1, args.particles)
    if workers < 1:
        raise ValueError("--workers must be positive")

    return RunConfig(
        seed=args.seed,
        n_particles=args.particles,
        total_charge_si=args.total_charge,
        e_z_si=args.e_z,
        t_end_s=args.t_end,
        n_outputs=args.outputs,
        r_min_m=args.r_min,
        smoothing_grid_cells=args.smoothing_grid_cells,
        smoothing_extent_sigmas=args.smoothing_extent_sigmas,
        rtol=args.rtol,
        atol=args.atol,
        workers=workers,
        output_dir=str(args.output_dir),
    )


def gamma_from_p(p_mev_c: np.ndarray) -> np.ndarray:
    return np.sqrt(1.0 + np.sum(p_mev_c * p_mev_c, axis=-1) / (M_E_MEV * M_E_MEV))


def kinetic_energy_mev(p_mev_c: np.ndarray) -> np.ndarray:
    return (gamma_from_p(p_mev_c) - 1.0) * M_E_MEV


def beta_from_p(p_mev_c: np.ndarray) -> np.ndarray:
    gamma = gamma_from_p(p_mev_c)[..., np.newaxis]
    return p_mev_c / (gamma * M_E_MEV)


def sample_initial_conditions(config: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    r0_m = rng.normal(POSITION_MEAN_M, POSITION_SIGMA_M, (config.n_particles, 3))
    p0_mev_c = rng.normal(MOMENTUM_MEAN_MEV_C, MOMENTUM_SIGMA_MEV_C, (config.n_particles, 3))
    return r0_m, p0_mev_c


def trajectory_at_time(
    t_s: float,
    r0_m: np.ndarray,
    p0_mev_c: np.ndarray,
    dpz_dt_mev_c_per_s: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    p_t = np.array(
        [
            p0_mev_c[0],
            p0_mev_c[1],
            p0_mev_c[2] + dpz_dt_mev_c_per_s * t_s,
        ],
        dtype=float,
    )

    p_perp2 = p0_mev_c[0] * p0_mev_c[0] + p0_mev_c[1] * p0_mev_c[1]
    m_eff = math.sqrt(M_E_MEV * M_E_MEV + p_perp2)
    pz0 = p0_mev_c[2]
    pzt = p_t[2]

    r_t = np.array(r0_m, dtype=float)
    if abs(dpz_dt_mev_c_per_s) < 1.0e-300:
        gamma0 = math.sqrt(1.0 + np.dot(p0_mev_c, p0_mev_c) / (M_E_MEV * M_E_MEV))
        r_t += (p0_mev_c / (gamma0 * M_E_MEV)) * C_SI * t_s
    else:
        asinh_delta = math.asinh(pzt / m_eff) - math.asinh(pz0 / m_eff)
        transverse_factor = C_SI * asinh_delta / dpz_dt_mev_c_per_s
        r_t[0] += p0_mev_c[0] * transverse_factor
        r_t[1] += p0_mev_c[1] * transverse_factor

        energy_delta = math.sqrt(m_eff * m_eff + pzt * pzt) - math.sqrt(
            m_eff * m_eff + pz0 * pz0
        )
        r_t[2] += C_SI * energy_delta / dpz_dt_mev_c_per_s

    gamma_t = math.sqrt(1.0 + np.dot(p_t, p_t) / (M_E_MEV * M_E_MEV))
    beta_t = p_t / (gamma_t * M_E_MEV)

    p_dot = np.array([0.0, 0.0, dpz_dt_mev_c_per_s])
    s = gamma_t * M_E_MEV
    beta_dot = p_dot / s - p_t * np.dot(p_t, p_dot) / (s * s * s)

    return r_t, p_t, gamma_t, beta_t, beta_dot


def vectorized_unperturbed(
    times_s: np.ndarray,
    r0_m: np.ndarray,
    p0_mev_c: np.ndarray,
    dpz_dt_mev_c_per_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_t = len(times_s)
    n_p = len(r0_m)
    r = np.empty((n_t, n_p, 3), dtype=float)
    p = np.empty((n_t, n_p, 3), dtype=float)
    for t_idx, t_s in enumerate(times_s):
        for i in range(n_p):
            r[t_idx, i], p[t_idx, i], _, _, _ = trajectory_at_time(
                float(t_s), r0_m[i], p0_mev_c[i], dpz_dt_mev_c_per_s
            )
    return r, p


def unperturbed_positions_at_time(
    t_s: float,
    r0_m: np.ndarray,
    p0_mev_c: np.ndarray,
    dpz_dt_mev_c_per_s: float,
) -> np.ndarray:
    r = np.empty_like(r0_m)
    for i in range(len(r0_m)):
        r[i], _, _, _, _ = trajectory_at_time(
            float(t_s), r0_m[i], p0_mev_c[i], dpz_dt_mev_c_per_s
        )
    return r


def dynamic_smoothing_length_m(
    t_s: float,
    r0_m: np.ndarray,
    p0_mev_c: np.ndarray,
    config: RunConfig,
) -> float:
    if config.smoothing_grid_cells == 0:
        return config.r_min_m

    r_t_m = unperturbed_positions_at_time(
        t_s, r0_m, p0_mev_c, config.dpz_dt_mev_c_per_s
    )
    rms_m = np.std(r_t_m, axis=0)
    full_width_m = 2.0 * config.smoothing_extent_sigmas * rms_m
    cell_width_m = full_width_m / config.smoothing_grid_cells
    cell_diagonal_m = float(np.linalg.norm(cell_width_m))
    return max(config.r_min_m, cell_diagonal_m)


def retarded_time(
    t_obs_s: float,
    r_obs_m: np.ndarray,
    r0_src_m: np.ndarray,
    p0_src_mev_c: np.ndarray,
    dpz_dt_mev_c_per_s: float,
) -> float:
    def light_cone_residual(t_ret_s: float) -> float:
        r_src_m, _, _, _, _ = trajectory_at_time(
            t_ret_s, r0_src_m, p0_src_mev_c, dpz_dt_mev_c_per_s
        )
        return C_SI * (t_obs_s - t_ret_s) - float(np.linalg.norm(r_obs_m - r_src_m))

    r_src_now, _, _, _, _ = trajectory_at_time(
        t_obs_s, r0_src_m, p0_src_mev_c, dpz_dt_mev_c_per_s
    )
    distance_now = float(np.linalg.norm(r_obs_m - r_src_now))
    span = max(distance_now / C_SI, 1.0e-15)
    t_low = t_obs_s - 2.0 * span
    f_low = light_cone_residual(t_low)
    expansions = 0
    while f_low <= 0.0 and expansions < 80:
        span *= 2.0
        t_low = t_obs_s - 2.0 * span
        f_low = light_cone_residual(t_low)
        expansions += 1
    if f_low <= 0.0:
        raise RuntimeError("Could not bracket retarded-time root")

    f_high = light_cone_residual(t_obs_s)
    if abs(f_high) < 1.0e-18:
        return t_obs_s
    return float(brentq(light_cone_residual, t_low, t_obs_s, xtol=1.0e-14, rtol=1.0e-12))


def lienard_wiechert_fields_si(
    t_obs_s: float,
    r_obs_m: np.ndarray,
    r0_src_m: np.ndarray,
    p0_src_mev_c: np.ndarray,
    q_source_si: float,
    dpz_dt_mev_c_per_s: float,
    r_min_m: float,
    smoothing_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if q_source_si == 0.0:
        return np.zeros(3), np.zeros(3)

    t_r = retarded_time(t_obs_s, r_obs_m, r0_src_m, p0_src_mev_c, dpz_dt_mev_c_per_s)
    r_src_m, _, gamma_src, beta_src, beta_dot_src = trajectory_at_time(
        t_r, r0_src_m, p0_src_mev_c, dpz_dt_mev_c_per_s
    )

    r_vec = r_obs_m - r_src_m
    r_raw_m = float(np.linalg.norm(r_vec))
    softening_m = max(r_min_m, smoothing_length_m)
    if softening_m > 0.0:
        r_mag = math.sqrt(r_raw_m * r_raw_m + softening_m * softening_m)
    elif r_raw_m == 0.0:
        r_mag = max(r_min_m, 1.0e-300)
        r_vec = np.array([r_mag, 0.0, 0.0])
    else:
        r_mag = r_raw_m
    n_vec = r_vec / r_mag

    one_minus_n_beta = 1.0 - float(np.dot(n_vec, beta_src))
    one_minus_n_beta = math.copysign(max(abs(one_minus_n_beta), 1.0e-14), one_minus_n_beta)

    velocity_term = (n_vec - beta_src) / (
        gamma_src * gamma_src * one_minus_n_beta**3 * r_mag * r_mag
    )
    acceleration_term = np.cross(n_vec, np.cross(n_vec - beta_src, beta_dot_src)) / (
        C_SI * one_minus_n_beta**3 * r_mag
    )

    e_field = q_source_si * (velocity_term + acceleration_term) / (
        4.0 * math.pi * EPSILON_0_SI
    )
    b_field = np.cross(n_vec, e_field) / C_SI
    return e_field, b_field


def velocity_jacobian_si_per_mev_c(p_mev_c: np.ndarray) -> np.ndarray:
    gamma = float(gamma_from_p(p_mev_c))
    s = gamma * M_E_MEV
    return C_SI * (np.eye(3) / s - np.outer(p_mev_c, p_mev_c) / (s * s * s))


def perturbation_rhs(
    t_s: float,
    y: np.ndarray,
    target_idx: int,
    r0_all_m: np.ndarray,
    p0_all_mev_c: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
    p1_mev_c = y[:3]
    r_target_0_m, p_target_0_mev_c, _, beta_target, _ = trajectory_at_time(
        t_s,
        r0_all_m[target_idx],
        p0_all_mev_c[target_idx],
        config.dpz_dt_mev_c_per_s,
    )
    r_obs_m = r_target_0_m + y[3:6]
    v_target_si = beta_target * C_SI
    smoothing_length_m = dynamic_smoothing_length_m(
        t_s, r0_all_m, p0_all_mev_c, config
    )

    e_total = np.zeros(3)
    b_total = np.zeros(3)
    for src_idx in range(config.n_particles):
        if src_idx == target_idx:
            continue
        e_src, b_src = lienard_wiechert_fields_si(
            t_obs_s=t_s,
            r_obs_m=r_obs_m,
            r0_src_m=r0_all_m[src_idx],
            p0_src_mev_c=p0_all_mev_c[src_idx],
            q_source_si=config.q_macro_si,
            dpz_dt_mev_c_per_s=config.dpz_dt_mev_c_per_s,
            r_min_m=config.r_min_m,
            smoothing_length_m=smoothing_length_m,
        )
        e_total += e_src
        b_total += b_src

    force_si = Q_E_SI * (e_total + np.cross(v_target_si, b_total))
    dp1_dt_mev_c = force_si / P_MEV_C_SI
    dr1_dt_m = velocity_jacobian_si_per_mev_c(p_target_0_mev_c) @ p1_mev_c
    return np.concatenate((dp1_dt_mev_c, dr1_dt_m))


def solve_target(args: tuple[int, np.ndarray, np.ndarray, np.ndarray, RunConfig]) -> tuple[int, bool, str, np.ndarray, np.ndarray]:
    target_idx, times_s, r0_all_m, p0_all_mev_c, config = args
    y0 = np.zeros(6)
    try:
        if config.total_charge_si == 0.0 or times_s[0] == times_s[-1]:
            y = np.zeros((6, len(times_s)))
            return target_idx, True, "", y[:3].T, y[3:].T

        sol = solve_ivp(
            perturbation_rhs,
            (float(times_s[0]), float(times_s[-1])),
            y0,
            t_eval=times_s,
            args=(target_idx, r0_all_m, p0_all_mev_c, config),
            method="RK45",
            rtol=config.rtol,
            atol=config.atol,
        )
        if not sol.success:
            return target_idx, False, sol.message, np.full((len(times_s), 3), np.nan), np.full((len(times_s), 3), np.nan)
        return target_idx, True, "", sol.y[:3].T, sol.y[3:].T
    except Exception as exc:  # noqa: BLE001 - report worker failure without hiding index
        return target_idx, False, repr(exc), np.full((len(times_s), 3), np.nan), np.full((len(times_s), 3), np.nan)


def write_initial_conditions(output_dir: Path, r0_m: np.ndarray, p0_mev_c: np.ndarray, config: RunConfig) -> None:
    path = output_dir / "initial_conditions.csv"
    header = [
        "id",
        "x_m",
        "y_m",
        "z_m",
        "px_beta_gamma",
        "py_beta_gamma",
        "pz_beta_gamma",
        "px_MeVc",
        "py_MeVc",
        "pz_MeVc",
        "macro_charge_C",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, (r, p) in enumerate(zip(r0_m, p0_mev_c, strict=True)):
            writer.writerow(
                [
                    i,
                    f"{r[0]:.17e}",
                    f"{r[1]:.17e}",
                    f"{r[2]:.17e}",
                    f"{p[0] / M_E_MEV:.17e}",
                    f"{p[1] / M_E_MEV:.17e}",
                    f"{p[2] / M_E_MEV:.17e}",
                    f"{p[0]:.17e}",
                    f"{p[1]:.17e}",
                    f"{p[2]:.17e}",
                    f"{config.q_macro_si:.17e}",
                ]
            )


def write_metadata(output_dir: Path, config: RunConfig) -> None:
    metadata = {
        "description": "Parallel 3D Lienard-Wiechert perturbation benchmark in a constant Ez field.",
        "config": asdict(config),
        "constants": {
            "electron_mass_MeV_c2": M_E_MEV,
            "electron_charge_C": Q_E_SI,
            "epsilon0_SI": EPSILON_0_SI,
            "speed_of_light_m_per_s": C_SI,
            "MeV_c_in_kg_m_per_s": P_MEV_C_SI,
        },
        "initial_distribution": {
            "position_mean_m": POSITION_MEAN_M.tolist(),
            "position_rms_m": POSITION_SIGMA_M.tolist(),
            "momentum_mean_MeV_c": MOMENTUM_MEAN_MEV_C.tolist(),
            "momentum_rms_MeV_c": MOMENTUM_SIGMA_MEV_C.tolist(),
            "momentum_rms_beta_gamma": (MOMENTUM_SIGMA_MEV_C / M_E_MEV).tolist(),
        },
        "smoothing": {
            "kernel": "Plummer-like finite-size regularization of the LW separation",
            "grid_cells_per_axis": config.smoothing_grid_cells,
            "extent_half_width_rms": config.smoothing_extent_sigmas,
            "cell_size_rule": "one virtual grid-cell diagonal from the current unperturbed bunch RMS extent",
            "minimum_length_m": config.r_min_m,
        },
        "units": {
            "position": "m",
            "momentum": "MeV/c",
            "momentum_plot": "beta*gamma = p/(m_e c)",
            "kinetic_energy": "MeV",
            "electric_field": "V/m",
            "time": "s",
        },
    }
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def write_moments(
    output_dir: Path,
    times_s: np.ndarray,
    r_m: np.ndarray,
    p_mev_c: np.ndarray,
    kinetic_mev: np.ndarray,
) -> None:
    p_bg = p_mev_c / M_E_MEV
    mean_r = np.mean(r_m, axis=1)
    rms_r = np.std(r_m, axis=1)
    mean_p_bg = np.mean(p_bg, axis=1)
    rms_p_bg = np.std(p_bg, axis=1)
    mean_ke = np.mean(kinetic_mev, axis=1)

    path = output_dir / "moments.csv"
    header = [
        "time_s",
        "mean_x_m",
        "mean_y_m",
        "mean_z_m",
        "rms_x_m",
        "rms_y_m",
        "rms_z_m",
        "mean_px_beta_gamma",
        "mean_py_beta_gamma",
        "mean_pz_beta_gamma",
        "rms_px_beta_gamma",
        "rms_py_beta_gamma",
        "rms_pz_beta_gamma",
        "mean_kinetic_energy_MeV",
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, t_s in enumerate(times_s):
            writer.writerow(
                [f"{t_s:.17e}"]
                + [f"{v:.17e}" for v in mean_r[i]]
                + [f"{v:.17e}" for v in rms_r[i]]
                + [f"{v:.17e}" for v in mean_p_bg[i]]
                + [f"{v:.17e}" for v in rms_p_bg[i]]
                + [f"{mean_ke[i]:.17e}"]
            )


def save_plots(output_dir: Path, times_s: np.ndarray, r_m: np.ndarray, p_mev_c: np.ndarray, kinetic_mev: np.ndarray) -> None:
    labels = ("x", "y", "z")
    times_ns = times_s * 1.0e9

    fig, ax = plt.subplots(figsize=(8, 5))
    rms_r = np.std(r_m, axis=1)
    for i, label in enumerate(labels):
        ax.plot(times_ns, rms_r[:, i], label=label)
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("RMS position [m]")
    ax.set_title("RMS positions vs time")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rms_positions_vs_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    rms_p_bg = np.std(p_mev_c / M_E_MEV, axis=1)
    for i, label in enumerate(labels):
        ax.plot(times_ns, rms_p_bg[:, i], label=label)
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("RMS momentum [beta*gamma]")
    ax.set_title("RMS momenta vs time")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "rms_momenta_betagamma_vs_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_ns, np.mean(kinetic_mev, axis=1))
    ax.set_xlabel("time [ns]")
    ax.set_ylabel("Mean kinetic energy [MeV]")
    ax.set_title("Mean kinetic energy vs time")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_kinetic_energy_vs_time.png", dpi=180)
    plt.close(fig)


def solve_all_targets(
    times_s: np.ndarray,
    r0_m: np.ndarray,
    p0_mev_c: np.ndarray,
    config: RunConfig,
) -> tuple[np.ndarray, np.ndarray]:
    p1 = np.empty((len(times_s), config.n_particles, 3), dtype=float)
    r1 = np.empty((len(times_s), config.n_particles, 3), dtype=float)
    work_items: Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray, RunConfig]] = (
        (idx, times_s, r0_m, p0_mev_c, config) for idx in range(config.n_particles)
    )

    failures: list[str] = []
    if config.workers == 1:
        for item in work_items:
            idx, success, message, p1_i, r1_i = solve_target(item)
            p1[:, idx, :] = p1_i
            r1[:, idx, :] = r1_i
            if not success:
                failures.append(f"target {idx}: {message}")
            print(f"finished target {idx + 1}/{config.n_particles}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            future_to_idx = {executor.submit(solve_target, item): item[0] for item in work_items}
            completed = 0
            for future in as_completed(future_to_idx):
                idx, success, message, p1_i, r1_i = future.result()
                p1[:, idx, :] = p1_i
                r1[:, idx, :] = r1_i
                completed += 1
                if not success:
                    failures.append(f"target {idx}: {message}")
                print(f"finished target {completed}/{config.n_particles}", flush=True)

    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(f"{len(failures)} target integrations failed:\n{preview}")
    return r1, p1


def run(config: RunConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("sampling initial conditions", flush=True)
    r0_m, p0_mev_c = sample_initial_conditions(config)
    times_s = np.linspace(0.0, config.t_end_s, config.n_outputs)

    write_initial_conditions(output_dir, r0_m, p0_mev_c, config)
    write_metadata(output_dir, config)

    print("computing unperturbed external-field trajectories", flush=True)
    r0_t_m, p0_t_mev_c = vectorized_unperturbed(
        times_s, r0_m, p0_mev_c, config.dpz_dt_mev_c_per_s
    )
    smoothing_t_m = np.array(
        [
            dynamic_smoothing_length_m(float(t_s), r0_m, p0_mev_c, config)
            for t_s in times_s
        ]
    )

    print(f"solving first-order perturbations with {config.workers} workers", flush=True)
    r1_t_m, p1_t_mev_c = solve_all_targets(times_s, r0_m, p0_mev_c, config)

    r_t_m = r0_t_m + r1_t_m
    p_t_mev_c = p0_t_mev_c + p1_t_mev_c
    p_t_beta_gamma = p_t_mev_c / M_E_MEV
    kinetic_t_mev = kinetic_energy_mev(p_t_mev_c)

    np.savez_compressed(
        output_dir / "trajectories.npz",
        times_s=times_s,
        unperturbed_r_m=r0_t_m,
        unperturbed_p_MeVc=p0_t_mev_c,
        perturbation_r_m=r1_t_m,
        perturbation_p_MeVc=p1_t_mev_c,
        r_m=r_t_m,
        p_MeVc=p_t_mev_c,
        p_beta_gamma=p_t_beta_gamma,
        kinetic_energy_MeV=kinetic_t_mev,
        smoothing_length_m=smoothing_t_m,
    )
    write_moments(output_dir, times_s, r_t_m, p_t_mev_c, kinetic_t_mev)
    save_plots(output_dir, times_s, r_t_m, p_t_mev_c, kinetic_t_mev)

    print(f"wrote outputs to {output_dir}", flush=True)


def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
