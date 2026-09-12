from pathlib import Path
import sys

import matplotlib

if "IPython" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

np.random.seed(33)


SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT_DIR = SCRIPT_DIR / "data" / "pic_snapshots"
OUTPUT_DIR = SCRIPT_DIR / "figures"
M_E_MEV_C2 = 0.511
MAX_PLOTTED_PARTICLES = 5000
SNAPSHOT_FILE_SUFFIXES = (
    "pic_snapshot_0.00ns.npz",
    "pic_snapshot_3.75ns.npz",
    "pic_snapshot_12.50ns.npz",
)
SNAPSHOT_TITLES = ("0 ns", "3.75 ns", "12.50 ns")


def load_snapshots(setup_name):
    snapshots = []
    for suffix in SNAPSHOT_FILE_SUFFIXES:
        filename = f"{setup_name}_{suffix}"
        path = SNAPSHOT_DIR / filename
        with np.load(path) as data:
            snapshots.append(
                {
                    "time_s": float(data["time_s"]),
                    "positions_m": np.asarray(data["positions_m"]),
                    "momenta_MeV_c": np.asarray(data["momenta_MeV_c"]),
                }
            )
    return snapshots


def make_figure(snapshots):
    plotted_snapshots = []
    for snapshot in snapshots:
        particle_count = len(snapshot["positions_m"])
        sample_size = min(MAX_PLOTTED_PARTICLES, particle_count)
        sample_indices = np.random.choice(particle_count, size=sample_size, replace=False)
        plotted_snapshots.append(
            {
                "time_s": snapshot["time_s"],
                "positions_m": snapshot["positions_m"][sample_indices],
                "momenta_MeV_c": snapshot["momenta_MeV_c"][sample_indices],
            }
        )

    centered_z = [snapshot["positions_m"][:, 2] - np.mean(snapshot["positions_m"][:, 2]) for snapshot in plotted_snapshots]
    longitudinal_u = [snapshot["momenta_MeV_c"][:, 2] / M_E_MEV_C2 for snapshot in plotted_snapshots]

    # Calculate global aspect ratios to enforce consistently across local zooms
    z_values_all = np.concatenate(centered_z)
    u_values_all = np.concatenate(longitudinal_u)
    z_global_min, z_global_max = np.percentile(z_values_all, [0.5, 99.5])
    u_global_min, u_global_max = np.percentile(u_values_all, [0.5, 99.5])
    global_z_span = max(z_global_max - z_global_min, 1e-12)
    global_u_span = max(u_global_max - u_global_min, 1e-12)
    phase_ratio = global_u_span / global_z_span

    all_positions = np.concatenate([snapshot["positions_m"] for snapshot in snapshots])
    global_z_span_sp = max(np.ptp(all_positions[:, 2]), 1e-12)
    global_transverse_span = max(
        2 * max(np.max(np.abs(all_positions[:, 0])), np.max(np.abs(all_positions[:, 1]))),
        1e-12
    )
    spatial_ratio = global_z_span_sp / global_transverse_span

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.2))

    for column, (snapshot, z, u) in enumerate(zip(plotted_snapshots, centered_z, longitudinal_u)):
        # --- Momentum Phase Space ---
        histogram_axis = fig.add_subplot(grid[0, column])
        histogram_axis.scatter(z, u, s=4, alpha=0.35, linewidths=0)
        histogram_axis.set_title(SNAPSHOT_TITLES[column])
        histogram_axis.set_xlabel(r"$z - \mu_z$ [m]")
        histogram_axis.set_ylabel(r"$u_z = p_z/(m_ec)$ [$\beta\gamma$]")

        # Evaluate local data spans and expand the shorter dimension to match the global aspect ratio
        local_z_min, local_z_max = np.percentile(z, [0.5, 99.5])
        local_u_min, local_u_max = np.percentile(u, [0.5, 99.5])
        local_z_span = max(local_z_max - local_z_min, 1e-12)
        local_u_span = max(local_u_max - local_u_min, 1e-12)

        if local_u_span / local_z_span > phase_ratio:
            target_u_span = local_u_span
            target_z_span = target_u_span / phase_ratio
        else:
            target_z_span = local_z_span
            target_u_span = target_z_span * phase_ratio

        # Apply 16% total margin (equivalent to 8% per side from the original padded_limits)
        target_z_span *= 1.16
        target_u_span *= 1.16

        z_center = (local_z_min + local_z_max) / 2.0
        u_center = (local_u_min + local_u_max) / 2.0

        histogram_axis.set_xlim(z_center - target_z_span / 2.0, z_center + target_z_span / 2.0)
        histogram_axis.set_ylim(u_center - target_u_span / 2.0, u_center + target_u_span / 2.0)
        histogram_axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

        # --- Spatial Distribution ---
        spatial_axis = fig.add_subplot(grid[1, column], projection="3d")
        positions = snapshot["positions_m"]
        spatial_axis.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=7,
            alpha=0.65,
            linewidths=0,
        )
        spatial_axis.set_xlabel("x [m]")
        spatial_axis.set_ylabel("y [m]")
        spatial_axis.set_zlabel("z [m]")

        # Enforce constant spatial aspect ratio based on local coordinates
        local_z_min_sp, local_z_max_sp = np.min(positions[:, 2]), np.max(positions[:, 2])
        local_z_span_sp = max(local_z_max_sp - local_z_min_sp, 1e-12)
        local_transverse_span = max(
            2 * max(np.max(np.abs(positions[:, 0])), np.max(np.abs(positions[:, 1]))),
            1e-12
        )

        if local_z_span_sp / local_transverse_span > spatial_ratio:
            target_z_span_sp = local_z_span_sp
            target_transverse_span = target_z_span_sp / spatial_ratio
        else:
            target_transverse_span = local_transverse_span
            target_z_span_sp = target_transverse_span * spatial_ratio

        # Apply 10% total spatial margin
        target_z_span_sp *= 1.10
        target_transverse_span *= 1.10

        z_center_sp = (local_z_min_sp + local_z_max_sp) / 2.0
        transverse_limit = target_transverse_span / 2.0

        spatial_axis.set_xlim(-transverse_limit, transverse_limit)
        spatial_axis.set_ylim(-transverse_limit, transverse_limit)
        spatial_axis.set_zlim(z_center_sp - target_z_span_sp / 2.0, z_center_sp + target_z_span_sp / 2.0)
        
        spatial_axis.xaxis.set_major_locator(MaxNLocator(3))
        spatial_axis.yaxis.set_major_locator(MaxNLocator(3))
        
        # Enforce strict uniform visual framing (translates to 1:1:2 based on the original Z/2 scaling)
        spatial_axis.set_box_aspect((1, 1, 2))

    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for setup_name in ("simple", "complex"):
        snapshots = load_snapshots(setup_name)
        figure = make_figure(snapshots)
        figure.savefig(OUTPUT_DIR / f"{setup_name}_pic_snapshots_2x3.pdf", bbox_inches="tight")
        figure.savefig(OUTPUT_DIR / f"{setup_name}_pic_snapshots_2x3.svg", bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()