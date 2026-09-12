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


def padded_limits(values, padding=0.05):
    lower = np.nanmin(values)
    upper = np.nanmax(values)
    span = upper - lower
    margin = padding * span if span > 0 else max(abs(lower) * padding, 1e-6)
    return lower - margin, upper + margin


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

    z_values = np.concatenate(centered_z)
    u_values = np.concatenate(longitudinal_u)
    z_limit = np.percentile(z_values, [0.5, 99.5])
    u_limit = np.percentile(u_values, [0.5, 99.5])
    z_plot_limits = padded_limits(z_limit, 0.08)
    u_plot_limits = padded_limits(u_limit, 0.08)

    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.2))

    all_positions = np.concatenate([snapshot["positions_m"] for snapshot in snapshots])
    transverse_limit = max(
        np.max(np.abs(all_positions[:, 0])),
        np.max(np.abs(all_positions[:, 1])),
    )
    z_limits = (np.min(all_positions[:, 2]), np.max(all_positions[:, 2]))
    z_axis_length = max(np.ptp(all_positions[:, 2]), 1e-3)
    transverse_axis_length = z_axis_length / 2.0

    for column, (snapshot, z, u) in enumerate(zip(plotted_snapshots, centered_z, longitudinal_u)):
        histogram_axis = fig.add_subplot(grid[0, column])
        histogram_axis.scatter(z, u, s=4, alpha=0.35, linewidths=0)
        histogram_axis.set_title(SNAPSHOT_TITLES[column])
        histogram_axis.set_xlabel(r"$z - \mu_z$ [m]")
        histogram_axis.set_ylabel(r"$u_z = p_z/(m_ec)$ [$\beta\gamma$]")
        histogram_axis.set_xlim(z_plot_limits)
        histogram_axis.set_ylim(u_plot_limits)
        histogram_axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

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
        spatial_axis.set_xlim(-transverse_limit, transverse_limit)
        spatial_axis.set_ylim(-transverse_limit, transverse_limit)
        spatial_axis.set_zlim(*z_limits)
        spatial_axis.xaxis.set_major_locator(MaxNLocator(3))
        spatial_axis.yaxis.set_major_locator(MaxNLocator(3))
        spatial_axis.set_box_aspect(
            (
                transverse_axis_length,
                transverse_axis_length,
                z_axis_length,
            )
        )

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
