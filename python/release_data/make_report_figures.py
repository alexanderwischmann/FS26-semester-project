from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import re

plt.rcParams["font.family"] = "serif"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "figures"
OPAL_STAT_PATH = DATA_DIR / "GH200_Drift-4-open-bins.stat"

COLUMN_NAMES = (
    "time_s",
    "mean_x_m", "mean_y_m", "mean_z_m",
    "rms_x_m", "rms_y_m", "rms_z_m",
    "mean_px_beta_gamma", "mean_py_beta_gamma", "mean_pz_beta_gamma",
    "rms_px_beta_gamma", "rms_py_beta_gamma", "rms_pz_beta_gamma",
)

MODEL_STYLES = {
    "single_particle": {"label": "single particle", "color": "0.35", "linestyle": (0, (8, 3))},
    "liouville": {"label": "Liouville ensemble", "color": "tab:purple", "linestyle": (0, (3, 2)), "linewidth": 2.2, "zorder": 5},
    "1d_envelope": {"label": "1D Euler envelope", "color": "tab:blue", "linestyle": "-"},
    "3d_envelope": {"label": "3D Euler envelope", "color": "tab:green", "linestyle": "-"},
    "first_order": {"label": "1st-order perturbation", "color": "tab:orange", "linestyle": "-"},
    "second_order": {"label": "2nd-order perturbation", "color": "tab:orange", "linestyle": "--"},
    "pic": {"label": "3D PIC", "color": "black", "linestyle": "-"},
    "opal": {"label": "OPALX reference particle", "color": "tab:red", "linestyle": ":"},
}


def load_series(filename):
    path = DATA_DIR / filename
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.dtype.names != COLUMN_NAMES:
        raise ValueError(f"Unexpected columns in {path}: {data.dtype.names}")
    return {name: np.atleast_1d(data[name]) for name in COLUMN_NAMES}


def load_opal_series(path=OPAL_STAT_PATH):
    column_names = []
    in_data = False
    column_definition = None
    data_rows = []
    column_pattern = re.compile(r"name\s*=\s*([^,\s]+)")

    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped.startswith("&column"):
                column_definition = stripped
            elif column_definition is not None:
                column_definition += " " + stripped
                if stripped.startswith("&end"):
                    match = column_pattern.search(column_definition)
                    column_definition = None
                    if not match:
                        continue
                    column_names.append(match.group(1))
            elif stripped.startswith("&data"):
                in_data = True
            elif in_data and stripped and not stripped.startswith("&"):
                values = stripped.split()
                if len(values) != len(column_names):
                    continue
                try:
                    data_rows.append([float(value) for value in values])
                except ValueError:
                    continue

    if not data_rows:
        raise ValueError(f"No numeric data found in {path}")

    data = dict(zip(column_names, np.asarray(data_rows).T))
    return {
        "time_s": data["t"] * 1e-9,
        "mean_x_m": data["mean_x"],
        "mean_y_m": data["mean_y"],
        "mean_z_m": data["s"],
        "rms_x_m": data["rms_x"],
        "rms_y_m": data["rms_y"],
        "rms_z_m": data["rms_s"],
        "mean_px_beta_gamma": data["ref_px"],
        "mean_py_beta_gamma": data["ref_py"],
        "mean_pz_beta_gamma": data["ref_pz"],
        "rms_px_beta_gamma": data["rms_px"],
        "rms_py_beta_gamma": data["rms_py"],
        "rms_pz_beta_gamma": data["rms_ps"],
    }


def limit_series_to_time(series, max_time_s):
    mask = series["time_s"] <= max_time_s
    return {name: values[mask] for name, values in series.items()}


def plot_series(ax, series, style, x_key, y_key, label=True):
    values = series[y_key]
    finite = np.isfinite(series[x_key]) & np.isfinite(values)
    if not np.any(finite):
        return
    ax.plot(
        series[x_key][finite] * 1e9,
        values[finite],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style.get("linewidth", 1.2),
        zorder=style.get("zorder", 2),
        label=style["label"] if label else None,
    )


def configure_longitudinal_axes(axes):
    axes[0, 0].set_ylabel("mean $z$ [m]")
    axes[0, 1].set_ylabel(r"mean $p_z/(m_ec)$ [$\beta\gamma$]")
    axes[1, 0].set_ylabel("RMS $z$ [m]")
    axes[1, 1].set_ylabel(r"RMS $p_z/(m_ec)$ [$\beta\gamma$]")
    for row in axes:
        for ax in row:
            ax.xaxis.set_major_locator(MaxNLocator(8))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_major_locator(MaxNLocator(7))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.8)
            ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.45)
            ax.set_xlabel("time [ns]")
def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def make_longitudinal_figure(series_by_model, filename, stem):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    quantities = (
        ("mean_z_m", axes[0, 0]),
        ("mean_pz_beta_gamma", axes[0, 1]),
        ("rms_z_m", axes[1, 0]),
        ("rms_pz_beta_gamma", axes[1, 1]),
    )
    # Draw PIC first so analytic curves, especially Liouville, remain visible.
    ordered_series = sorted(
        series_by_model,
        key=lambda item: 0 if item[0] == "pic" else 1,
    )
    for model_name, series in ordered_series:
        style = MODEL_STYLES[model_name]
        for index, (quantity, ax) in enumerate(quantities):
            plot_series(ax, series, style, "time_s", quantity, label=index == 0)
    configure_longitudinal_axes(axes)
    legend_handles = [
        Line2D(
            [], [],
            color=MODEL_STYLES[model_name]["color"],
            linestyle=MODEL_STYLES[model_name]["linestyle"],
            linewidth=MODEL_STYLES[model_name].get("linewidth", 1.2),
            label=MODEL_STYLES[model_name]["label"],
        )
        for model_name, _ in series_by_model
    ]
    axes[0, 0].legend(handles=legend_handles, frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, stem)


def make_transverse_figure(series_by_model):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    quantities = (("rms_x_m", axes[0]), ("rms_px_beta_gamma", axes[1]))
    for model_name, series in series_by_model:
        style = MODEL_STYLES[model_name]
        for index, (quantity, ax) in enumerate(quantities):
            plot_series(ax, series, style, "time_s", quantity, label=index == 0)
    axes[0].set_ylabel("RMS $x$ [m]")
    axes[1].set_ylabel(r"RMS $p_x/(m_ec)$ [$\beta\gamma$]")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(8))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_major_locator(MaxNLocator(7))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel("time [ns]")
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.8)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.45)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, "simple_transverse_rms")


def main():
    complex_pic = load_series("complex_3d_pic.csv")
    opal = limit_series_to_time(load_opal_series(), complex_pic["time_s"][-1])
    complex_series = [
        ("single_particle", load_series("complex_single_particle.csv")),
        ("liouville", load_series("complex_liouville.csv")),
        ("1d_envelope", load_series("complex_1d_envelope.csv")),
        ("pic", complex_pic),
        ("opal", opal),
    ]
    simple_series = [
        ("1d_envelope", load_series("simple_1d_envelope.csv")),
        ("3d_envelope", load_series("simple_3d_envelope.csv")),
        ("first_order", load_series("simple_1st_order_perturbation.csv")),
        ("second_order", load_series("simple_2nd_order_perturbation.csv")),
        ("pic", load_series("simple_3d_pic.csv")),
    ]

    make_longitudinal_figure(complex_series, "complex_moments.csv", "complex_longitudinal_rms")
    make_longitudinal_figure(simple_series, "simple_moments.csv", "simple_longitudinal_rms")
    make_transverse_figure(
        [
            simple_series[1],
            simple_series[2],
            simple_series[3],
            simple_series[4],
        ]
    )


if __name__ == "__main__":
    main()
