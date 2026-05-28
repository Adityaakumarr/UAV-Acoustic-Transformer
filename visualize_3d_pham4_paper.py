"""
visualize_3d_pham4_paper.py
───────────────────────────
Geospatial 3D Visualization script matching the EXACT format of the DJI Phantom 4
drone trajectory from the ICASSP 2026 paper.

Generates:
  1. Static High-Resolution Plot -> outputs/trajectory_3d_pham4_final.png (Matplotlib)
  2. Interactive 3D Dashboard    -> outputs/trajectory_3d_interact_pham4.html (Plotly)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go


def main():
    # ── 1. Setup & Load Data ──────────────────────────────────────────────────
    csv_path = "outputs/predictions_full.csv"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Predictions file not found at '{csv_path}'. "
            "Please run 'python evaluate.py' first."
        )

    print(f"Loading predictions from '{csv_path}'...")
    df = pd.read_csv(csv_path)

    # Extract coordinates
    true_x = df["true_x"].values
    true_y = df["true_y"].values
    true_z = df["true_z"].values

    pred_x = df["pred_x"].values
    pred_y = df["pred_y"].values
    pred_z = df["pred_z"].values

    # ── 2. Static Publication-Grade Matplotlib Plot ───────────────────────────
    print("Generating static Matplotlib 3D plot matching paper layout...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # A. Render continuous solid Ground Truth line in pure blue
    ax.plot(
        true_x,
        true_y,
        true_z,
        color="blue",
        linestyle="-",
        linewidth=1.8,
        zorder=5,  # Render underneath
    )

    # B. Render dense predicted path as red dots with no connecting lines
    ax.plot(
        pred_x,
        pred_y,
        pred_z,
        color="red",
        linestyle="None",
        marker="o",
        markersize=1.8,
        zorder=10,  # Render on top
    )

    # C. Limits & Box Bounds
    ax.set_xlim([-1.5, 5.5])
    ax.set_ylim([-2.0, 16.0])
    ax.set_zlim([0.0, 20.0])

    # D. Ticks
    ax.set_xticks(np.arange(-1.0, 6.0, 1.0))
    ax.set_yticks(np.arange(-2.0, 16.1, 2.0))
    z_ticks = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    ax.set_zticks(z_ticks)

    # E. Axis Labels & Aspect Ratio
    ax.set_xlabel("X (m)", fontsize=10, labelpad=10)
    ax.set_ylabel("Y (m)", fontsize=10, labelpad=10)
    ax.set_zlabel("Z (m)", fontsize=10, labelpad=10)
    ax.set_box_aspect([1, 1, 1])  # Equal scale aspect ratio

    # F. Styling: Clean White Grid, No Grey Panes, No Border Edges
    ax.set_facecolor("white")
    
    # Hide the background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Hide pane borders to match the clean white borderless style of the paper
    ax.xaxis.pane.set_edgecolor("white")
    ax.yaxis.pane.set_edgecolor("white")
    ax.zaxis.pane.set_edgecolor("white")

    # Set light gray grid lines
    ax.grid(True, color="lightgray", linestyle="-", linewidth=0.5)

    # Set exact elevation and azimuth viewing angle to match the perspective in the paper
    ax.view_init(elev=20, azim=-60)

    # G. Typography: Place bold centered "Pham4" text directly below the canvas
    plt.figtext(
        0.5,
        0.04,
        "Pham4",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        fontfamily="sans-serif",
    )

    # Save PNG
    static_png = os.path.join(out_dir, "trajectory_3d_pham4_final.png")
    plt.savefig(static_png, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"SUCCESS: Static high-resolution 3D plot saved to '{static_png}'")

    # ── 3. Interactive Plotly 3D HTML Dashboard ──────────────────────────────
    print("Generating interactive Plotly 3D HTML dashboard...")
    fig_plotly = go.Figure()

    # A. Ground Truth Trace (Added first to be rendered underneath)
    fig_plotly.add_trace(
        go.Scatter3d(
            x=true_x,
            y=true_y,
            z=true_z,
            mode="lines",
            line=dict(color="blue", width=4),
            showlegend=False,
        )
    )

    # B. Predicted Trace (Added second to be rendered on top)
    fig_plotly.add_trace(
        go.Scatter3d(
            x=pred_x,
            y=pred_y,
            z=pred_z,
            mode="markers",
            marker=dict(size=2.2, color="red", symbol="circle", opacity=1.0),
            showlegend=False,
        )
    )

    # C. Formatting & Layout Settings
    fig_plotly.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-1.5, 5.5],
                title="X (m)",
                gridcolor="lightgray",
                backgroundcolor="white",
                showbackground=False,
                zeroline=False,
            ),
            yaxis=dict(
                range=[-2.0, 16.0],
                title="Y (m)",
                gridcolor="lightgray",
                backgroundcolor="white",
                showbackground=False,
                zeroline=False,
            ),
            zaxis=dict(
                range=[0.0, 20.0],
                title="Z (m)",
                gridcolor="lightgray",
                backgroundcolor="white",
                showbackground=False,
                zeroline=False,
                tickvals=z_ticks,
            ),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),  # Match ax.set_box_aspect([1, 1, 1])
        ),
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=80),
        # Place "Pham4" bold text at the center below the plot
        title=dict(
            text="<b>Pham4</b>",
            x=0.5,
            y=0.03,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=20, color="black", family="Arial"),
        ),
        showlegend=False,
    )

    # Save HTML
    interact_html = os.path.join(out_dir, "trajectory_3d_interact_pham4.html")
    fig_plotly.write_html(interact_html)
    print(f"SUCCESS: Interactive Plotly 3D HTML saved to '{interact_html}'")


if __name__ == "__main__":
    main()
