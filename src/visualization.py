from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

def plot_hand_on_axes(
    row: pd.Series,
    ax: plt.Axes,
    feature_cols: list[str],
    title: str | None = None
) -> None:
    """
    Plot one hand gesture (21 landmarks) on the given Matplotlib Axes.
    Expects flattened (x,y,z) features; uses x,y to draw the hand skeleton.
    """

    coords = row[feature_cols].values
    points = coords.reshape(-1, 3)  # (21, 3)
    xs, ys = points[:, 0], points[:, 1]

    # Hand skeleton connections (indices 0–20)
    HAND_CONNECTIONS = [
        (5,9), (9,13), (13,17),                # Palm
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    # Plot connections
    for i, j in HAND_CONNECTIONS:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color='steelblue', linewidth=2)

    # Plot keypoints
    ax.scatter(xs, ys, c='red', s=40)

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=10)


def plot_gesture_grid(
    df: pd.DataFrame,
    n_rows: int = 3,
    n_cols: int = 5,
    random_state: int | None = None
) -> None:
    """
    Visualize a random grid of hand gesture samples.

    This function randomly samples gestures from the input DataFrame and
    displays them in a grid layout using the hand skeleton visualization.
    Each subplot represents one gesture labeled with its class.
    """

    n_samples = n_rows * n_cols
    feature_cols = [col for col in df.columns if col != 'label']

    # Sample data
    sampled_df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten()

    # Plot each sample
    for i in range(n_samples):
        row = sampled_df.iloc[i]
        lbl = row['label']
        plot_hand_on_axes(row, axes[i], feature_cols, title=str(lbl))

    # Hide unused axes (safe guard)
    for j in range(n_samples, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Sample of Gestures', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()