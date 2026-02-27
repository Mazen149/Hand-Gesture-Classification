import pandas as pd
import numpy as np

def normalize_hand_xy(
    X: pd.DataFrame,
    wrist_col_idx: int = 1,
    mid_finger_tip_col_idx: int = 13,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Normalize hand landmark X/Y to be translation- and scale-invariant.

    This preprocessing addresses different hand positions and scales:
    - Recenter: subtract the wrist (landmark 0) so it becomes the origin.
    - Rescale: divide all X/Y landmarks by the distance from the wrist to the
      middle-finger tip (landmark 12).
    Z coordinates are not changed.
    """

    Xn = X.copy()

    x_cols = [f'x{i}' for i in range(1, 22)]
    y_cols = [f'y{i}' for i in range(1, 22)]

    wrist_x_col = f'x{wrist_col_idx}'
    wrist_y_col = f'y{wrist_col_idx}'
    mid_x_col = f'x{mid_finger_tip_col_idx}'
    mid_y_col = f'y{mid_finger_tip_col_idx}'

    # Recenter to wrist
    wrist_x = Xn[wrist_x_col].to_numpy()
    wrist_y = Xn[wrist_y_col].to_numpy()
    Xn.loc[:, x_cols] = Xn.loc[:, x_cols].sub(wrist_x, axis=0)
    Xn.loc[:, y_cols] = Xn.loc[:, y_cols].sub(wrist_y, axis=0)

    # Scale using distance to middle finger tip (after recentering)
    mid_x = Xn[mid_x_col].to_numpy()
    mid_y = Xn[mid_y_col].to_numpy()
    scale = np.sqrt(mid_x**2 + mid_y**2)
    scale = np.maximum(scale, eps)

    Xn.loc[:, x_cols] = Xn.loc[:, x_cols].div(scale, axis=0)
    Xn.loc[:, y_cols] = Xn.loc[:, y_cols].div(scale, axis=0)

    return Xn