import jax
import jax.numpy as jnp
from jax import jacfwd
from jax import vmap
from mpl_toolkits.mplot3d import Axes3D

from plotting_tools.basic_plotting_tools import Arrow3D

def plot_curve_on_manifold(
    ax,
    curve,
    lambda_range=(-jnp.pi, jnp.pi),
    n_points=300,
    color="black",
    linewidth=2.0,
    label=None,
    label_position="center",  # "center", "start", "end"
    label_fontsize=14,
    label_offset=(0, 0, 0.2)
):
    """
    Plot a parametrized curve on the manifold with optional tangent arrows and label.

    Args:
        ax: Matplotlib 3D axis.
        curve: A Curve object.
        lambda_range: Tuple (lambda_min, lambda_max) for curve parameter range.
        n_points: Number of sampling points along the curve.
        color: Color of the curve and tangents.
        linewidth: Width of the curve line.
        show_tangents: Whether to plot tangent arrows.
        tangent_step: Plot an arrow every 'tangent_step' points.
        tangent_scale: Scale factor for arrow length.
        tangent_style: Dict of arrow style kwargs (merged with defaults).
        tangent_method: "finite_difference" (default) or "autodiff".
        label: LaTeX string label for the curve (e.g. r"$\gamma$") or None.
        label_position: "center", "start", or "end".
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy, dz) offset for label position.
    """
    # === Sample curve ===
    lambda_vals = jnp.linspace(lambda_range[0], lambda_range[1], n_points)
    points = curve.evaluate_on_manifold(lambda_vals)

    # === Plot the curve ===
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=color,
        linewidth=linewidth
    )

    # === Add label ===
    if label is not None:
        if label_position == "center":
            idx = len(points) // 2
        elif label_position == "start":
            idx = 0
        elif label_position == "end":
            idx = -1
        else:
            raise ValueError(f"Invalid label_position: {label_position}")

        p_label = points[idx]
        dx, dy, dz = label_offset

        ax.text(
            p_label[0] + dx,
            p_label[1] + dy,
            p_label[2] + dz,
            label,
            fontsize=label_fontsize,
            color=color
        )


def plot_curve_in_chart(
    ax,
    curve,
    chart,
    lambda_range=(-jnp.pi, jnp.pi),
    n_points=300,
    color="black",
    linewidth=2.0,
    label=None,
    label_position="center",  # "center", "start", "end"
    label_fontsize=14,
    label_offset=(0, 0)
):
    """
    Plot a parametrized curve in chart coordinates (i.e., X(gamma(lambda))).

    Args:
        ax: Matplotlib 2D axis.
        curve: A Curve object.
        chart: A Chart object for coordinate transformation.
        lambda_range: Tuple (lambda_min, lambda_max) for curve parameter range.
        n_points: Number of sampling points along the curve.
        color: Color of the curve.
        linewidth: Width of the curve line.
        label: Optional LaTeX string label (e.g. r"$\gamma$").
        label_position: "center", "start", or "end".
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy) offset for label position.
    """
    # === Sample the curve in chart ===
    lambda_vals = jnp.linspace(lambda_range[0], lambda_range[1], n_points)
    coords = curve.evaluate_in_chart(chart, lambda_vals)

    if coords.shape[-1] != 2:
        raise ValueError(f"plot_curve_in_chart expects 2D chart coordinates, got shape {coords.shape}")

    # === Plot the curve in chart coordinates ===
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=color,
        linewidth=linewidth
    )

    # === Add label ===
    if label is not None:
        if label_position == "center":
            idx = len(coords) // 2
        elif label_position == "start":
            idx = 0
        elif label_position == "end":
            idx = -1
        else:
            raise ValueError(f"Invalid label_position: {label_position}")

        x_label, y_label = coords[idx]
        dx, dy = label_offset

        ax.text(
            x_label + dx,
            y_label + dy,
            label,
            fontsize=label_fontsize,
            color=color,
            ha="center", va="center"
        )


def plot_curve_tangents_in_chart(
    ax,
    curve,
    chart,
    lambda_range=(0, 2 * jnp.pi),
    n_points=500,
    n_arrows=30,
    color="crimson",
    scale=20,
    width=0.005,
    label=r"Tangent",
    label_fontsize=12,
    label_offset=(0.0, 0.0)
):
    """
    Plot tangent vectors of a curve in chart coordinates.

    Args:
        ax: Matplotlib 2D axis.
        curve: A Curve object.
        chart: Chart object.
        lambda_range: Tuple (min, max) defining sampling interval.
        n_points: Number of lambda values for full resolution.
        n_arrows: Number of tangent vectors (quiver) to show.
        color: Color of the tangent arrows.
        scale: Quiver arrow scale (higher = shorter arrows).
        width: Width of arrows.
        label: Optional label for legend.
        label_fontsize: Font size of label text.
        label_offset: Tuple for shifting label from arrow base.
    """
    lambdas = jnp.linspace(lambda_range[0], lambda_range[1], n_points)
    coords = curve.evaluate_in_chart(chart, lambdas)
    tangents = curve.derivative_in_chart(chart, lambdas)

    # Subsample
    stride = max(1, len(lambdas) // n_arrows)
    sub_coords = coords[::stride]
    sub_tangents = tangents[::stride]

    # Normalize tangent vectors
    norms = jnp.linalg.norm(sub_tangents, axis=1, keepdims=True)
    normed_tangents = sub_tangents / norms

    # Plot the quiver (tangent arrows)
    q = ax.quiver(
        sub_coords[:, 0], sub_coords[:, 1],
        normed_tangents[:, 0], normed_tangents[:, 1],
        angles="xy",
        color=color, scale=scale, width=width
    )

    # Add optional label at the middle arrow
    if label is not None and len(sub_coords) > 0:
        mid_idx = len(sub_coords) // 2
        x0, y0 = sub_coords[mid_idx]
        dx, dy = label_offset
        ax.text(
            x0 + dx, y0 + dy,
            label,
            fontsize=label_fontsize,
            color=color,
            ha="center",
            va="center"
        )


def plot_curve_tangents_on_manifold(
    ax,
    curve,
    lambda_range=(0, 2 * jnp.pi),
    n_points=500,
    n_arrows=30,
    color="crimson",
    scale=0.3,
    linewidth=1.5,
    arrowstyle="-|>",
    mutation_scale=15,
    alpha=1.0,
    label=r"Tangent",
    label_fontsize=12,
    label_offset=(0.0, 0.0, 0.0)
):
    """
    Plot tangent vectors of a curve directly on the manifold (ambient space).

    Args:
        ax: Matplotlib 3D axis.
        curve: A Curve object.
        lambda_range: Tuple (min, max) defining sampling interval.
        n_points: Number of lambda values for full resolution.
        n_arrows: Number of tangent vectors (quiver) to show.
        color: Color of the tangent arrows.
        scale: Scaling factor for arrow lengths.
        linewidth: Arrow line width.
        arrowstyle: Style for 3D arrow heads.
        mutation_scale: Arrowhead size.
        alpha: Opacity of arrows.
        label: Optional label for legend.
        label_fontsize: Font size of label text.
        label_offset: Tuple (dx, dy, dz) to offset label position.
    """
    # === Compute tangent vectors using Curve's method ===
    sub_positions, normed_tangents = curve.compute_tangent_vectors_on_manifold(
        lambda_range=lambda_range,
        n_points=n_points,
        n_arrows=n_arrows
    )

    # === Plot arrows using Arrow3D ===
    for pos, vec in zip(sub_positions, normed_tangents):
        tip = pos + scale * vec
        arrow = Arrow3D(
            [pos[0], tip[0]],
            [pos[1], tip[1]],
            [pos[2], tip[2]],
            mutation_scale=mutation_scale,
            lw=linewidth,
            arrowstyle=arrowstyle,
            color=color,
            alpha=alpha
        )
        ax.add_artist(arrow)

    # === Optional label at midpoint ===
    if label is not None and len(sub_positions) > 0:
        mid_idx = len(sub_positions) // 2
        base = sub_positions[mid_idx]
        dx, dy, dz = label_offset
        label_pos = base + jnp.array([dx, dy, dz])
        ax.text(
            label_pos[0], label_pos[1], label_pos[2],
            label,
            fontsize=label_fontsize,
            color=color,
            ha="center",
            va="center"
        )
