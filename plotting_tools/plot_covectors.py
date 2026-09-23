import numpy as np
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from scipy.spatial import Delaunay


def plot_covector_in_chart(
    ax,
    base_point,
    components,
    n_lines=8,
    level_step=1.0,
    extent=None,
    color="crimson",
    linestyle="solid",
    linewidth=1.0,
    zero_linewidth=2.2,
    alpha=0.9,
    zorder=2,
    annotate_levels=True,
    level_fontsize=8,
    level_label_shift=0.0,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0),
):
    """
    Plot a covector in chart coordinates as its stack of level lines.

    A covector X_p acts linearly on tangent vectors, so its natural picture is
    NOT an arrow but the family of parallel lines

        { w : sum_i x_i * (w - X(p))^i = n * level_step },  n integer,

    drawn in the chart coordinates. Consecutive lines are separated by
    level_step / ||x_i||, so a "denser stack" means a "bigger" covector, and
    the action of the covector on a tangent vector is the (signed) number of
    lines pierced by the corresponding arrow.

    Args:
        ax: Matplotlib 2D axis.
        base_point: Array (2,) — the point p in chart coordinates.
        components: Array (2,) — components x_i of the covector in the dual
            chart-induced basis of the same chart.
        n_lines: Level lines are drawn for n = -n_lines, ..., n_lines
            (they are infinite lines, automatically clipped by the axes).
        level_step: Spacing, in units of the covector value, between drawn lines.
        extent: If None, the level lines are drawn as infinite lines (clipped by
            the axes). Otherwise, each level line is drawn as a segment of
            half-length `extent` (in chart units) centered on the normal ray
            through the base point, which gives a more local picture.
        color: Color of the stack.
        linestyle: Line style of the stack.
        linewidth: Line width of the non-zero level lines.
        zero_linewidth: Line width of the level line through p (value 0).
        alpha: Opacity of the lines.
        zorder: Drawing order.
        annotate_levels: If True, writes the value carried by each level line.
        level_fontsize: Font size of the level annotations.
        level_label_shift: Shift of the level annotations along the level lines
            (in chart units), useful to avoid overlapping stacks.
        label: Optional LaTeX string for the covector (e.g. r"$\\mathsf{X}$").
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy) offset of the label from the base point.

    Returns:
        Float — the distance between consecutive level lines in chart coordinates.
    """
    base_point = np.asarray(base_point, dtype=float).reshape(2)
    components = np.asarray(components, dtype=float).reshape(2)

    norm = float(np.linalg.norm(components))
    if norm == 0.0:
        raise ValueError("The zero covector has no level-line representation.")

    # Unit normal (direction of increasing value) and direction of the level lines
    normal = components / norm
    tangent = np.array([-normal[1], normal[0]])
    spacing = level_step / norm

    for n in range(-n_lines, n_lines + 1):
        foot = base_point + n * spacing * normal
        line_width = zero_linewidth if n == 0 else linewidth

        if extent is None:
            ax.axline(
                tuple(foot),
                tuple(foot + tangent),
                color=color,
                linestyle=linestyle,
                linewidth=line_width,
                alpha=alpha,
                zorder=zorder,
            )
        else:
            end_points = np.stack([foot - extent * tangent, foot + extent * tangent])
            ax.plot(
                end_points[:, 0], end_points[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=line_width,
                alpha=alpha,
                zorder=zorder,
            )

        if annotate_levels and n != 0:
            text_position = foot + level_label_shift * tangent
            ax.text(
                text_position[0], text_position[1],
                f"{n * level_step:g}",
                fontsize=level_fontsize,
                color=color,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                zorder=zorder + 1,
                clip_on=True,
            )

    if label is not None:
        ax.text(
            base_point[0] + label_offset[0],
            base_point[1] + label_offset[1],
            label,
            fontsize=label_fontsize,
            color=color,
            ha="center", va="center",
            zorder=zorder + 1,
            clip_on=True,
        )

    return spacing


def plot_vector_in_chart(
    ax,
    base_point,
    components,
    scale=1.0,
    color="black",
    width=0.008,
    alpha=1.0,
    zorder=4,
    draw_point=True,
    point_color="black",
    point_size=40,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0),
):
    """
    Plot a single tangent vector in chart coordinates, at true scale.

    The arrow is drawn with its actual components (scale=1.0 by default), which
    is what makes the "number of level lines pierced" reading of a covector
    meaningful.

    Args:
        ax: Matplotlib 2D axis.
        base_point: Array (2,) — the point p in chart coordinates.
        components: Array (2,) — components v^i in the chart-induced basis.
        scale: Multiplicative factor applied to the components (1.0 = true scale).
        color: Arrow color.
        width: Shaft width of the arrow (in axes units, as in Matplotlib quiver).
        alpha: Opacity.
        zorder: Drawing order.
        draw_point: If True, marks the base point.
        point_color: Color of the base point marker.
        point_size: Size of the base point marker.
        label: Optional LaTeX string (e.g. r"$\\mathcal{X}$").
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy) offset of the label from the arrow tip.
    """
    base_point = np.asarray(base_point, dtype=float).reshape(2)
    components = np.asarray(components, dtype=float).reshape(2) * scale

    ax.quiver(
        base_point[0], base_point[1],
        components[0], components[1],
        angles="xy", scale_units="xy", scale=1.0,
        color=color, width=width, alpha=alpha, zorder=zorder,
    )

    if draw_point:
        ax.scatter(
            base_point[0], base_point[1],
            color=point_color, s=point_size, zorder=zorder + 1,
        )

    if label is not None:
        tip = base_point + components
        ax.text(
            tip[0] + label_offset[0], tip[1] + label_offset[1],
            label,
            fontsize=label_fontsize,
            color=color,
            ha="center", va="center",
            zorder=zorder + 1,
            clip_on=True,
        )


def plot_covector_on_manifold(
    ax,
    manifold,
    chart,
    param_point,
    components,
    n_lines=3,
    level_step=1.0,
    extent=1.0,
    color="crimson",
    linestyle="solid",
    linewidth=1.2,
    zero_linewidth=2.4,
    alpha=0.9,
    zorder=5,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0, 0.0),
):
    """
    Plot a covector at p as its stack of level lines drawn *inside the tangent
    plane* T_pM, embedded in the ambient space.

    This is the chart-free picture of a covector: the kernel line through p
    (value 0) and its parallel translates, equally spaced by level_step. The
    result does not depend on which chart is used to supply the components,
    since the pushforward of the chart-induced basis compensates the change of
    components (a good consistency check: call it with the components and the
    chart (U, X), then with (V, Y), and the drawn lines coincide).

    Only implemented for 2-dimensional manifolds (the level sets of a covector
    are then lines inside the 2-dimensional tangent plane).

    Args:
        ax: Matplotlib 3D axis.
        manifold: Manifold object.
        chart: Chart whose induced basis the components refer to.
        param_point: Array (2,) — the point p in parameter space.
        components: Array (2,) — components x_i in the dual basis of `chart`.
        n_lines: Level lines are drawn for n = -n_lines, ..., n_lines.
        level_step: Spacing, in units of the covector value, between drawn lines.
        extent: Half-length of the drawn segments, measured in ambient units.
        color: Color of the stack.
        linestyle: Line style of the stack.
        linewidth: Line width of the non-zero level lines.
        zero_linewidth: Line width of the level line through p (value 0).
        alpha: Opacity.
        zorder: Drawing order.
        label: Optional LaTeX string for the covector.
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy, dz) offset of the label in ambient space.

    Returns:
        Array (ambient_dim, 2) — the pushforward of the chart-induced basis at p,
        i.e. the columns used to embed the tangent plane in ambient space.
    """
    if manifold.dim != 2:
        raise NotImplementedError("plot_covector_on_manifold requires dim = 2.")

    param_point = jnp.asarray(param_point).reshape(manifold.dim)
    components = jnp.asarray(components).reshape(manifold.dim)

    # === Pushforward of the chart-induced basis: D(Phi o X^{-1}) at X(p) ===
    jacobian_embedding = jax.jacrev(manifold.embedding_func)(param_point)   # (ambient, dim)
    jacobian_chart = jax.jacrev(chart.chart_map)(param_point)               # (dim, dim)
    pushforward = jacobian_embedding @ jnp.linalg.inv(jacobian_chart)       # (ambient, dim)

    # === Level lines of the linear form, in chart components ===
    squared_norm = float(jnp.dot(components, components))
    if squared_norm == 0.0:
        raise ValueError("The zero covector has no level-line representation.")

    unit_value = np.asarray(components) / squared_norm            # <x, unit_value> = 1
    kernel_direction = np.array([-components[1], components[0]])  # <x, kernel> = 0

    base_ambient = np.asarray(manifold.embedding_func(param_point))
    pushforward = np.asarray(pushforward)

    # normalize the kernel direction so that `extent` is measured in ambient units
    kernel_direction = kernel_direction / np.linalg.norm(pushforward @ kernel_direction)

    for n in range(-n_lines, n_lines + 1):
        foot = n * level_step * unit_value
        end_points = np.stack([
            foot - extent * kernel_direction,
            foot + extent * kernel_direction,
        ])                                                        # (2, dim)
        embedded = base_ambient[None, :] + end_points @ pushforward.T

        ax.plot(
            embedded[:, 0], embedded[:, 1], embedded[:, 2],
            color=color,
            linestyle=linestyle,
            linewidth=zero_linewidth if n == 0 else linewidth,
            alpha=alpha,
            zorder=zorder,
        )

    if label is not None:
        tip = base_ambient + (n_lines * level_step * unit_value) @ pushforward.T
        ax.text(
            tip[0] + label_offset[0], tip[1] + label_offset[1], tip[2] + label_offset[2],
            label, fontsize=label_fontsize, color=color,
        )

    return pushforward


def plot_function_level_sets_in_chart(
    ax,
    chart,
    function,
    param_space_data,
    levels=12,
    simplices=None,
    max_edge=None,
    color="black",
    linestyles="solid",
    linewidths=0.8,
    alpha=0.6,
    zorder=1,
    label_levels=False,
    label_fontsize=7,
):
    """
    Plot the level sets of a scalar field in chart coordinates.

    The triangulation is built in parameter space (where the sampling is
    well behaved) and then mapped to chart coordinates, exactly as done in
    plot_functions.plot_parametric_function_in_chart_coordinates. Triangles
    that become degenerate or extremely stretched under the chart map (e.g.
    across the branch cut of a polar chart) are masked out.

    Args:
        ax: Matplotlib 2D axis.
        chart: Chart object.
        function: Function object (scalar field) on the same manifold.
        param_space_data: Array (N, 2) — sample points in parameter space.
        levels: Number of level sets, or explicit list of levels.
        simplices: Optional Delaunay simplices on param_space_data.
        max_edge: Chart-space edge length above which a triangle is masked.
            If None, it is set to 10 times the median edge length.
        color: Color of the level lines.
        linestyles: Line style of the level lines.
        linewidths: Line width of the level lines.
        alpha: Opacity.
        zorder: Drawing order.
        label_levels: If True, adds inline labels with the level values.
        label_fontsize: Font size of the inline labels.

    Returns:
        The Matplotlib TriContourSet.
    """
    if chart.manifold != function.manifold:
        raise ValueError("Chart and function must be defined on the same manifold.")

    param_space_data = np.asarray(param_space_data, dtype=float)
    values = np.asarray(function.evaluate_in_param_space(jnp.array(param_space_data)))
    chart_points = np.asarray(chart.map_to_chart(jnp.array(param_space_data)))

    if simplices is None:
        simplices = Delaunay(param_space_data).simplices

    # Replace non-finite chart coordinates (they are masked out below anyway)
    finite_points = np.isfinite(chart_points).all(axis=1)
    chart_points = np.where(finite_points[:, None], chart_points, 0.0)

    triangulation = mtri.Triangulation(
        chart_points[:, 0], chart_points[:, 1], triangles=simplices
    )

    # === Mask degenerate / stretched triangles ===
    triangle_points = chart_points[simplices]                       # (M, 3, 2)
    edges = np.stack([
        triangle_points[:, 1] - triangle_points[:, 0],
        triangle_points[:, 2] - triangle_points[:, 1],
        triangle_points[:, 0] - triangle_points[:, 2],
    ], axis=1)                                                      # (M, 3, 2)
    edge_lengths = np.linalg.norm(edges, axis=-1).max(axis=1)       # (M,)

    if max_edge is None:
        max_edge = 10.0 * np.median(edge_lengths)

    mask = (edge_lengths > max_edge) | (~finite_points[simplices].all(axis=1))
    triangulation.set_mask(mask)

    contour_set = ax.tricontour(
        triangulation, values,
        levels=levels,
        colors=color,
        linestyles=linestyles,
        linewidths=linewidths,
        alpha=alpha,
        zorder=zorder,
    )

    if label_levels:
        ax.clabel(contour_set, inline=True, fontsize=label_fontsize, fmt="%.1f")

    return contour_set


def plot_function_level_sets_on_manifold(
    ax,
    manifold,
    function,
    levels=12,
    resolution=200,
    xmin=-jnp.pi, xmax=jnp.pi,
    ymin=-jnp.pi, ymax=jnp.pi,
    color="black",
    linestyle="solid",
    linewidth=1.0,
    alpha=0.8,
    zorder=3,
):
    """
    Plot the level sets of a scalar field directly on the embedded manifold.

    The level sets are computed in parameter space with Matplotlib's contour
    algorithm (on an auxiliary figure that is immediately closed), and each
    resulting polyline is then embedded in the ambient space.

    Args:
        ax: Matplotlib 3D axis.
        manifold: Manifold object with embed().
        function: Function object (scalar field) on the same manifold.
        levels: Number of level sets, or explicit list of levels.
        resolution: Grid resolution used in parameter space.
        xmin, xmax, ymin, ymax: Bounds of the parameter-space grid.
        color: Color of the level lines.
        linestyle: Line style of the level lines.
        linewidth: Line width of the level lines.
        alpha: Opacity.
        zorder: Drawing order.

    Returns:
        Array with the level values that were drawn.
    """
    x_values = jnp.linspace(xmin, xmax, resolution)
    y_values = jnp.linspace(ymin, ymax, resolution)
    XX, YY = jnp.meshgrid(x_values, y_values)
    param_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    F_values = np.asarray(function.evaluate_in_param_space(param_points)).reshape(XX.shape)

    # === Extract the level sets in parameter space ===
    auxiliary_figure, auxiliary_ax = plt.subplots()
    contour_set = auxiliary_ax.contour(np.asarray(XX), np.asarray(YY), F_values, levels=levels)
    level_values = np.asarray(contour_set.levels)
    all_segments = contour_set.allsegs
    plt.close(auxiliary_figure)

    # === Embed each polyline and draw it on the manifold ===
    for segments in all_segments:
        for segment in segments:
            if len(segment) < 2:
                continue
            embedded = np.asarray(manifold.embed(jnp.array(segment)))
            ax.plot(
                embedded[:, 0], embedded[:, 1], embedded[:, 2],
                color=color, linestyle=linestyle, linewidth=linewidth,
                alpha=alpha, zorder=zorder,
            )

    return level_values
