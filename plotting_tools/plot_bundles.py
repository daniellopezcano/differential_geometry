import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import proj3d

from plotting_tools.basic_plotting_tools import Arrow3D


# ==========================================================
# === Total space, fibres and sections (3D panel)
# ==========================================================

def plot_zero_section(
    ax,
    bundle,
    param_range,
    n_points=400,
    color="rebeccapurple",
    linewidth=3.0,
    alpha=1.0,
    zorder=5,
    label=None,
    label_fontsize=14,
    label_position=0.5,
    label_offset=(0.0, 0.0, 0.0),
):
    """
    Plot the zero section of the bundle, i.e. the copy of the base manifold sitting
    inside the total space.

    Args:
        ax: Matplotlib 3D axis.
        bundle: TangentBundle object.
        param_range: Tuple (t_min, t_max) of the base parameter.
        n_points: Number of sampling points.
        color, linewidth, alpha, zorder: Line styling.
        label: Optional LaTeX label (e.g. the triplet of the base manifold).
        label_fontsize: Font size of the label.
        label_position: Position of the label along the curve, in [0, 1].
        label_offset: Tuple (dx, dy, dz) offset of the label in ambient space.
    """
    param_points = jnp.linspace(param_range[0], param_range[1], n_points).reshape(-1, 1)
    embedded = np.asarray(bundle.embed(bundle.zero_section(param_points)))

    ax.plot(
        embedded[:, 0], embedded[:, 1], embedded[:, 2],
        color=color, linewidth=linewidth, alpha=alpha, zorder=zorder,
    )

    if label is not None:
        index = int(label_position * (n_points - 1))
        position = embedded[index] + np.asarray(label_offset)
        ax.text(position[0], position[1], position[2], label,
                fontsize=label_fontsize, color=color, zorder=zorder + 1)


def plot_fibre(
    ax,
    bundle,
    param_point,
    fibre_range,
    n_points=60,
    color="green",
    linewidth=1.5,
    linestyle="solid",
    alpha=1.0,
    zorder=6,
    arrowheads=True,
    mutation_scale=12,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0, 0.0),
):
    """
    Plot the fibre T_pM = preim_pi({p}) over a base point, as drawn in the total space.

    Args:
        ax: Matplotlib 3D axis.
        bundle: TangentBundle object.
        param_point: Array (base_dim,) — the base point in parameter space.
        fibre_range: Tuple (u_min, u_max) of fibre components to draw.
        n_points: Number of sampling points along the fibre.
        color, linewidth, linestyle, alpha, zorder: Line styling.
        arrowheads: If True, adds an arrowhead at each end (the fibre is a full copy of R).
        mutation_scale: Arrowhead size.
        label: Optional LaTeX label (e.g. r"$T_p\\mathcal{M}$").
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy, dz) offset of the label in ambient space.
    """
    fibre_values = jnp.linspace(fibre_range[0], fibre_range[1], n_points)
    embedded = np.asarray(bundle.embed(bundle.fibre(param_point, fibre_values)))

    ax.plot(
        embedded[:, 0], embedded[:, 1], embedded[:, 2],
        color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder,
    )

    if arrowheads:
        for start, end in [(embedded[-2], embedded[-1]), (embedded[1], embedded[0])]:
            ax.add_artist(Arrow3D(
                [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                mutation_scale=mutation_scale, lw=linewidth, arrowstyle="-|>",
                color=color, alpha=alpha,
            ))

    if label is not None:
        position = embedded[-1] + np.asarray(label_offset)
        ax.text(position[0], position[1], position[2], label,
                fontsize=label_fontsize, color=color, zorder=zorder + 1)

    return embedded


def plot_preimage_of_chart_domain(
    ax,
    bundle,
    chart,
    fibre_range,
    resolution=(80, 2),
    color="0.4",
    alpha=0.25,
    boundary_color="red",
    boundary_linestyle="dashed",
    boundary_linewidth=1.2,
    zorder=3,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0, 0.0),
):
    """
    Plot  preim_pi(U) = TU,  the part of the total space sitting over a chart domain,
    as a shaded patch with a dashed boundary.

    Args:
        ax: Matplotlib 3D axis.
        bundle: TangentBundle object.
        chart: Chart object on the base manifold (1-dimensional base: it uses
            chart.boundary_interval).
        fibre_range: Tuple (u_min, u_max) of fibre components to draw.
        resolution: Tuple (n_base, n_fibre) of grid points.
        color, alpha: Styling of the shaded patch.
        boundary_color, boundary_linestyle, boundary_linewidth: Styling of the boundary.
        zorder: Drawing order.
        label: Optional LaTeX label (e.g. r"$T\\mathcal{U}$").
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy, dz) offset of the label in ambient space.
    """
    if bundle.base_dim != 1:
        raise NotImplementedError("Only implemented for a 1-dimensional base manifold.")

    t_min, t_max = chart.boundary_interval
    base_values = jnp.linspace(t_min, t_max, resolution[0])
    fibre_values = jnp.linspace(fibre_range[0], fibre_range[1], resolution[1])

    base_grid, fibre_grid = jnp.meshgrid(base_values, fibre_values)
    bundle_grid = jnp.stack([base_grid.ravel(), fibre_grid.ravel()], axis=-1)
    embedded = np.asarray(bundle.embed(bundle_grid))

    X = embedded[:, 0].reshape(base_grid.shape)
    Y = embedded[:, 1].reshape(base_grid.shape)
    Z = embedded[:, 2].reshape(base_grid.shape)

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor="none",
                    shade=False, zorder=zorder)

    # === Boundary: the two fibres over the endpoints and the two extreme sections ===
    for boundary_points in [
        bundle.fibre(jnp.array([t_min]), fibre_values),
        bundle.fibre(jnp.array([t_max]), fibre_values),
        bundle.bundle_points(base_values.reshape(-1, 1),
                             jnp.full((resolution[0], 1), fibre_range[0])),
        bundle.bundle_points(base_values.reshape(-1, 1),
                             jnp.full((resolution[0], 1), fibre_range[1])),
    ]:
        boundary = np.asarray(bundle.embed(boundary_points))
        ax.plot(boundary[:, 0], boundary[:, 1], boundary[:, 2],
                color=boundary_color, linestyle=boundary_linestyle,
                linewidth=boundary_linewidth, zorder=zorder + 1)

    if label is not None:
        corner = np.asarray(bundle.embed(
            bundle.bundle_points(jnp.array([[t_max]]), jnp.array([[fibre_range[1]]]))
        ))[0]
        position = corner + np.asarray(label_offset)
        ax.text(position[0], position[1], position[2], label,
                fontsize=label_fontsize, color=boundary_color, zorder=zorder + 2)


def plot_projection_arrow(
    ax,
    bundle,
    param_point,
    from_value,
    to_value=0.0,
    color="green",
    linewidth=1.8,
    mutation_scale=15,
    arrowstyle="-|>",
    alpha=1.0,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0, 0.0),
):
    """
    Plot the action of the projection map pi on one point of the total space: an arrow
    from (p ; from_value) down to the base point (p ; to_value).

    Args:
        ax: Matplotlib 3D axis.
        bundle: TangentBundle object.
        param_point: Array (base_dim,) — the base point in parameter space.
        from_value: Fibre component of the point being projected.
        to_value: Fibre component of the arrow tip (0 = the zero section).
        color, linewidth, mutation_scale, arrowstyle, alpha: Arrow styling.
        label: Optional LaTeX label (e.g. r"$\\pi$").
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy, dz) offset of the label in ambient space.
    """
    start = np.asarray(bundle.embed(bundle.fibre(param_point, jnp.array([from_value]))))[0]
    end = np.asarray(bundle.embed(bundle.fibre(param_point, jnp.array([to_value]))))[0]

    ax.add_artist(Arrow3D(
        [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
        mutation_scale=mutation_scale, lw=linewidth, arrowstyle=arrowstyle,
        color=color, alpha=alpha,
    ))

    if label is not None:
        position = 0.5 * (start + end) + np.asarray(label_offset)
        ax.text(position[0], position[1], position[2], label,
                fontsize=label_fontsize, color=color)


def plot_section_on_total_space(
    ax,
    bundle,
    field,
    param_range,
    n_points=400,
    color="magenta",
    linewidth=2.0,
    alpha=1.0,
    zorder=7,
    label=None,
    label_fontsize=16,
    label_position=0.5,
    label_offset=(0.0, 0.0, 0.0),
):
    """
    Plot a vector field as a section of the bundle: the curve traced inside the total
    space by p -> (p ; X(p)).

    Args:
        ax: Matplotlib 3D axis.
        bundle: TangentBundle object.
        field: VectorField object on the base manifold.
        param_range: Tuple (t_min, t_max) of the base parameter.
        n_points: Number of sampling points.
        color, linewidth, alpha, zorder: Line styling.
        label: Optional LaTeX label (e.g. r"$\\mathcal{X}$").
        label_fontsize: Font size of the label.
        label_position: Position of the label along the curve, in [0, 1].
        label_offset: Tuple (dx, dy, dz) offset of the label in ambient space.
    """
    param_points = jnp.linspace(param_range[0], param_range[1], n_points).reshape(-1, 1)
    embedded = np.asarray(bundle.embed(bundle.section_of_field(field, param_points)))

    ax.plot(
        embedded[:, 0], embedded[:, 1], embedded[:, 2],
        color=color, linewidth=linewidth, alpha=alpha, zorder=zorder,
    )

    if label is not None:
        index = int(label_position * (n_points - 1))
        position = embedded[index] + np.asarray(label_offset)
        ax.text(position[0], position[1], position[2], label,
                fontsize=label_fontsize, color=color, zorder=zorder + 1)

    return embedded


def plot_vector_in_fibre(
    ax,
    bundle,
    param_point,
    fibre_component,
    color="blue",
    linewidth=2.5,
    mutation_scale=18,
    arrowstyle="-|>",
    alpha=1.0,
    zorder=8,
    draw_base_point=True,
    base_point_color="green",
    base_point_size=45,
    draw_tip_marker=True,
    tip_marker="*",
    tip_marker_size=160,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0, 0.0),
):
    """
    Plot one tangent vector inside its fibre: an arrow from the base point (p ; 0) to
    the point (p ; X(p)) of the total space, with the base point and the tip marked.

    Args:
        ax: Matplotlib 3D axis.
        bundle: TangentBundle object.
        param_point: Array (base_dim,) — the base point in parameter space.
        fibre_component: Component of the vector w.r.t. the parameter-space basis.
        color, linewidth, mutation_scale, arrowstyle, alpha, zorder: Arrow styling.
        draw_base_point, base_point_color, base_point_size: Base point marker.
        draw_tip_marker, tip_marker, tip_marker_size: Marker at the tip of the arrow.
        label: Optional LaTeX label (e.g. r"$v_{\\gamma, p}$").
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy, dz) offset of the label in ambient space.

    Returns:
        Tuple (base, tip) with the ambient coordinates of both ends of the arrow.
    """
    base = np.asarray(bundle.embed(bundle.fibre(param_point, jnp.array([0.0]))))[0]
    tip = np.asarray(bundle.embed(bundle.fibre(param_point, jnp.array([fibre_component]))))[0]

    ax.add_artist(Arrow3D(
        [base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
        mutation_scale=mutation_scale, lw=linewidth, arrowstyle=arrowstyle,
        color=color, alpha=alpha,
    ))

    if draw_base_point:
        ax.scatter(base[0], base[1], base[2], color=base_point_color,
                   s=base_point_size, zorder=zorder + 1)
    if draw_tip_marker:
        ax.scatter(tip[0], tip[1], tip[2], color=color, marker=tip_marker,
                   s=tip_marker_size, zorder=zorder + 1)

    if label is not None:
        position = 0.5 * (base + tip) + np.asarray(label_offset)
        ax.text(position[0], position[1], position[2], label,
                fontsize=label_fontsize, color=color, zorder=zorder + 2)

    return base, tip


# ==========================================================
# === Bundle chart image (2D panel)
# ==========================================================

def style_bundle_chart_axes(
    ax,
    xlim,
    ylim,
    axis_color="black",
    axis_linewidth=1.5,
    base_axis_label=r"$\mathbb{R}^d$",
    space_label=r"$\mathbb{R}^{2d}$",
    label_fontsize=16,
    space_label_position=(0.94, 0.94),
):
    """
    Style a 2D axis as the target R^{2d} of a bundle chart: axes crossing at the origin,
    with an arrowhead at each end, no frame and no ticks.

    Args:
        ax: Matplotlib 2D axis.
        xlim, ylim: Axis limits.
        axis_color, axis_linewidth: Styling of the two axes.
        base_axis_label: Label of the horizontal axis (the base block, R^d).
        space_label: Label of the whole plane (the bundle chart image, R^{2d}).
        label_fontsize: Font size of the labels.
        space_label_position: Position of the plane label, in axes fraction.
    """
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.annotate("", xy=(xlim[1], 0), xytext=(xlim[0], 0),
                arrowprops=dict(arrowstyle="-|>", color=axis_color, lw=axis_linewidth))
    ax.annotate("", xy=(0, ylim[1]), xytext=(0, ylim[0]),
                arrowprops=dict(arrowstyle="-|>", color=axis_color, lw=axis_linewidth))

    if base_axis_label is not None:
        ax.text(xlim[1], -0.04 * (ylim[1] - ylim[0]), base_axis_label,
                fontsize=label_fontsize, ha="right", va="top", color=axis_color)
    if space_label is not None:
        ax.text(space_label_position[0], space_label_position[1], space_label,
                transform=ax.transAxes, fontsize=label_fontsize, ha="right", va="top",
                color=axis_color)


def plot_bundle_chart_image(
    ax,
    bundle,
    chart,
    component_range,
    color="red",
    facecolor="mistyrose",
    alpha=0.45,
    linestyle="dotted",
    linewidth=1.4,
    zorder=1,
    mark_base_interval=True,
    base_interval_linewidth=2.5,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0),
):
    """
    Plot the image  xi_X(TU) = X(U) x R  of the bundle chart, as a shaded rectangle,
    together with the image X(U) of the chart domain on the horizontal axis.

    Args:
        ax: Matplotlib 2D axis.
        bundle: TangentBundle object.
        chart: Chart object on the base manifold (1-dimensional base).
        component_range: Tuple (v_min, v_max) — vertical extent to draw (the fibre
            block really spans the whole of R).
        color: Color of the boundary and of the base interval.
        facecolor, alpha: Fill of the rectangle.
        linestyle, linewidth: Styling of the boundary.
        zorder: Drawing order.
        mark_base_interval: If True, marks X(U) on the horizontal axis.
        base_interval_linewidth: Line width of that mark.
        label: Optional LaTeX label for the rectangle.
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy) offset of the label.

    Returns:
        Tuple (x_min, x_max) with the image X(U) of the chart domain.
    """
    if bundle.base_dim != 1:
        raise NotImplementedError("Only implemented for a 1-dimensional base manifold.")

    t_min, t_max = chart.boundary_interval
    endpoints = np.asarray(chart.map_to_chart(jnp.array([[t_min], [t_max]]))).ravel()
    x_min, x_max = float(np.min(endpoints)), float(np.max(endpoints))

    ax.add_patch(Rectangle(
        (x_min, component_range[0]),
        x_max - x_min,
        component_range[1] - component_range[0],
        facecolor=facecolor, alpha=alpha, edgecolor=color,
        linestyle=linestyle, linewidth=linewidth, zorder=zorder,
    ))

    if mark_base_interval:
        ax.plot([x_min, x_max], [0.0, 0.0], color=color, linestyle=linestyle,
                linewidth=base_interval_linewidth, zorder=zorder + 2)
        for x_value in (x_min, x_max):
            ax.plot([x_value, x_value],
                    [-0.03 * (component_range[1] - component_range[0]),
                     0.03 * (component_range[1] - component_range[0])],
                    color=color, linewidth=base_interval_linewidth, zorder=zorder + 2)

    if label is not None:
        ax.text(x_max + label_offset[0], component_range[1] + label_offset[1], label,
                fontsize=label_fontsize, color=color, ha="left", va="bottom")

    return x_min, x_max


def plot_section_in_bundle_chart(
    ax,
    bundle,
    chart,
    field,
    param_range=None,
    n_points=400,
    color="magenta",
    linewidth=2.0,
    alpha=1.0,
    zorder=4,
    label=None,
    label_fontsize=16,
    label_position=0.5,
    label_offset=(0.0, 0.0),
):
    """
    Plot a vector field in bundle-chart coordinates: the graph of its component
    function, i.e. the curve  x -> ( x , X^1(x) )  with x = X(p) the chart coordinate.

    Args:
        ax: Matplotlib 2D axis.
        bundle: TangentBundle object.
        chart: Chart object on the base manifold.
        field: VectorField object on the base manifold.
        param_range: Tuple (t_min, t_max) of the base parameter. Defaults to the
            chart domain.
        n_points: Number of sampling points.
        color, linewidth, alpha, zorder: Line styling.
        label: Optional LaTeX label.
        label_fontsize: Font size of the label.
        label_position: Position of the label along the curve, in [0, 1].
        label_offset: Tuple (dx, dy) offset of the label.
    """
    if param_range is None:
        param_range = chart.boundary_interval

    param_points = jnp.linspace(param_range[0], param_range[1], n_points).reshape(-1, 1)
    bundle_points = bundle.section_of_field(field, param_points)
    chart_points = np.asarray(bundle.chart_map(chart, bundle_points))

    ax.plot(chart_points[:, 0], chart_points[:, 1], color=color,
            linewidth=linewidth, alpha=alpha, zorder=zorder)

    if label is not None:
        index = int(label_position * (n_points - 1))
        position = chart_points[index] + np.asarray(label_offset)
        ax.text(position[0], position[1], label, fontsize=label_fontsize,
                color=color, zorder=zorder + 1)

    return chart_points


def plot_vector_in_bundle_chart(
    ax,
    bundle,
    chart,
    param_point,
    fibre_component,
    component_range=None,
    color="blue",
    linewidth=2.5,
    zorder=5,
    draw_fibre=True,
    fibre_color="green",
    fibre_linewidth=1.2,
    draw_base_point=True,
    base_point_size=45,
    draw_tip_marker=True,
    tip_marker="*",
    tip_marker_size=160,
    label=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0),
):
    """
    Plot one tangent vector in bundle-chart coordinates: the base point on the
    horizontal axis, the image of its fibre as a vertical line, and the vector itself
    as an arrow of length equal to its component in the chart-induced basis.

    Args:
        ax: Matplotlib 2D axis.
        bundle: TangentBundle object.
        chart: Chart object on the base manifold.
        param_point: Array (base_dim,) — the base point in parameter space.
        fibre_component: Component of the vector w.r.t. the parameter-space basis.
        component_range: Tuple (v_min, v_max) — vertical extent of the fibre image.
        color, linewidth, zorder: Arrow styling.
        draw_fibre, fibre_color, fibre_linewidth: Image of the fibre.
        draw_base_point, base_point_size: Base point marker.
        draw_tip_marker, tip_marker, tip_marker_size: Marker at the tip of the arrow.
        label: Optional LaTeX label.
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy) offset of the label.

    Returns:
        Tuple (base, tip) with the 2D coordinates of both ends of the arrow.
    """
    bundle_point = bundle.fibre(param_point, jnp.array([fibre_component]))
    chart_point = np.asarray(bundle.chart_map(chart, bundle_point))[0]

    base = np.array([chart_point[0], 0.0])
    tip = np.array([chart_point[0], chart_point[1]])

    if draw_fibre and component_range is not None:
        ax.plot([base[0], base[0]], [component_range[0], component_range[1]],
                color=fibre_color, linewidth=fibre_linewidth, zorder=zorder - 1)
        for end_value, inner_value in [(component_range[1], 0.92 * component_range[1]),
                                       (component_range[0], 0.92 * component_range[0])]:
            ax.annotate("", xy=(base[0], end_value), xytext=(base[0], inner_value),
                        arrowprops=dict(arrowstyle="-|>", color=fibre_color,
                                        lw=fibre_linewidth))

    ax.annotate("", xy=tuple(tip), xytext=tuple(base),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=linewidth),
                zorder=zorder)

    if draw_base_point:
        ax.scatter(base[0], base[1], color=fibre_color, s=base_point_size, zorder=zorder + 1)
    if draw_tip_marker:
        ax.scatter(tip[0], tip[1], color=color, marker=tip_marker,
                   s=tip_marker_size, zorder=zorder + 1)

    if label is not None:
        position = 0.5 * (base + tip) + np.asarray(label_offset)
        ax.text(position[0], position[1], label, fontsize=label_fontsize, color=color,
                zorder=zorder + 2)

    return base, tip


# ==========================================================
# === Arrows between panels (the maps themselves)
# ==========================================================

def point_to_figure_fraction(ax, point):
    """
    Convert a data point of a 2D or 3D axis into figure-fraction coordinates, so that
    arrows representing maps can be drawn between panels.

    The figure must have been drawn at least once (call fig.canvas.draw()) for the 3D
    projection to be up to date.

    Args:
        ax: Matplotlib axis (2D or 3D).
        point: Data coordinates, of length 2 or 3.

    Returns:
        Array (2,) with the position in figure-fraction coordinates.
    """
    point = np.asarray(point, dtype=float)

    if hasattr(ax, "get_proj") and point.size == 3:
        x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
        display = ax.transData.transform((x2, y2))
    else:
        display = ax.transData.transform((point[0], point[1]))

    return ax.figure.transFigure.inverted().transform(display)


def add_map_arrow(
    fig,
    start,
    end,
    label=None,
    rad=0.3,
    color="red",
    linestyle=(0, (2, 2)),
    linewidth=1.2,
    mutation_scale=16,
    arrowstyle="-|>",
    label_fontsize=16,
    label_offset=(0.0, 0.02),
    label_position=0.5,
    zorder=10,
):
    """
    Draw a curved arrow between two points given in figure-fraction coordinates, to
    represent a map acting between two panels of the figure (e.g. the chart map or the
    bundle chart).

    Args:
        fig: Matplotlib figure.
        start, end: Tuples (x, y) in figure-fraction coordinates, typically obtained
            with point_to_figure_fraction.
        label: Optional LaTeX label for the map (e.g. r"$\\xi_\\mathscr{X}$").
        rad: Curvature of the arc (positive or negative).
        color, linestyle, linewidth, mutation_scale, arrowstyle: Arrow styling.
        label_fontsize: Font size of the label.
        label_offset: Tuple (dx, dy) offset of the label, in figure fraction.
        label_position: Position of the label along the straight chord, in [0, 1].
        zorder: Drawing order.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    arrow = FancyArrowPatch(
        tuple(start), tuple(end),
        transform=fig.transFigure,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linestyle=linestyle,
        linewidth=linewidth,
        color=color,
        zorder=zorder,
    )
    fig.add_artist(arrow)

    if label is not None:
        chord = start + label_position * (end - start)
        direction = end - start
        normal = np.array([-direction[1], direction[0]])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        position = chord - rad * 0.5 * np.linalg.norm(direction) * normal + np.asarray(label_offset)
        fig.text(position[0], position[1], label, fontsize=label_fontsize,
                 color=color, ha="center", va="center", zorder=zorder + 1)

    return arrow
