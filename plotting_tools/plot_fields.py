import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plotting_tools.basic_plotting_tools import Arrow3D


def _resolve_cmap(cmap):
    """Accept a colormap name (str) or a Matplotlib Colormap object."""
    if isinstance(cmap, mcolors.Colormap):
        return cmap                      # e.g. tc.sunset, tc.colormaps["sunset"]
    if isinstance(cmap, str):
        return matplotlib.colormaps[cmap]  # e.g. "viridis"
    raise TypeError(f"cmap must be a str or a matplotlib Colormap, got {type(cmap)}")


def _arrow_colors(magnitudes, color, colormap, vmin=None, vmax=None):
    """
    Colors for a set of arrows: a fixed color, or a colormap applied to the
    magnitudes of the vectors (so that the color carries information).

    Returns:
        Tuple (colors, norm, cmap_object). norm and cmap_object are None when no
        colormap is used.
    """
    if colormap is None:
        return [color] * len(magnitudes), None, None

    cmap_object = _resolve_cmap(colormap)
    norm = Normalize(
        vmin=float(np.min(magnitudes)) if vmin is None else vmin,
        vmax=float(np.max(magnitudes)) if vmax is None else vmax,
    )
    return cmap_object(norm(magnitudes)), norm, cmap_object


def _add_colorbar_inset(
    ax,
    norm,
    cmap_object,
    label=None,
    label_fontsize=12,
    inset_position=(0.02, 0.02),
    inset_size=(0.6, 0.2),
    inset_orientation="horizontal",
    show_ticks=True,
    tick_fontsize=8,
    inset_box_alpha=0.8,
    inset_box_color="white",
    inset_border_color="black",
):
    """
    Small colorbar inset, following the same style as the other plotting modules.
    """
    inset_ax = inset_axes(
        ax,
        width=inset_size[0],
        height=inset_size[1],
        loc="lower left",
        bbox_to_anchor=(inset_position[0], inset_position[1], 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )

    colorbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap_object),
        cax=inset_ax,
        orientation=inset_orientation,
    )

    if show_ticks:
        colorbar.ax.tick_params(labelsize=tick_fontsize)
    else:
        colorbar.set_ticks([])
    colorbar.outline.set_visible(False)

    if label is not None:
        if inset_orientation == "horizontal":
            inset_ax.set_title(label, fontsize=label_fontsize, pad=4)
        else:
            inset_ax.set_ylabel(label, fontsize=label_fontsize, labelpad=4)

    for spine in inset_ax.spines.values():
        spine.set_edgecolor(inset_border_color)
        spine.set_linewidth(1.0)
    inset_ax.set_facecolor(inset_box_color)
    inset_ax.patch.set_alpha(inset_box_alpha)

    return inset_ax


def param_grid(xlim, ylim, grid_resolution):
    """
    Regular grid of points in parameter space.

    Args:
        xlim, ylim: Bounds of the grid.
        grid_resolution: Number of points per axis.

    Returns:
        Array of shape (grid_resolution**2, 2).
    """
    x_values = jnp.linspace(xlim[0], xlim[1], grid_resolution)
    y_values = jnp.linspace(ylim[0], ylim[1], grid_resolution)
    XX, YY = jnp.meshgrid(x_values, y_values)
    return jnp.stack([XX.ravel(), YY.ravel()], axis=-1)


def chart_grid_in_param_space(chart, xlim, ylim, grid_resolution):
    """
    Regular grid in *chart* coordinates, pulled back to parameter space with the
    inverse chart map. Plotting a field at these base points gives a tidy grid of
    arrows in the chart panel.

    Args:
        chart: Chart object with an inverse_chart_map.
        xlim, ylim: Bounds of the grid, in chart coordinates.
        grid_resolution: Number of points per axis.

    Returns:
        Array of shape (grid_resolution**2, 2) with points in parameter space.
    """
    if chart.inverse_chart_map is None:
        raise ValueError(
            "chart_grid_in_param_space requires an inverse_chart_map; "
            "pass param_points explicitly instead."
        )
    chart_points = param_grid(xlim, ylim, grid_resolution)
    return jax.vmap(chart.inverse_chart_map)(chart_points)


# ==========================================================
# === Vector fields on the manifold (ambient space)
# ==========================================================

def plot_vector_field_on_manifold(
    ax,
    field,
    grid_resolution=16,
    xlim=(-jnp.pi, jnp.pi),
    ylim=(-jnp.pi, jnp.pi),
    param_points=None,
    vector_scale=0.3,
    normalize=True,
    color="purple",
    colormap=None,
    vmin=None,
    vmax=None,
    arrow_style=None,
    label=None,
    label_fontsize=12,
    inset_colorbar=True,
    inset_position=(0.02, 0.02),   # bottom-left corner (x, y)
    inset_size=(0.6, 0.2),         # width, height
    inset_orientation="horizontal",
    inset_box_alpha=0.8,
    inset_box_color="white",
    inset_border_color="black",
):
    """
    Plot a vector field on the manifold, with an optional magnitude colormap.

    The arrows are the pushforward of the field to the ambient space, attached at
    the corresponding points of the embedded manifold.

    Args:
        ax: Matplotlib 3D axis.
        field: VectorField object defined on the manifold.
        param_points: Optional array (N, dim) of base points in parameter space.
            If None, a regular grid over xlim x ylim is used.
        grid_resolution: Mesh resolution for sampling (ignored if param_points given).
        xlim, ylim: Bounds in parameter space (ignored if param_points given).
        vector_scale: Arrow length scaling.
        normalize: If True, all arrows are drawn with the same length and the
            magnitude information is carried by the colormap.
        color: Fixed color if colormap is None.
        colormap: Matplotlib colormap name or object; colors the arrows by the
            magnitude of the field (in ambient space).
        vmin, vmax: Optional bounds for the magnitude colormap (useful when a few
            large values would otherwise flatten the scale).
        arrow_style: Dict for customizing arrows (merged with the defaults).
        label: Optional label for the colorbar inset (e.g. r"$\\mathcal{X}$").
        label_fontsize: Font size for the label.
        inset_colorbar: Whether to draw the colorbar inset (needs a colormap).
        inset_position: (x, y) — position of the inset box in axes fraction.
        inset_size: (width, height) — size of the inset box, in inches.
        inset_orientation: 'horizontal' or 'vertical'.
        inset_box_alpha, inset_box_color, inset_border_color: Inset box styling.

    Returns:
        Array (N,) with the magnitudes of the plotted vectors (ambient space).
    """
    if param_points is None:
        param_points = param_grid(xlim, ylim, grid_resolution)
    param_points = jnp.asarray(param_points)

    points_on_manifold = field.manifold.embed(param_points)
    vectors_ambient = field.evaluate_on_manifold(param_points)

    magnitudes = np.asarray(jnp.linalg.norm(vectors_ambient, axis=1))
    if normalize:
        vectors_ambient = vectors_ambient / jnp.linalg.norm(vectors_ambient, axis=1, keepdims=True)

    colors, norm, cmap_object = _arrow_colors(magnitudes, color, colormap, vmin, vmax)

    # === Arrow style ===
    default_arrow_style = dict(
        arrowstyle='-|>,head_length=4.,head_width=3.',
        mutation_scale=0.8,
        lw=0.8,
        alpha=0.8,
    )
    if arrow_style is not None:
        default_arrow_style.update(arrow_style)

    # === Plot arrows ===
    for (point, vector, arrow_color) in zip(points_on_manifold, vectors_ambient, colors):
        ax.add_artist(Arrow3D(
            [float(point[0]), float(point[0] + vector[0] * vector_scale)],
            [float(point[1]), float(point[1] + vector[1] * vector_scale)],
            [float(point[2]), float(point[2] + vector[2] * vector_scale)],
            color=arrow_color,
            **default_arrow_style
        ))

    # === Colorbar inset ===
    if colormap is not None and inset_colorbar:
        _add_colorbar_inset(
            ax, norm, cmap_object, label=label, label_fontsize=label_fontsize,
            inset_position=inset_position, inset_size=inset_size,
            inset_orientation=inset_orientation, inset_box_alpha=inset_box_alpha,
            inset_box_color=inset_box_color, inset_border_color=inset_border_color,
        )

    return magnitudes


def plot_field_section_on_manifold(
    ax,
    field,
    param_points,
    plane_size=0.5,
    plane_resolution=6,
    plane_color="lightgray",
    plane_alpha=0.35,
    vector_scale=1.0,
    normalize=False,
    color="crimson",
    linewidth=2.0,
    mutation_scale=15,
    arrowstyle="-|>",
    alpha=1.0,
    draw_points=True,
    point_color="black",
    point_size=40,
    labels=None,
    label_fontsize=14,
    label_offset=(0.0, 0.0, 0.1),
):
    """
    Plot a vector field as a *section of the tangent bundle*: at each given base
    point, the fibre T_pM is drawn as a patch of the tangent plane, and the vector
    picked by the field inside that fibre is drawn as an arrow.

    Args:
        ax: Matplotlib 3D axis.
        field: VectorField object.
        param_points: Array (N, dim) — base points in parameter space.
        plane_size: Half-extent of the fibre patches, in units of the
            parameter-space basis vectors.
        plane_resolution: Grid resolution of the fibre patches.
        plane_color: Color of the fibre patches.
        plane_alpha: Opacity of the fibre patches.
        vector_scale: Scaling factor for the arrows.
        normalize: If True, all arrows are drawn with the same length.
        color: Color of the arrows.
        linewidth: Arrow line width.
        mutation_scale: Arrowhead size.
        arrowstyle: Arrow style for FancyArrowPatch.
        alpha: Opacity of the arrows.
        draw_points: If True, marks the base points.
        point_color: Color of the base point markers.
        point_size: Size of the base point markers.
        labels: Optional list of N strings, one label per fibre (e.g. r"$T_p\\mathcal{M}$").
        label_fontsize: Font size of the labels.
        label_offset: Tuple (dx, dy, dz) offset of the labels in ambient space.
    """
    param_points = jnp.atleast_2d(jnp.asarray(param_points))

    base_points = np.asarray(field.manifold.embed(param_points))                  # (N, ambient)
    tangent_bases = np.asarray(field.manifold.derivatives_at_params(param_points))  # (N, dim, ambient)
    vectors_ambient = np.asarray(field.evaluate_on_manifold(param_points))        # (N, ambient)

    if normalize:
        vectors_ambient = vectors_ambient / np.linalg.norm(vectors_ambient, axis=1, keepdims=True)

    steps = np.linspace(-plane_size, plane_size, plane_resolution)
    S, T = np.meshgrid(steps, steps)

    for index in range(base_points.shape[0]):
        base = base_points[index]
        basis_1, basis_2 = tangent_bases[index, 0], tangent_bases[index, 1]

        plane = (
            base.reshape(3, 1, 1)
            + S * basis_1.reshape(3, 1, 1)
            + T * basis_2.reshape(3, 1, 1)
        )
        ax.plot_surface(
            plane[0], plane[1], plane[2],
            color=plane_color, alpha=plane_alpha, edgecolor="none", shade=False,
        )

        tip = base + vector_scale * vectors_ambient[index]
        ax.add_artist(Arrow3D(
            [base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]],
            mutation_scale=mutation_scale, lw=linewidth, arrowstyle=arrowstyle,
            color=color, alpha=alpha,
        ))

        if draw_points:
            ax.scatter(base[0], base[1], base[2], color=point_color, s=point_size)

        if labels is not None:
            position = base + np.asarray(label_offset)
            ax.text(
                position[0], position[1], position[2], labels[index],
                fontsize=label_fontsize, color=plane_color if plane_color != "lightgray" else "black",
            )


# ==========================================================
# === Vector fields in chart coordinates
# ==========================================================

def plot_vector_field_in_chart(
    ax,
    points_in_chart,
    vectors_in_chart,
    vector_scale=0.3,
    normalize=True,
    color="purple",
    colormap=None,
    vmin=None,
    vmax=None,
    width=0.005,
    alpha=0.9,
    zorder=3,
    label=None,
    label_fontsize=12,
    inset_colorbar=True,
    inset_position=(0.02, 0.02),
    inset_size=(0.6, 0.2),
    inset_orientation="horizontal",
    inset_box_alpha=0.8,
    inset_box_color="white",
    inset_border_color="black",
):
    """
    Plot a vector field in chart coordinates (2D), with an optional magnitude
    colormap. This is the low-level primitive: the positions and the components
    are given already expressed in the chart.

    Args:
        ax: Matplotlib 2D axis.
        points_in_chart: (N, 2) array of base points in chart coordinates.
        vectors_in_chart: (N, 2) array of component functions X^i at those points.
        vector_scale: Scale factor for the arrow length.
        normalize: If True, all arrows are drawn with the same length and the
            magnitude information is carried by the colormap.
        color: Fixed color (used if colormap is None).
        colormap: Matplotlib colormap name or object; colors arrows by magnitude.
            Note that the magnitude of the *components* in a chart is not an
            intrinsic quantity: it changes from chart to chart.
        vmin, vmax: Optional bounds for the magnitude colormap.
        width: Shaft width of the arrows (Matplotlib quiver units).
        alpha: Opacity.
        zorder: Drawing order.
        label: Label for the colorbar inset (e.g. r"$\\mathcal{X}$").
        label_fontsize: Font size for the label.
        inset_colorbar: Whether to draw the colorbar inset (needs a colormap).
        inset_position: (x, y) position of the inset (axes fraction).
        inset_size: (width, height) size of the inset, in inches.
        inset_orientation: 'horizontal' or 'vertical'.
        inset_box_alpha, inset_box_color, inset_border_color: Inset box styling.

    Returns:
        Array (N,) with the magnitudes of the plotted vectors (chart coordinates).
    """
    points_in_chart = np.asarray(points_in_chart, dtype=float).reshape(-1, 2)
    vectors_in_chart = np.asarray(vectors_in_chart, dtype=float).reshape(-1, 2)

    magnitudes = np.linalg.norm(vectors_in_chart, axis=1)
    if normalize:
        vectors_in_chart = vectors_in_chart / magnitudes[:, None]

    colors, norm, cmap_object = _arrow_colors(magnitudes, color, colormap, vmin, vmax)

    ax.quiver(
        points_in_chart[:, 0], points_in_chart[:, 1],
        vectors_in_chart[:, 0] * vector_scale, vectors_in_chart[:, 1] * vector_scale,
        angles="xy", scale_units="xy", scale=1.0,
        color=colors, width=width, alpha=alpha, zorder=zorder,
    )

    if colormap is not None and inset_colorbar:
        _add_colorbar_inset(
            ax, norm, cmap_object, label=label, label_fontsize=label_fontsize,
            inset_position=inset_position, inset_size=inset_size,
            inset_orientation=inset_orientation, inset_box_alpha=inset_box_alpha,
            inset_box_color=inset_box_color, inset_border_color=inset_border_color,
        )

    return magnitudes


def plot_field_in_chart(
    ax,
    field,
    chart,
    xlim,
    ylim,
    grid_resolution=16,
    param_points=None,
    **kwargs,
):
    """
    Plot a vector field in chart coordinates, computing the component functions
    X^i from the field itself.

    By default the base points are a regular grid *in chart coordinates* (pulled
    back to parameter space with the inverse chart map), so that the arrows form a
    tidy grid in the panel.

    Args:
        ax: Matplotlib 2D axis.
        field: VectorField object.
        chart: Chart object defined on the same manifold.
        xlim, ylim: Bounds of the grid, in chart coordinates.
        grid_resolution: Number of grid points per axis.
        param_points: Optional array (N, dim) of base points in parameter space,
            used instead of the chart grid.
        **kwargs: Forwarded to plot_vector_field_in_chart (styling, colormap, ...).

    Returns:
        Array (N,) with the magnitudes of the plotted vectors (chart coordinates).
    """
    if param_points is None:
        param_points = chart_grid_in_param_space(chart, xlim, ylim, grid_resolution)
    param_points = jnp.asarray(param_points)

    points_in_chart = field.base_points_in_chart(chart, param_points)
    components = field.components_in_chart(chart, param_points)

    return plot_vector_field_in_chart(ax, points_in_chart, components, **kwargs)
