import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors, tri
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

import numpy as np
import jax
import jax.numpy as jnp

from shapely.geometry import Polygon, Point


class Arrow3D(FancyArrowPatch):
    """
    A custom 3D arrow for matplotlib.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)

        # Explicit conversion to NumPy
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.mean(zs)  # Use numpy mean, not jnp

def plot_tangent_planes(
    ax,
    embedded_points,
    jacobians,
    size=1.0,
    color="lightgray",
    alpha=0.5,
    resolution=10,
):
    """
    Plot tangent planes in 3D for multiple manifold points.

    Args:
        ax: Matplotlib 3D axis.
        embedded_points: Array (N, 3) — embedded base points on the manifold.
        jacobians: Array (N, 2, 3) — tangent vectors at each point.
        size: Size scaling for each plane.
        color: Plane surface color.
        alpha: Plane surface transparency.
        resolution: Grid resolution for the plane surface.
    """
    N, D = embedded_points.shape
    _, param_dim, D2 = jacobians.shape
    assert D == D2 == 3, "This function only supports ambient_dim == 3."
    assert param_dim == 2, "Only defined for 2D parameter spaces."

    s = np.linspace(-size, size, resolution)
    t = np.linspace(-size, size, resolution)
    S, T = np.meshgrid(s, t)

    for i in range(N):
        p_xyz = np.array(embedded_points[i])
        v1 = np.array(jacobians[i, 0])
        v2 = np.array(jacobians[i, 1])

        plane = (
            p_xyz.reshape(3, 1, 1)
            + S * v1.reshape(3, 1, 1)
            + T * v2.reshape(3, 1, 1)
        )

        X, Y, Z = plane[0], plane[1], plane[2]
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor="none")




def plot_colored_curve_directional_derivative(
    ax,
    curve,
    scalar_function,
    lambda_range=(-jnp.pi, jnp.pi),
    n_points=300,
    cmap="plasma",
    linewidth=3.0,
    label=None,
    label_fontsize=14,
    label_offset=(0.1, 0.1, 0.2),
    label_position="center",
    method="autodiff",
    inset_colorbar=True,
    inset_position=(0.02, 0.02),   # (x, y) in axes fraction
    inset_size=(0.25, 0.02),       # (width, height) in axes fraction
    colorbar_orientation="horizontal",  # "horizontal" or "vertical"
    label_colorbar=r"$\partial_\lambda f$",
    label_colorbar_position="top",       # "top", "bottom", "left", "right"
    inset_box_alpha=0.8,                 # Opacity for box
    inset_box_color="white",             # Background box color
    inset_border_color="black"           # Border color
):
    """
    Plot a curve colored by the directional derivative of a function.

    Includes an optional colorbar inset placed using coordinates relative to the plot.
    """

    # === Sample curve ===
    lambdas = jnp.linspace(lambda_range[0], lambda_range[1], n_points)
    points = curve.evaluate_on_manifold(lambdas)

    # === Compute directional derivatives ===
    derivatives = jnp.array([
        scalar_function.directional_derivative_along_curve(curve, float(lmb), method=method)
        for lmb in lambdas
    ])

    # === Normalize colormap ===
    norm = mcolors.Normalize(vmin=float(derivatives.min()), vmax=float(derivatives.max()))
    cmap_obj = cm.get_cmap(cmap)
    colors = cmap_obj(norm(derivatives))

    # === Build line segments ===
    segments = [[points[i], points[i + 1]] for i in range(len(points) - 1)]

    line_collection = Line3DCollection(
        segments, colors=colors[:-1], linewidths=linewidth, alpha=0.95
    )
    ax.add_collection3d(line_collection)

    # === Plot label ===
    if label is not None:
        idx = (
            len(points) // 2 if label_position == "center"
            else 0 if label_position == "start"
            else -1
        )
        p_label = points[idx]
        dx, dy, dz = label_offset
        ax.text(
            p_label[0] + dx, p_label[1] + dy, p_label[2] + dz,
            label, fontsize=label_fontsize, color="black"
        )

    # === Add inset colorbar ===
    if inset_colorbar:
        cbax = inset_axes(
            ax,
            width=inset_size[0],
            height=inset_size[1],
            loc='lower left',
            bbox_to_anchor=(inset_position[0], inset_position[1], 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(derivatives)

        cbar = plt.colorbar(
            mappable,
            cax=cbax,
            orientation=colorbar_orientation
        )

        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)

        # === Background box ===
        for spine in cbax.spines.values():
            spine.set_edgecolor(inset_border_color)
            spine.set_linewidth(1.0)

        cbax.set_facecolor(inset_box_color)
        cbax.patch.set_alpha(inset_box_alpha)

        # === Place colorbar label ===
        if label_colorbar:
            if colorbar_orientation == "horizontal":
                if label_colorbar_position == "top":
                    cbax.set_title(label_colorbar, fontsize=10, pad=4)
                elif label_colorbar_position == "bottom":
                    cbax.set_xlabel(label_colorbar, fontsize=10, labelpad=4)
                else:
                    raise ValueError("Label position for horizontal colorbar must be 'top' or 'bottom'.")
            else:  # vertical
                if label_colorbar_position == "right":
                    cbax.set_ylabel(label_colorbar, fontsize=10, rotation=-90, labelpad=10)
                elif label_colorbar_position == "left":
                    cbax.yaxis.set_label_position("left")
                    cbax.set_ylabel(label_colorbar, fontsize=10, rotation=90, labelpad=10)
                else:
                    raise ValueError("Label position for vertical colorbar must be 'left' or 'right'.")

    return line_collection

def plot_colored_curve_directional_derivative_in_chart(
    ax,
    chart,
    curve,
    scalar_function,
    lambda_range=(-jnp.pi, jnp.pi),
    n_points=300,
    cmap="plasma",
    linewidth=3.0,
    label=None,
    label_fontsize=14,
    label_offset=(0.2, 0.2),
    label_position="center",
    method="autodiff",
    inset_colorbar=True,
    inset_position=(0.02, 0.02),   # (x, y) in axes fraction
    inset_size=(0.25, 0.02),       # (width, height) in axes fraction
    colorbar_orientation="horizontal",  # "horizontal" or "vertical"
    label_colorbar=r"$\partial_\lambda f$",
    label_colorbar_position="top",       # "top", "bottom", "left", "right"
    inset_box_alpha=0.8,                 # Opacity for box
    inset_box_color="white",             # Background box color
    inset_border_color="black"           # Border color
):
    """
    Plot a curve in chart coordinates colored by the directional derivative of a function.

    Includes an optional colorbar inset placed using coordinates relative to the plot.
    """

    # === Sample curve ===
    lambdas = jnp.linspace(lambda_range[0], lambda_range[1], n_points)
    param_points = curve.evaluate_in_param_space(lambdas)
    points_chart = chart.map_to_chart(param_points)

    # === Compute directional derivatives ===
    derivatives = jnp.array([
        scalar_function.directional_derivative_along_curve(curve, float(lmb), method=method)
        for lmb in lambdas
    ])

    # === Normalize colormap ===
    norm = Normalize(vmin=float(derivatives.min()), vmax=float(derivatives.max()))
    cmap_obj = cm.get_cmap(cmap)
    colors = cmap_obj(norm(derivatives))

    # === Build line segments ===
    segments = [[points_chart[i], points_chart[i + 1]] for i in range(len(points_chart) - 1)]

    line_collection = LineCollection(
        segments, colors=colors[:-1], linewidths=linewidth, alpha=0.7, zorder=4
    )
    ax.add_collection(line_collection)

    # === Plot label ===
    if label is not None:
        idx = (
            len(points_chart) // 2 if label_position == "center"
            else 0 if label_position == "start"
            else -1
        )
        pos = points_chart[idx] + jnp.array(label_offset)

        ax.text(
            pos[0], pos[1], label,
            fontsize=label_fontsize, color="black",
            ha="center", va="center"
        )

    # === Add inset colorbar ===
    if inset_colorbar:
        cbax = inset_axes(
            ax,
            width=inset_size[0],
            height=inset_size[1],
            loc='lower left',
            bbox_to_anchor=(inset_position[0], inset_position[1], 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(derivatives)

        cbar = mpl.colorbar.ColorbarBase(
            cbax, cmap=cmap_obj, norm=norm, orientation=colorbar_orientation
        )

        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)

        # === Background box ===
        for spine in cbax.spines.values():
            spine.set_edgecolor(inset_border_color)
            spine.set_linewidth(1.0)

        cbax.set_facecolor(inset_box_color)
        cbax.patch.set_alpha(inset_box_alpha)

        # === Place colorbar label ===
        if label_colorbar:
            if colorbar_orientation == "horizontal":
                if label_colorbar_position == "top":
                    cbax.set_title(label_colorbar, fontsize=10, pad=4)
                elif label_colorbar_position == "bottom":
                    cbax.set_xlabel(label_colorbar, fontsize=10, labelpad=4)
                else:
                    raise ValueError("Label position for horizontal colorbar must be 'top' or 'bottom'.")
            else:  # vertical
                if label_colorbar_position == "right":
                    cbax.set_ylabel(label_colorbar, fontsize=10, rotation=-90, labelpad=10)
                elif label_colorbar_position == "left":
                    cbax.yaxis.set_label_position("left")
                    cbax.set_ylabel(label_colorbar, fontsize=10, rotation=90, labelpad=10)
                else:
                    raise ValueError("Label position for vertical colorbar must be 'left' or 'right'.")

    return line_collection





def plot_vector_field_on_manifold(
    ax,
    field,
    grid_resolution=16,
    xlim=(-jnp.pi, jnp.pi),
    ylim=(-jnp.pi, jnp.pi),
    vector_scale=0.3,
    color="purple",
    colormap=None,
    arrow_style=None,
    label=None,
    label_fontsize=14,
    inset_position=(0.02, 0.02),  # bottom-left corner (x, y)
    inset_size=(0.25, 0.02),      # width, height
    inset_orientation="horizontal"
):
    """
    Plot a vector field on the manifold with optional colormap inset.

    Args:
        ax: Matplotlib 3D axis.
        field: Field object defined on the manifold.
        grid_resolution: Mesh resolution for sampling.
        xlim, ylim: Bounds in parameter space.
        vector_scale: Arrow length scaling.
        color: Fixed color if colormap is not used.
        colormap: Matplotlib colormap name or object.
        arrow_style: Dict for customizing arrows.
        label: Optional label for the inset (e.g. "$X$").
        label_fontsize: Font size for label.
        inset_position: (x, y) — position of the inset box in axes fraction.
        inset_size: (width, height) — size of the inset box in axes fraction.
        inset_orientation: 'horizontal' or 'vertical'.
    """

    # === Build meshgrid ===
    x_vals = jnp.linspace(xlim[0], xlim[1], grid_resolution)
    y_vals = jnp.linspace(ylim[0], ylim[1], grid_resolution)
    XX, YY = jnp.meshgrid(x_vals, y_vals)
    param_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    # === Compute field ===
    points_on_manifold = field.manifold.embed(param_points)
    vectors_ambient = field.evaluate_on_manifold(param_points)

    norms = jnp.linalg.norm(vectors_ambient, axis=1, keepdims=True)
    unit_vectors = vectors_ambient / norms

    # === Colormap setup ===
    if colormap is not None:
        cmap = cm.get_cmap(colormap) if isinstance(colormap, str) else colormap
        num_points = param_points.shape[0]
        colors = cmap(np.linspace(0, 1, num_points))
    else:
        colors = [color] * param_points.shape[0]

    # === Arrow style ===
    default_arrow_style = dict(
        arrowstyle='-|>,head_length=4.,head_width=3.',
        mutation_scale=0.8,
        lw=0.8,
        alpha=0.6
    )
    if arrow_style is not None:
        default_arrow_style.update(arrow_style)

    # === Plot arrows ===
    for (p, v, c) in zip(points_on_manifold, unit_vectors, colors):
        ax.add_artist(Arrow3D(
            [float(p[0]), float(p[0] + v[0] * vector_scale)],
            [float(p[1]), float(p[1] + v[1] * vector_scale)],
            [float(p[2]), float(p[2] + v[2] * vector_scale)],
            color=c,
            **default_arrow_style
        ))

    # === Add colormap inset ===
    if colormap is not None and label is not None:
        inset_ax = inset_axes(
            ax,
            width=inset_size[0],
            height=inset_size[1],
            loc='lower left',
            bbox_to_anchor=(inset_position[0], inset_position[1], 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        norm = Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=inset_ax,
            orientation=inset_orientation
        )
        cb.set_ticks([])
        if inset_orientation == "horizontal":
            cb.ax.set_xlabel(label, fontsize=label_fontsize, labelpad=2)
        else:
            cb.ax.set_ylabel(label, fontsize=label_fontsize, labelpad=2)

        cb.outline.set_visible(False)

def plot_vector_field_in_chart(
    ax,
    points_in_chart,
    vectors_in_chart,
    vector_scale=0.3,
    color="purple",
    colormap=None,
    arrow_style=None,
    label=None,
    label_fontsize=10,
    inset_position=(0.02, 0.02),
    inset_size=(0.25, 0.02),
    inset_orientation="horizontal",
    inset_box_alpha=0.8,
    inset_box_color="white",
    inset_border_color="black"
):
    """
    Plot a vector field in chart coordinates (2D) with optional colorbar inset.

    Args:
        ax: Matplotlib 2D axis.
        points_in_chart: (N, 2) array of positions in chart coordinates.
        vectors_in_chart: (N, 2) array of vector components.
        vector_scale: Scale factor for arrow length.
        color: Fixed color (ignored if colormap is provided).
        colormap: Matplotlib colormap name or object (optional).
        arrow_style: Dict of arrow style parameters (overrides defaults).
        label: Label for the colormap inset (e.g., r"$X$").
        label_fontsize: Font size for the label.
        inset_position: (x, y) position of inset (axes fraction).
        inset_size: (width, height) size of inset (axes fraction).
        inset_orientation: 'horizontal' or 'vertical'.
        inset_box_alpha: Opacity of the background box.
        inset_box_color: Background box color.
        inset_border_color: Border color of the box.
    """

    N = points_in_chart.shape[0]

    # === Colormap handling ===
    if colormap is not None:
        cmap = cm.get_cmap(colormap) if isinstance(colormap, str) else colormap
        colors = cmap(np.linspace(0, 1, N))
    else:
        colors = [color] * N

    # === Arrow style handling ===
    default_arrow_style = dict(
        head_width=0.1,
        head_length=0.15,
        linewidth=0.8,
        alpha=0.6,
        length_includes_head=True
    )
    # If arrow_style provided, override defaults
    if arrow_style is not None:
        default_arrow_style.update(arrow_style)

    # === Plot arrows ===
    for p, v, c in zip(points_in_chart, vectors_in_chart, colors):
        ax.arrow(
            p[0], p[1],
            v[0] * vector_scale, v[1] * vector_scale,
            fc=c, ec=c,
            **default_arrow_style
        )

    # === Add colormap inset ===
    if colormap is not None and label is not None:
        inset_ax = inset_axes(
            ax,
            width=inset_size[0],
            height=inset_size[1],
            loc='lower left',
            bbox_to_anchor=(inset_position[0], inset_position[1], 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        norm = Normalize(vmin=0, vmax=1)
        cb = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=inset_ax,
            orientation=inset_orientation
        )
        cb.set_ticks([])
        cb.outline.set_visible(False)

        if inset_orientation == "horizontal":
            cb.ax.set_xlabel(label, fontsize=label_fontsize, labelpad=2)
        else:
            cb.ax.set_ylabel(label, fontsize=label_fontsize, labelpad=2)

        # === Background box ===
        for spine in inset_ax.spines.values():
            spine.set_edgecolor(inset_border_color)
            spine.set_linewidth(1.0)

        inset_ax.set_facecolor(inset_box_color)
        inset_ax.patch.set_alpha(inset_box_alpha)

