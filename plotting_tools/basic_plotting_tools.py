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

# === Plot Setup Function ===
def create_3d_axis_for_manifold(
    xlim, ylim, zlim,
    axis_labels=None,   # <- Default is None → no arrows
    figsize=(8, 8),
    coord_arrow_shift=0.5,
    coord_arrow_style=None,
    axis_label_fontsize=14
):
    """
    Create a 3D matplotlib figure with manifold plotting conventions.

    Args:
        xlim, ylim, zlim: Tuple of (min, max) for each axis.
        axis_labels: None → no coordinate arrows;
                     or tuple like ('x', 'y', 'z') to draw them with labels.
        figsize: Figure size in inches.
        coord_arrow_shift: Shift to position coordinate arrows.
        coord_arrow_style: Dict for customizing arrow appearance.
        axis_label_fontsize: Font size for axis labels.

    Returns:
        fig, ax: Matplotlib figure and axis.
    """

    if coord_arrow_style is None:
        coord_arrow_style = dict(
            mutation_scale=20,
            arrowstyle='-|>',
            color='black',
            linewidth=1.5
        )

    # === Create figure and 3D axis ===
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # === Set limits ===
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    # === Hide grid, ticks, panes ===
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.grid(False)

    # === Optional: Coordinate arrows ===
    if axis_labels is not None:
        x_min, x_max = xlim
        y_min, y_max = ylim
        z_min, z_max = zlim

        # X arrow
        ax.add_artist(Arrow3D(
            [x_min - coord_arrow_shift, x_min - coord_arrow_shift],
            [y_min - coord_arrow_shift, y_max/4 - coord_arrow_shift],
            [z_min - coord_arrow_shift, 0 - coord_arrow_shift],
            **coord_arrow_style))

        # Y arrow
        ax.add_artist(Arrow3D(
            [x_min - coord_arrow_shift, x_max/4 - coord_arrow_shift],
            [y_min - coord_arrow_shift, y_min - coord_arrow_shift],
            [z_min - coord_arrow_shift, 0 - coord_arrow_shift],
            **coord_arrow_style))

        # Z arrow
        ax.add_artist(Arrow3D(
            [x_min - coord_arrow_shift, x_min - coord_arrow_shift],
            [y_min - coord_arrow_shift, y_min - coord_arrow_shift],
            [z_min - coord_arrow_shift, z_max/2 - coord_arrow_shift],
            **coord_arrow_style))

        # Labels
        ax.text(
            x_min - coord_arrow_shift,
            y_max/4 - coord_arrow_shift,
            0 - coord_arrow_shift,
            f"${axis_labels[0]}$",
            fontsize=axis_label_fontsize
        )
        ax.text(
            x_max/4 - coord_arrow_shift,
            y_min - coord_arrow_shift,
            0 - coord_arrow_shift,
            f"${axis_labels[1]}$",
            fontsize=axis_label_fontsize
        )
        ax.text(
            x_min - coord_arrow_shift,
            y_min - coord_arrow_shift,
            z_max/2 - coord_arrow_shift,
            f"${axis_labels[2]}$",
            fontsize=axis_label_fontsize
        )

    return fig, ax

def plot_manifold_surface(
    ax,
    surface_manifold,
    resolution=100,
    xmin=-jnp.pi, xmax=jnp.pi,
    ymin=-jnp.pi, ymax=jnp.pi,
    color="skyblue",
    alpha=0.2,
    edgecolor="none",
    label=None,
    label_position="center",
    label_fontsize=12,
    function=None,
    cmap="viridis",
    label_function=None
):
    """
    Plot the manifold surface over a rectangular region in parameter space.
    Optionally color by a scalar function defined on the manifold.

    Args:
        ax: Matplotlib 3D axis.
        surface_manifold: Manifold object.
        resolution: Number of points per axis.
        xmin, xmax, ymin, ymax: Parameter space limits.
        color: Surface color if function is not provided.
        alpha: Transparency.
        edgecolor: Mesh line color.
        label: Optional LaTeX label for the surface.
        label_position: ["center", "top", "bottom", "left", "right"].
        label_fontsize: Font size for label.
        function: Optional scalar Function object to colormap.
        cmap: Colormap (str or matplotlib colormap).
        label_function: Label for the colorbar.

    Returns:
        surface: The plotted surface object.
    """

    # === Create grid in parameter space ===
    x = jnp.linspace(xmin, xmax, resolution)
    y = jnp.linspace(ymin, ymax, resolution)
    XX, YY = jnp.meshgrid(x, y)
    param_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    # === Embed points ===
    embedded = surface_manifold.embed(param_points)
    X = embedded[:, 0].reshape(XX.shape)
    Y = embedded[:, 1].reshape(XX.shape)
    Z = embedded[:, 2].reshape(XX.shape)

    # === Compute function for coloring if provided ===
    if function is not None:
        # === Compute normalized function values ===
        F_vals = function.evaluate_in_param_space(param_points).reshape(XX.shape)
        norm = mcolors.Normalize(vmin=float(F_vals.min()), vmax=float(F_vals.max()))
        cmap_obj = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        face_colors = cmap_obj(norm(F_vals))

        # === Plot the surface with facecolors ===
        surface = ax.plot_surface(
            X, Y, Z,
            facecolors=face_colors,
            rstride=1, cstride=1,
            edgecolor=edgecolor,
            linewidth=0.,
            antialiased=False,
            shade=False,  # Disable matplotlib auto-shading (since colormap provides color)
            alpha=alpha
        )

        # Add colorbar
        mappable = cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=float(F_vals.min()), vmax=float(F_vals.max())),
            cmap=cmap
        )
        mappable.set_array(F_vals)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
        cbar.set_label(label_function or "Function value", fontsize=12)

    else:
        surface = ax.plot_surface(
            X, Y, Z,
            color=color,
            rstride=1, cstride=1,
            edgecolor=edgecolor,
            alpha=alpha,
            antialiased=True,
            linewidth=0.2 if edgecolor != "none" else 0
        )

    # === Add label (optional) ===
    if label is not None:
        X_flat, Y_flat, Z_flat = X.ravel(), Y.ravel(), Z.ravel()

        if label_position == "center":
            pos = (jnp.mean(X_flat), jnp.mean(Y_flat), jnp.mean(Z_flat))
        elif label_position == "top":
            idx = jnp.argmax(Z_flat)
            pos = (X_flat[idx], Y_flat[idx], Z_flat[idx])
        elif label_position == "bottom":
            idx = jnp.argmin(Z_flat)
            pos = (X_flat[idx], Y_flat[idx], Z_flat[idx])
        elif label_position == "right":
            idx = jnp.argmax(X_flat)
            pos = (X_flat[idx], Y_flat[idx], Z_flat[idx])
        elif label_position == "left":
            idx = jnp.argmin(X_flat)
            pos = (X_flat[idx], Y_flat[idx], Z_flat[idx])
        else:
            raise ValueError(f"Invalid label_position: {label_position}")

        ax.text(
            pos[0], pos[1], pos[2],
            label,
            fontsize=label_fontsize,
            color="black" if function is not None else color
        )

    return surface

def plot_chart_region_manifold(
    ax,
    surface_manifold,
    chart,
    color,
    n_points=2000,
    linestyle='dashed',
    linewidth=1.0,
    fill_surface=True,
    alpha=0.15,
    edgecolor='none',
    label=None,
    label_position="center",
    label_fontsize=12
):
    """
    Plot a chart region on the manifold with boundary and optional label.

    Args:
        ax: Matplotlib 3D axis.
        surface_manifold: The manifold object with an .embed() method.
        chart: The Chart object defining the region.
        color: Color for boundary and surface.
        n_points: Approximate number of points to sample in the interior.
        linestyle: Boundary line style (e.g., 'dashed', 'solid').
        linewidth: Boundary line width.
        fill_surface: Whether to fill the region surface.
        alpha: Transparency of the surface fill.
        edgecolor: Edge color for triangles (None disables grid).
        label: LaTeX string for labeling the region (e.g., r"$U$"), or None.
        label_position: One of ["center", "top", "bottom", "left", "right"].
        label_fontsize: Font size for the label.
    """

    # === Sample interior points ===
    param_points = chart.sample_region_in_param_space(n_points=n_points)
    embedded = surface_manifold.embed(param_points)

    X_, Y_, Z_ = embedded[:, 0], embedded[:, 1], embedded[:, 2]

    # === Plot the surface patch using triangulation ===
    if fill_surface:
        ax.plot_trisurf(
            X_, Y_, Z_,
            color=color,
            alpha=alpha,
            linewidth=0.01 if edgecolor != 'none' else 0,
            edgecolor=edgecolor if edgecolor != 'none' else 'none',
            antialiased=True
        )

    # === Plot the boundary ===
    lambdas = jnp.linspace(0, 2 * jnp.pi, 300)
    boundary_param = chart.boundary_in_param_space(lambdas)
    boundary_embed = surface_manifold.embed(boundary_param)

    ax.plot(
        boundary_embed[:, 0],
        boundary_embed[:, 1],
        boundary_embed[:, 2],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth
    )

    # === Add label (optional) ===
    if label is not None:
        bx, by, bz = boundary_embed[:, 0], boundary_embed[:, 1], boundary_embed[:, 2]

        if label_position == "center":
            pos = (jnp.mean(bx), jnp.mean(by), jnp.mean(bz))
        elif label_position == "top":
            idx = jnp.argmax(bz)
            pos = (bx[idx], by[idx], bz[idx])
        elif label_position == "bottom":
            idx = jnp.argmin(bz)
            pos = (bx[idx], by[idx], bz[idx])
        elif label_position == "right":
            idx = jnp.argmax(bx)
            pos = (bx[idx], by[idx], bz[idx])
        elif label_position == "left":
            idx = jnp.argmin(bx)
            pos = (bx[idx], by[idx], bz[idx])
        else:
            raise ValueError(f"Invalid label_position: {label_position}")

        ax.text(
            pos[0], pos[1], pos[2],
            label,
            fontsize=label_fontsize,
            color=color
        )

def sample_polygon_interior(polygon, key, n_points=2000, max_attempts=10):
    """
    Uniformly sample points inside a shapely Polygon using JAX.

    Args:
        polygon: Shapely Polygon in param space.
        key: JAX PRNG key.
        n_points: Number of points to sample.

    Returns:
        Array of shape (N, 2) of valid samples inside the polygon.
    """

    minx, miny, maxx, maxy = polygon.bounds
    samples = []
    total_collected = 0
    attempts = 0

    while total_collected < n_points and attempts < max_attempts:
        attempts += 1
        key, key_x, key_y = jax.random.split(key, 3)

        x = jax.random.uniform(key_x, shape=(n_points,), minval=minx, maxval=maxx)
        y = jax.random.uniform(key_y, shape=(n_points,), minval=miny, maxval=maxy)

        points = jnp.stack([x, y], axis=-1)
        points_np = jnp.array(points).tolist()  # Convert to Python lists for shapely

        mask = jnp.array([polygon.contains(Point(p)) for p in points_np])

        accepted = points[mask]
        samples.append(accepted)

        total_collected += accepted.shape[0]

    if len(samples) == 0:
        raise RuntimeError("No points found inside the polygon.")

    samples = jnp.concatenate(samples, axis=0)[:n_points]
    return samples

def plot_chart_intersection(
    ax,
    surface_manifold,
    chart1,
    chart2,
    color='green',
    n_points=2000,
    linestyle='dashed',
    linewidth=1.0,
    fill_surface=True,
    alpha=0.2,
    edgecolor='none',
    label=r"$U \cap V$",
    label_position="center",
    label_fontsize=12
):
    """
    Plot the intersection region between two charts on the manifold.
    """

    intersection_poly = compute_chart_intersection_polygon(chart1, chart2)
    param_points = sample_polygon_interior(intersection_poly, key=jax.random.PRNGKey(42), n_points=n_points)
    embedded = surface_manifold.embed(param_points)

    X_, Y_, Z_ = embedded[:, 0], embedded[:, 1], embedded[:, 2]

    # --- Plot fill ---
    if fill_surface:
        ax.plot_trisurf(
            X_, Y_, Z_,
            color=color,
            alpha=alpha,
            linewidth=0.01 if edgecolor != 'none' else 0,
            edgecolor=edgecolor if edgecolor != 'none' else 'none',
            antialiased=True
        )

    # --- Plot boundary ---
    boundary = jnp.array(intersection_poly.exterior.coords)
    boundary_embed = surface_manifold.embed(jnp.array(boundary))

    ax.plot(
        boundary_embed[:, 0],
        boundary_embed[:, 1],
        boundary_embed[:, 2],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth
    )

    # --- Label ---
    if label is not None:
        bx, by, bz = boundary_embed[:, 0], boundary_embed[:, 1], boundary_embed[:, 2]

        if label_position == "center":
            pos = (jnp.mean(bx), jnp.mean(by), jnp.mean(bz))
        elif label_position == "top":
            idx = jnp.argmax(bz)
            pos = (bx[idx], by[idx], bz[idx])
        elif label_position == "bottom":
            idx = jnp.argmin(bz)
            pos = (bx[idx], by[idx], bz[idx])
        elif label_position == "right":
            idx = jnp.argmax(bx)
            pos = (bx[idx], by[idx], bz[idx])
        elif label_position == "left":
            idx = jnp.argmin(bx)
            pos = (bx[idx], by[idx], bz[idx])
        else:
            raise ValueError(f"Invalid label_position: {label_position}")

        ax.text(
            pos[0], pos[1], pos[2],
            label,
            fontsize=label_fontsize,
            color=color
        )

def plot_curve_on_manifold(
    ax,
    curve,
    lambda_range=(-jnp.pi, jnp.pi),
    n_points=300,
    color="black",
    linewidth=2.0,
    show_tangents=True,
    tangent_step=20,
    tangent_scale=0.4,
    tangent_style=None,
    tangent_method="finite_difference",  # or "autodiff"
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

    # === Plot tangent arrows ===
    if show_tangents:
        tangents = curve.tangent_vector_on_manifold(lambda_vals, method=tangent_method)
        norms = jnp.linalg.norm(tangents, axis=1, keepdims=True)
        unit_tangents = tangents / norms

        sampled_points = points[::tangent_step]
        sampled_tangents = unit_tangents[::tangent_step]

        arrow_style = dict(
            mutation_scale=1.0,
            arrowstyle="->,head_length=4.,head_width=2.",
            color=color,
            lw=1.5,
            alpha=0.7
        )
        if tangent_style is not None:
            arrow_style.update(tangent_style)

        for p, v in zip(sampled_points, sampled_tangents):
            ax.add_artist(Arrow3D(
                [float(p[0]), float(p[0] + v[0] * tangent_scale)],
                [float(p[1]), float(p[1] + v[1] * tangent_scale)],
                [float(p[2]), float(p[2] + v[2] * tangent_scale)],
                **arrow_style
            ))

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

def compute_label_position(coords, position="center", offset=(0.2, 0.2)):
    center = jnp.mean(coords, axis=0)

    if position == "center":
        return center
    elif position == "top":
        idx = np.argmax(coords[:, 1])
    elif position == "bottom":
        idx = np.argmin(coords[:, 1])
    elif position == "left":
        idx = np.argmin(coords[:, 0])
    elif position == "right":
        idx = np.argmax(coords[:, 0])
    else:
        raise ValueError(f"Unknown label position: {position}")

    point = coords[idx]
    return point + jnp.array(offset)

def plot_tangent_plane(ax, p_xyz, tangent_vec1, tangent_vec2,
                        size=1.0, color="lightgray", alpha=0.5):
    """
    Plot the tangent plane at point p_xyz.

    Args:
        ax: Matplotlib 3D axis.
        p_xyz: Point on manifold (3,).
        tangent_vec1: First tangent vector (3,).
        tangent_vec2: Second tangent vector (3,).
        size: Size scaling for the plane.
        color: Color of the plane.
        alpha: Transparency.
    """
    # Create grid in tangent plane coordinates
    s = np.linspace(-size, size, 10)
    t = np.linspace(-size, size, 10)
    S, T = np.meshgrid(s, t)

    # Parametric equation of plane
    plane = (
        p_xyz.reshape(3, 1, 1)
        + S * tangent_vec1.reshape(3, 1, 1)
        + T * tangent_vec2.reshape(3, 1, 1)
    )

    X, Y, Z = plane[0], plane[1], plane[2]

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor="none")

def plot_dual_plane_pair(
    ax,
    p_xyz,
    t1,
    t2,
    z_shift=1.0,
    scale=1.0,
    plane_color="darkorange",
    plane_alpha=0.35,
    label=r"$T_p^*\mathcal{M}$",
    label_offset=(0.4, 0.4, 0.15),
    label_fontsize=16,
    arrow_color="black",
    arrow_alpha=0.8,
    arrow_lw=1.0,
    arrow_mutation_scale=20
):
    """
    Plot a cotangent plane shifted from a tangent plane, with connecting arrows.

    Args:
        ax: Matplotlib 3D axis.
        p_xyz: Center point of the tangent plane (array-like, shape (3,)).
        t1, t2: Basis vectors of the tangent plane (array-like, shape (3,)).
        z_shift: Vertical shift along z to place the cotangent plane.
        scale: Size of the planes.
        plane_color: Color of the cotangent plane.
        plane_alpha: Transparency of the plane.
        label: Text label for the cotangent plane.
        label_offset: Offset for label positioning (dx, dy, dz).
        label_fontsize: Font size for label.
        arrow_color: Color of connecting arrows.
        arrow_alpha: Transparency of arrows.
        arrow_lw: Line width of arrows.
        arrow_mutation_scale: Arrow head scale.
    """
    # === Compute cotangent plane center ===
    p_cotangent = p_xyz + jnp.array([0.0, 0.0, z_shift])

    # === Plot cotangent plane ===
    plot_tangent_plane(ax, p_cotangent, t1, t2, size=scale, color=plane_color, alpha=plane_alpha)

    # === Add label ===
    ax.text(
        p_cotangent[0] + label_offset[0],
        p_cotangent[1] + label_offset[1],
        p_cotangent[2] + label_offset[2],
        label,
        fontsize=label_fontsize,
        color=plane_color
    )

    # === Compute corners of both planes ===
    corner_offsets = [
        -t1 * scale / 2 - t2 * scale / 2,
        -t1 * scale / 2 + t2 * scale / 2,
        +t1 * scale / 2 - t2 * scale / 2,
        +t1 * scale / 2 + t2 * scale / 2,
    ]
    corners_TpM = [p_xyz + offset for offset in corner_offsets]
    corners_TpM_star = [p_cotangent + offset for offset in corner_offsets]

    # === Plot connecting arrows ===
    for pt_from, pt_to in zip(corners_TpM, corners_TpM_star):
        ax.add_artist(Arrow3D(
            [pt_from[0], pt_to[0]],
            [pt_from[1], pt_to[1]],
            [pt_from[2], pt_to[2]],
            mutation_scale=arrow_mutation_scale,
            arrowstyle="->",
            lw=arrow_lw,
            color=arrow_color,
            alpha=arrow_alpha
        ))

def compute_chart_intersection_polygon(chart1, chart2):
    """
    Compute the intersection polygon (in parameter space) between two charts.

    Returns:
        A shapely Polygon representing the intersection region.
    """

    # === Sample boundaries ===
    lambda_vals = jnp.linspace(0, 2 * jnp.pi, 500)

    boundary1 = jnp.array(chart1.boundary_in_param_space(lambda_vals))
    boundary2 = jnp.array(chart2.boundary_in_param_space(lambda_vals))

    # === Ensure boundaries are closed ===
    if not jnp.allclose(boundary1[0], boundary1[-1]):
        boundary1 = jnp.vstack([boundary1, boundary1[0]])

    if not jnp.allclose(boundary2[0], boundary2[-1]):
        boundary2 = jnp.vstack([boundary2, boundary2[0]])

    poly1 = Polygon(boundary1).buffer(0)  # buffer(0) cleans invalidities
    poly2 = Polygon(boundary2).buffer(0)

    if not poly1.is_valid or not poly2.is_valid:
        raise ValueError("One or both input polygons are invalid even after cleaning.")

    # === Compute intersection ===
    intersection = poly1.intersection(poly2)

    if intersection.is_empty:
        raise ValueError("The two charts do not overlap.")

    if not isinstance(intersection, Polygon):
        raise ValueError(
            "Intersection resulted in multiple disjoint regions or a degenerate shape. "
            "Check if the input regions are correct and simply connected."
        )

    return intersection

def plot_chart_region_2D(
    ax, boundary,
    color="red", linestyle="dashed",
    linewidth=1.5, alpha=0.05,
    label=None, label_position="center", label_offset=(0.2, 0.2),
    label_fontsize=16
):

    ax.fill(boundary[:, 0], boundary[:, 1],
            color=color, alpha=alpha, zorder=1)
    ax.plot(boundary[:, 0], boundary[:, 1],
            color=color, linestyle=linestyle, linewidth=linewidth, zorder=2)

    if label is not None:
        label_pos = compute_label_position(
            boundary, position=label_position, offset=label_offset
        )
        ax.text(
            label_pos[0], label_pos[1], label,
            color=color, fontsize=label_fontsize,
            ha="center", va="center"
        )

def plot_intersection_region_2D(
    ax, intersection_coords_list,
    color="green", linestyle="dotted",
    linewidth=3.0, alpha=0.3,
    label=None, label_position="center", label_offset=(0.2, 0.2),
    label_fontsize=16
):
    """
    Plot the intersection region(s) given a list of boundary coordinate arrays.

    Args:
        intersection_coords_list: list of (N_i, 2) arrays.
    """
    for coords in intersection_coords_list:
        ax.fill(coords[:, 0], coords[:, 1],
                color=color, alpha=alpha, zorder=2)
        ax.plot(coords[:, 0], coords[:, 1],
                color=color, linestyle=linestyle, linewidth=linewidth, zorder=3)

    if label is not None:
        # Place label on the largest polygon
        largest = max(intersection_coords_list, key=lambda c: len(c))
        label_pos = compute_label_position(
            largest, position=label_position, offset=label_offset
        )
        ax.text(
            label_pos[0], label_pos[1], label,
            color=color, fontsize=label_fontsize,
            ha="center", va="center"
        )

def plot_curve_in_chart(
    ax, chart, curve,
    lambda_range=(-jnp.pi, jnp.pi), n_points=300,
    color="black", linewidth=2.0,
    label=None, label_position="end", label_offset=(0.2, 0.2),
    label_fontsize=16
):
    lambdas = jnp.linspace(lambda_range[0], lambda_range[1], n_points)
    points = chart.map_to_chart(curve.evaluate_in_param_space(lambdas))

    ax.plot(points[:, 0], points[:, 1],
            color=color, linewidth=linewidth, zorder=4)

    if label is not None:
        if label_position == "start":
            pos = points[0]
        elif label_position == "end":
            pos = points[-1]
        elif label_position == "center":
            pos = points[len(points) // 2]
        else:
            raise ValueError(f"Unknown label position: {label_position}")

        pos = pos + jnp.array(label_offset)

        ax.text(
            pos[0], pos[1], label,
            color=color, fontsize=label_fontsize,
            ha="center", va="center"
        )

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
        segments, colors=colors[:-1], linewidths=linewidth, alpha=0.95, zorder=4
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

def plot_tangent_vectors_in_chart(
    ax, chart, curve,
    lambda_range=(-jnp.pi, jnp.pi), n_points=300,
    step=30, scale=0.9,
    color="black", linewidth=1.2,
    head_width=0.15, head_length=0.2,
    zorder=4, length_includes_head=True,
    tangent_method="finite_difference",
):
    """
    Plot tangent vectors of a curve in chart coordinates (correctly transformed).

    Args:
        ax: Matplotlib axis.
        chart: Chart object.
        curve: Curve object.
        lambda_range: Range of lambda parameter for curve.
        n_points: Number of points to sample along the curve.
        step: Interval for sampling tangent vectors (density).
        scale: Scaling factor for arrow length.
        color: Color of arrows.
        linewidth: Arrow line width.
        head_width: Width of arrowhead.
        head_length: Length of arrowhead.
        zorder: Plot order.
        length_includes_head: Whether arrow length includes head.
        tangent_method: "finite_difference" (default) or "autodiff".
    """
    # === Sample lambdas ===
    lambdas = jnp.linspace(lambda_range[0], lambda_range[1], n_points)
    sampled_lambdas = lambdas[::step]

    # === Evaluate curve points in chart ===
    points_in_chart = chart.map_to_chart(curve.evaluate_in_param_space(sampled_lambdas))

    # === Tangents in parameter space ===
    tangents_in_param = curve.tangent_vector_in_param_space(
        sampled_lambdas, method=tangent_method
    )

    # === Compute Jacobian of chart map at each curve point ===
    def chart_map_fn(param_point):
        return chart.map_to_chart(param_point)

    jacobian_fn = jax.jacrev(chart_map_fn)
    jacobians = jax.vmap(jacobian_fn)(curve.evaluate_in_param_space(sampled_lambdas))
    # Shape (N, chart_dim, param_dim)

    # === Pushforward tangent: v_chart = Jacobian @ v_param
    tangents_in_chart = jnp.einsum('nij,nj->ni', jacobians, tangents_in_param)

    # === Normalize tangents for plotting
    norms = jnp.linalg.norm(tangents_in_chart, axis=1, keepdims=True)
    unit_tangents = tangents_in_chart / norms

    # === Plot arrows
    for p, v in zip(points_in_chart, unit_tangents):
        ax.arrow(
            p[0], p[1],
            v[0] * scale, v[1] * scale,
            head_width=head_width, head_length=head_length,
            fc=color, ec=color,
            linewidth=linewidth,
            zorder=zorder,
            length_includes_head=length_includes_head
        )

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

def plot_function_in_chart(
    ax,
    chart,
    function,
    param_sampling_bounds,
    chart_xlim,
    chart_ylim,
    resolution=200,
    cmap="viridis",
    alpha=0.8,
    label_function=None,
    shading="gouraud"
):
    """
    Plot the scalar function values over a chart using Delaunay triangulation.

    Args:
        ax: Matplotlib 2D axis.
        chart: Chart object.
        function: Function object defined on the manifold.
        param_sampling_bounds: ((xmin, xmax), (ymin, ymax)) in parameter space.
        chart_xlim: Plot limits in chart x direction.
        chart_ylim: Plot limits in chart y direction.
        resolution: Number of grid points per axis.
        cmap: Colormap.
        alpha: Transparency.
        label_function: Label for the colorbar (optional).
        shading: 'gouraud' (smooth) or 'flat'.
    """

    # === Build sampling grid in parameter space ===
    (x_min, x_max), (y_min, y_max) = param_sampling_bounds
    xv = jnp.linspace(x_min, x_max, resolution)
    yv = jnp.linspace(y_min, y_max, resolution)
    XX, YY = jnp.meshgrid(xv, yv)
    param_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    # === Map to chart coordinates ===
    chart_points = chart.map_to_chart(param_points)

    # === Evaluate function ===
    F_vals = function.evaluate_in_param_space(param_points)

    # === Triangulate in chart space ===
    triang = Triangulation(chart_points[:, 0], chart_points[:, 1])

    # === Plot ===
    tpc = ax.tripcolor(
        triang,
        F_vals,
        cmap=cmap,
        shading=shading,
        alpha=alpha
    )

    # === Colorbar ===
    cbar = plt.colorbar(tpc, ax=ax, shrink=0.8, pad=0.02)
    if label_function:
        cbar.set_label(label_function, fontsize=12)

    # === Plot limits ===
    ax.set_xlim(chart_xlim)
    ax.set_ylim(chart_ylim)

    return tpc