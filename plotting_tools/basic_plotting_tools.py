import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Patch
import jax
import jax.numpy as jnp
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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
    xmin=-jnp.pi,
    xmax=jnp.pi,
    ymin=-jnp.pi,
    ymax=jnp.pi,
    color="skyblue",
    alpha=0.2,
    edgecolor="none",
    label=None,
    label_position="center",
    label_fontsize=12
):
    """
    Plot the manifold surface over a rectangular region in parameter space.

    Args:
        ax: Matplotlib 3D axis.
        surface_manifold: The manifold object with an .embed() method.
        resolution: Number of points per axis (int).
        xmin, xmax: Bounds in x param direction.
        ymin, ymax: Bounds in y param direction.
        color: Surface color.
        alpha: Surface transparency.
        edgecolor: Edge color of the mesh (use 'none' to disable grid lines).
        label: Optional LaTeX string to label the surface (e.g., r"$\mathcal{M}$"), or None.
        label_position: One of ["center", "top", "bottom", "left", "right"].
        label_fontsize: Font size for the label.

    Returns:
        The plotted surface object (from ax.plot_surface()).
    """

    # === Create grid in parameter space ===
    x = jnp.linspace(xmin, xmax, resolution)
    y = jnp.linspace(ymin, ymax, resolution)
    X, Y = jnp.meshgrid(x, y)

    param_points = jnp.stack([X.flatten(), Y.flatten()], axis=-1)

    # === Evaluate manifold embedding ===
    Z = surface_manifold.embed(param_points)[:, 2].reshape(X.shape)

    # === Plot surface ===
    surface = ax.plot_surface(
        X, Y, Z,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor
    )

    # === Add label (optional) ===
    if label is not None:
        # Flatten data for label positioning
        X_flat, Y_flat, Z_flat = X.flatten(), Y.flatten(), Z.flatten()

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
            color=color
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
    return_legend_handle=False
):
    """
    Plot a vector field on the manifold as arrows.

    Args:
        ax: Matplotlib 3D axis.
        field: A Field object defined on the manifold.
        grid_resolution: Number of points along each parameter axis.
        xlim, ylim: Tuple limits in parameter space.
        vector_scale: Scale factor for arrow lengths.
        color: Fixed color (ignored if colormap is provided).
        colormap: Matplotlib colormap name or object (optional).
        arrow_style: Dict with additional Arrow3D style parameters.
        label: Optional label (e.g., r"$\chi$").
        label_fontsize: Font size for label.
        return_legend_handle: If True, returns a legend handle (Patch).
    Returns:
        Patch handle for legend if return_legend_handle is True, else None.
    """

    # === Build meshgrid in parameter space ===
    x_vals = jnp.linspace(xlim[0], xlim[1], grid_resolution)
    y_vals = jnp.linspace(ylim[0], ylim[1], grid_resolution)
    XX, YY = jnp.meshgrid(x_vals, y_vals)
    param_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    # === Compute points and vectors on manifold ===
    points_on_manifold = field.manifold.embed(param_points)  # (N, 3)
    vectors_ambient = field.evaluate_on_manifold(param_points)  # (N, 3)

    # Normalize vectors
    norms = jnp.linalg.norm(vectors_ambient, axis=1, keepdims=True)
    unit_vectors = vectors_ambient / norms

    # === Prepare colormap ===
    if colormap is not None:
        cmap = cm.get_cmap(colormap) if isinstance(colormap, str) else colormap
        num_points = param_points.shape[0]
        colors = cmap(np.linspace(0, 1, num_points))
        color_sample = cmap(0.6)
    else:
        colors = [color] * param_points.shape[0]
        color_sample = color

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

    # === Create legend handle ===
    if label is not None and return_legend_handle:
        legend_patch = Patch(facecolor=color_sample, edgecolor="k", label=label)
        return legend_patch

    return None

def plot_vector_field_in_chart(
    ax,
    points_in_chart,
    vectors_in_chart,
    vector_scale=0.3,
    color="purple",
    colormap=None,
    arrow_style=None,
    label=None,
    label_fontsize=14,
    return_legend_handle=False
):
    """
    Plot a vector field in chart coordinates (2D).

    Args:
        ax: Matplotlib 2D axis.
        points_in_chart: Array (N, 2) of positions in chart coordinates.
        vectors_in_chart: Array (N, 2) of corresponding vector components.
        vector_scale: Scale factor for arrow length.
        color: Arrow color (ignored if colormap is provided).
        colormap: Matplotlib colormap name or object (optional).
        arrow_style: Dict of arrow style parameters.
        label: Optional label (e.g. r"$\chi$").
        label_fontsize: Font size for the label.
        return_legend_handle: If True, returns a legend handle (Patch).
    Returns:
        Patch handle for legend if return_legend_handle is True, else None.
    """

    N = points_in_chart.shape[0]

    # === Colormap handling ===
    if colormap is not None:
        cmap = cm.get_cmap(colormap) if isinstance(colormap, str) else colormap
        colors = cmap(np.linspace(0, 1, N))
        color_sample = cmap(0.6)
    else:
        colors = [color] * N
        color_sample = color

    # === Arrow style defaults ===
    default_arrow_style = dict(
        head_width=0.1,
        head_length=0.15,
        linewidth=0.8,
        alpha=0.6,
        length_includes_head=True
    )
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

    # === Return legend handle if needed ===
    if label is not None and return_legend_handle:
        legend_patch = Patch(facecolor=color_sample, edgecolor="k", label=label)
        return legend_patch

    return None
