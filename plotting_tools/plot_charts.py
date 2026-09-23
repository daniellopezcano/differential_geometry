import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay

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
    linewidth_fill=0.5,
    label=None,
    label_position="center",
    label_fontsize=12,
    draw_endpoints=True,
    endpoint_color="black",
    endpoint_size=30,
):
    """
    Generalized function to plot a chart region on a manifold.

    Works for:
        - 1D → 2D curves
        - 1D → 3D curves
        - 2D → 3D surfaces

    Args:
        ax: Matplotlib axis (2D or 3D).
        surface_manifold: Manifold instance with embed().
        chart: Chart instance.
        color: Color of chart region.
        n_points: Number of interior points to sample.
        linestyle: Line style for boundary/curve.
        linewidth: Line width.
        fill_surface: Whether to fill surface patch (only for 2D charts).
        alpha: Transparency of surface or curve.
        edgecolor: Edge color for 2D patch triangulation.
        linewidth_fill: Line width for filled surface edges (if fill_surface=True).
        label: Optional LaTeX string.
        label_position: One of ["center", "top", "bottom", "left", "right"].
        label_fontsize: Font size for label.
        draw_endpoints: If True, marks the endpoints (for 1D).
        endpoint_color: Color of endpoints (for 1D).
        endpoint_size: Size of endpoint markers (for 1D).
    """
    dim = chart.manifold.dim
    ambient_dim = chart.manifold.ambient_dim

    # === Sample parameter region ===
    param_points = chart.sample_region_in_param_space(n_points=n_points)
    embedded = surface_manifold.embed(param_points)

    # === 1D case: draw embedded curve ===
    if dim == 1:
        if ambient_dim == 2:
            ax.plot(
                embedded[:, 0],
                embedded[:, 1],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha
            )
        elif ambient_dim == 3:
            ax.plot(
                embedded[:, 0],
                embedded[:, 1],
                embedded[:, 2],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha
            )

        # Optional: mark endpoints
        if draw_endpoints:
            start_pt, end_pt = embedded[0], embedded[-1]
            if ambient_dim == 2:
                ax.scatter(*start_pt, color=endpoint_color, s=endpoint_size)
                ax.scatter(*end_pt, color=endpoint_color, s=endpoint_size)
            else:
                ax.scatter(*start_pt, color=endpoint_color, s=endpoint_size)
                ax.scatter(*end_pt, color=endpoint_color, s=endpoint_size)

        # Optional: label
        if label is not None:
            center_idx = len(embedded) // 2
            pos = embedded[center_idx]
            if ambient_dim == 2:
                ax.text(pos[0], pos[1], label, fontsize=label_fontsize, color=color)
            else:
                ax.text(pos[0], pos[1], pos[2], label, fontsize=label_fontsize, color=color)

    # === 2D case: draw surface patch and boundary ===
    elif dim == 2 and ambient_dim == 3:
        X_, Y_, Z_ = embedded[:, 0], embedded[:, 1], embedded[:, 2]

        if fill_surface:
            ax.plot_trisurf(
                X_, Y_, Z_,
                color=color,
                alpha=alpha,
                linewidth=linewidth_fill,
                edgecolor=edgecolor if edgecolor != 'none' else 'none',
                antialiased=True
            )

        # Boundary
        lambdas = jnp.linspace(0, 2 * jnp.pi, 300)
        try:
            boundary_param = chart.boundary_in_param_space(lambdas)
        except RuntimeError:
            boundary_param = chart.boundary_curve  # fallback if already evaluated
        boundary_embed = surface_manifold.embed(boundary_param)

        ax.plot(
            boundary_embed[:, 0],
            boundary_embed[:, 1],
            boundary_embed[:, 2],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth
        )

        # Label
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

    else:
        raise NotImplementedError(
            f"plot_chart_region_manifold not implemented for (dim={dim}, ambient_dim={ambient_dim})"
        )

def _identify_boundary_edges(simplices):
    """
    Identify boundary edges from a set of Delaunay simplices.

    Returns:
        A (M, 2) array of unique boundary edges.
    """
    edges = {}
    for tri in simplices:
        for i in range(3):
            edge = tuple(sorted((tri[i], tri[(i + 1) % 3])))
            if edge in edges:
                edges[edge] += 1
            else:
                edges[edge] = 1
    boundary_edges = np.array([edge for edge, count in edges.items() if count == 1])
    return boundary_edges

def _reorder_boundary_vertices(boundary_edges, start_idx=None):
    """
    Reorder boundary edges into a continuous loop of vertex indices.

    Args:
        boundary_edges: (M, 2) array of edge vertex index pairs.
        start_idx: Optional starting vertex index.

    Returns:
        (K,) array of ordered vertex indices forming the boundary loop.
    """
    # Build adjacency map
    adjacency = {}
    for i, j in boundary_edges:
        adjacency.setdefault(i, []).append(j)
        adjacency.setdefault(j, []).append(i)

    # Find a starting point
    if start_idx is None:
        start_idx = boundary_edges[0][0]

    ordered = [start_idx]
    visited = {start_idx}
    current = start_idx

    while True:
        neighbors = [n for n in adjacency[current] if n not in visited]
        if not neighbors:
            break
        next_vertex = neighbors[0]
        ordered.append(next_vertex)
        visited.add(next_vertex)
        current = next_vertex

    return np.array(ordered)


def plot_parametric_region_in_chart_coordinates(
    ax,
    chart,
    param_space_data: np.ndarray,
    color="red",
    linestyle="dashed",
    linewidth=1.5,
    alpha=0.2,
    label=None,
    label_position="center",
    label_fontsize=14,
    tessellated=False,
    simplices=None,
    tessellation_edgecolor="none",
    tessellation_linewidth=0.1,
    mark_boundary_points=False,
    boundary_marker_color="black",
    boundary_marker_size=10,
):
    """
    Plot the image under a chart map of a region in parameter space.

    Args:
        ax: Matplotlib 2D axis.
        chart: A Chart object (provides chart_map).
        param_space_data: Array of shape (N, 2). Interpreted as:
            - Interior samples (if tessellated=True)
            - Boundary points (if tessellated=False)
        color: Fill color.
        linestyle: Line style (used if tessellated=False).
        linewidth: Line width (used if tessellated=False).
        alpha: Fill transparency.
        label: Optional label string.
        label_position: "center", "top", "bottom", "left", "right".
        label_fontsize: Font size.
        tessellated: If True, uses triangulation.
        simplices: Optional Delaunay simplices on param_space_data.
        tessellation_edgecolor: Triangle edge color.
        tessellation_linewidth: Triangle edge width.
        mark_boundary_points: If True, highlights projected boundary vertices.
        boundary_marker_color: Color of boundary markers.
        boundary_marker_size: Size of boundary markers.
    """
    chart_map = chart.chart_map
    param_space_data = np.asarray(param_space_data)

    if chart.manifold.dim != 2:
        raise ValueError("This function only supports 2D parameter spaces.")

    if tessellated:
        if simplices is None:
            tri = Delaunay(param_space_data)
            simplices = tri.simplices

        triangles = param_space_data[simplices]  # (M, 3, 2)
        mapped_triangles = jax.vmap(chart_map)(jnp.array(triangles))  # (M, 3, 2)
        valid_mask = ~jnp.isnan(mapped_triangles).any(axis=(1, 2))
        mapped_triangles = mapped_triangles[valid_mask]

        for tri_mapped in mapped_triangles:
            ax.fill(
                tri_mapped[:, 0],
                tri_mapped[:, 1],
                color=color,
                alpha=alpha,
                edgecolor=tessellation_edgecolor,
                linewidth=tessellation_linewidth
            )

        # --- Identify boundary edges and reorder them ---
        boundary_edges = _identify_boundary_edges(simplices)
        boundary_vertex_indices = _reorder_boundary_vertices(boundary_edges)
        boundary_coords_param = param_space_data[boundary_vertex_indices]
        boundary_coords_chart = chart_map(jnp.array(boundary_coords_param))

        # --- Remove any NaNs before plotting ---
        valid = ~jnp.isnan(boundary_coords_chart).any(axis=1)
        boundary_coords_chart = boundary_coords_chart[valid]

        # --- Optional scatter markers ---
        if mark_boundary_points:
            ax.scatter(
                boundary_coords_chart[:, 0],
                boundary_coords_chart[:, 1],
                color=boundary_marker_color,
                s=boundary_marker_size,
                zorder=10
            )

        # --- Proper boundary line ---
        ax.plot(
            boundary_coords_chart[:, 0],
            boundary_coords_chart[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=3
        )

        if label is not None and len(mapped_triangles) > 0:
            all_pts = mapped_triangles.reshape(-1, 2)
            x, y = all_pts[:, 0], all_pts[:, 1]

            if label_position == "center":
                pos = (jnp.mean(x), jnp.mean(y))
            elif label_position == "top":
                idx = jnp.argmax(y)
                pos = (x[idx], y[idx])
            elif label_position == "bottom":
                idx = jnp.argmin(y)
                pos = (x[idx], y[idx])
            elif label_position == "right":
                idx = jnp.argmax(x)
                pos = (x[idx], y[idx])
            elif label_position == "left":
                idx = jnp.argmin(x)
                pos = (x[idx], y[idx])
            else:
                raise ValueError(f"Invalid label_position: {label_position}")

            ax.text(
                pos[0], pos[1], label,
                fontsize=label_fontsize, color=color,
                ha="center", va="center"
            )

    else:
        chart_image = chart_map(jnp.array(param_space_data))
        chart_image = np.array(chart_image)

        ax.fill(
            chart_image[:, 0],
            chart_image[:, 1],
            color=color,
            alpha=alpha,
            zorder=1
        )
        ax.plot(
            chart_image[:, 0],
            chart_image[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=2
        )

        if label is not None:
            x, y = chart_image[:, 0], chart_image[:, 1]

            if label_position == "center":
                pos = (jnp.mean(x), jnp.mean(y))
            elif label_position == "top":
                idx = jnp.argmax(y)
                pos = (x[idx], y[idx])
            elif label_position == "bottom":
                idx = jnp.argmin(y)
                pos = (x[idx], y[idx])
            elif label_position == "right":
                idx = jnp.argmax(x)
                pos = (x[idx], y[idx])
            elif label_position == "left":
                idx = jnp.argmin(x)
                pos = (x[idx], y[idx])
            else:
                raise ValueError(f"Invalid label_position: {label_position}")

            ax.text(
                pos[0], pos[1], label,
                fontsize=label_fontsize, color=color,
                ha="center", va="center"
            )
