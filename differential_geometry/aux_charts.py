from typing import Callable, Optional
import numpy as np
import jax.numpy as jnp
from matplotlib.path import Path
from shapely.geometry import Polygon

def compute_chart_intersection(boundary_U, boundary_V) -> np.ndarray:
    """
    Compute the intersection region between two chart domains in parameter space.

    Supports:
        - 1D: boundaries are (2,) tuples or arrays representing [t_min, t_max]
        - 2D: boundaries are (N, 2) and (M, 2) arrays representing closed polygons

    Returns:
        - For 1D: (2,) array with intersection interval or (0,) if disjoint
        - For 2D: (K, 2) array with intersection polygon or (0, 2) if disjoint
    """
    # --- 1D CASE ---
    if isinstance(boundary_U, (list, tuple, np.ndarray)) and np.shape(boundary_U) == (2,) \
       and np.shape(boundary_V) == (2,):
        
        t1_min, t1_max = sorted(boundary_U)
        t2_min, t2_max = sorted(boundary_V)

        t_min = max(t1_min, t2_min)
        t_max = min(t1_max, t2_max)

        if t_min < t_max:
            return (t_min, t_max)
        else:
            return ()  # No intersection

    # --- 2D CASE ---
    boundary_U = np.asarray(boundary_U)
    boundary_V = np.asarray(boundary_V)

    # --- 2D CASE ---
    boundary_U = np.asarray(boundary_U)
    boundary_V = np.asarray(boundary_V)

    def ensure_closed(boundary):
        if not np.allclose(boundary[0], boundary[-1]):
            boundary = np.vstack([boundary, boundary[0]])
        return boundary

    if boundary_U.ndim == 2 and boundary_U.shape[1] == 2 and \
       boundary_V.ndim == 2 and boundary_V.shape[1] == 2:

        boundary_U = ensure_closed(boundary_U)
        boundary_V = ensure_closed(boundary_V)

        poly_U = Polygon(boundary_U).buffer(0)  # 🛠 FIX: Clean geometry
        poly_V = Polygon(boundary_V).buffer(0)

        if not poly_U.is_valid:
            raise ValueError("boundary_U produced an invalid polygon")
        if not poly_V.is_valid:
            raise ValueError("boundary_V produced an invalid polygon")

        intersection = poly_U.intersection(poly_V)

        if intersection.is_empty:
            return np.zeros((0, 2))
        elif intersection.geom_type == 'Polygon':
            return np.array(intersection.exterior.coords)
        elif intersection.geom_type == 'MultiPolygon':
            largest = max(intersection.geoms, key=lambda g: g.area)
            return np.array(largest.exterior.coords)
        else:
            raise RuntimeError(f"Unexpected geometry type: {intersection.geom_type}")
    
    raise ValueError("Unsupported boundary formats. Expected 1D intervals or 2D polygons.")

def polar_region_sampler_from_boundary(
    boundary: np.ndarray,
    center: Optional[np.ndarray] = None,
) -> Callable[[int], jnp.ndarray]:
    """
    Return a region_sampler(n_points) that adapts polar mesh resolution to n_points.
    """
    if boundary.ndim != 2 or boundary.shape[1] != 2:
        raise ValueError("Boundary must have shape (N, 2)")

    if center is None:
        center = np.mean(boundary, axis=0)
    else:
        center = np.asarray(center)
        if center.shape != (2,):
            raise ValueError("Provided center must be a 1D array of shape (2,)")

    boundary_path = Path(boundary)

    def region_sampler(n_points: int) -> jnp.ndarray:
        # Determine radial and angular resolution heuristically
        n_radial = max(3, int(np.sqrt(n_points) // 2))
        n_angular = max(6, int(n_points // n_radial))

        # Estimate max radius
        vectors = boundary - center
        radii = np.linalg.norm(vectors, axis=1)
        max_radius = np.max(radii)

        # Build mesh
        r = np.linspace(0.0, 1.0, n_radial + 1)[1:] * max_radius
        theta = np.linspace(0.0, 2 * np.pi, n_angular, endpoint=False)
        R, Theta = np.meshgrid(r, theta, indexing="ij")

        x = R * np.cos(Theta) + center[0]
        y = R * np.sin(Theta) + center[1]
        grid_points = np.stack([x, y], axis=-1).reshape(-1, 2)

        # Filter only inside
        mask = boundary_path.contains_points(grid_points)
        inside_points = grid_points[mask]

        # Downsample if needed
        if len(inside_points) > n_points:
            idx = np.random.choice(len(inside_points), size=n_points, replace=False)
            inside_points = inside_points[idx]

        return jnp.array(inside_points)

    return region_sampler


def boundary_ellipse_in_param_space(lambdas, center=(1.0, 0.0), axes=(1.5, 1.0)):
    """
    Parametric boundary of an ellipse in parameter space.

    Args:
        lambdas (array): Angles [0, 2π] parameterizing the ellipse.
        center (tuple): (x_center, y_center) of the ellipse.
        axes (tuple): (a, b) semi-axes along x and y directions.

    Returns:
        Array of shape (N, 2) with (x, y) boundary points.
    """
    x_center, y_center = center
    a, b = axes
    x = a * jnp.cos(lambdas) + x_center
    y = b * jnp.sin(lambdas) + y_center
    return jnp.stack([x, y], axis=-1)

def boundary_rectangle_in_param_space(
    lambdas: jnp.ndarray,
    xlim: tuple = (0.0, 1.0),
    ylim: tuple = (0.0, 1.0),
) -> jnp.ndarray:
    """
    Parametric boundary of a rectangle in parameter space (closed curve).

    This uses the input lambdas ∈ [0, 4), mapped to the rectangle's perimeter.

    Args:
        lambdas: 1D array of parameter values in [0, 4).
        xlim: Tuple (x_min, x_max), x-range of the rectangle.
        ylim: Tuple (y_min, y_max), y-range of the rectangle.

    Returns:
        Array of shape (N+1, 2) with (x, y) boundary points, closed (last point == first).
    """
    x_min, x_max = xlim
    y_min, y_max = ylim

    lambdas = lambdas % 4.0  # Ensure periodicity

    x = jnp.zeros_like(lambdas)
    y = jnp.zeros_like(lambdas)

    # Edge 1: Bottom
    mask1 = (lambdas >= 0) & (lambdas < 1)
    x = x.at[mask1].set(x_min + (x_max - x_min) * (lambdas[mask1] - 0.0))
    y = y.at[mask1].set(y_min)

    # Edge 2: Right
    mask2 = (lambdas >= 1) & (lambdas < 2)
    x = x.at[mask2].set(x_max)
    y = y.at[mask2].set(y_min + (y_max - y_min) * (lambdas[mask2] - 1.0))

    # Edge 3: Top
    mask3 = (lambdas >= 2) & (lambdas < 3)
    x = x.at[mask3].set(x_max - (x_max - x_min) * (lambdas[mask3] - 2.0))
    y = y.at[mask3].set(y_max)

    # Edge 4: Left
    mask4 = (lambdas >= 3) & (lambdas < 4)
    x = x.at[mask4].set(x_min)
    y = y.at[mask4].set(y_max - (y_max - y_min) * (lambdas[mask4] - 3.0))

    curve = jnp.stack([x, y], axis=-1)

    # Ensure the curve is closed
    curve_closed = jnp.vstack([curve, curve[0:1]])

    return curve_closed


def cartesian_chart_map(params):
    """
    Chart map for U: Identity map (x, y) → (x, y).
    """
    return params

def inverse_cartesian_chart_map(chart_coords):
    """
    Inverse of the identity map (x, y) → (x, y).
    """
    return chart_coords  # trivial identity

def polar_chart_map(params, center=(0.0, 0.0)):
    """
    Chart map for V: Maps (x, y) in param space to (r, θ) in polar coordinates.

    Args:
        params: Array of shape (N, 2) or (2,) — parameter points.
        center: (x_center, y_center) — pole of the polar chart.

    Returns:
        Array of shape (N, 2) or (2,) with (r, θ) in chart space.
        Returns NaNs for points that are ill-defined.
    """
    params = jnp.atleast_2d(params)

    x = params[:, 0] - center[0]
    y = params[:, 1] - center[1]

    # Check validity: finite x and y
    valid = jnp.isfinite(x) & jnp.isfinite(y)

    # Compute r and theta normally
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)

    # Where invalid, set r and theta to NaN
    r = jnp.where(valid, r, jnp.nan)
    theta = jnp.where(valid, theta, jnp.nan)

    result = jnp.stack([r, theta], axis=-1)

    return result if params.shape[0] > 1 else result[0]

def inverse_polar_chart_map(chart_coords, center=(0.0, 0.0)):
    """
    Inverse of the polar chart map: (r, θ) → (x, y).

    Args:
        chart_coords: Array of shape (N, 2) or (2,) with (r, θ)
        center: (x_center, y_center) — origin of the polar chart

    Returns:
        Param space point(s) (x, y)
    """
    chart_coords = jnp.atleast_2d(chart_coords)

    r = chart_coords[:, 0]
    theta = chart_coords[:, 1]

    x = r * jnp.cos(theta) + center[0]
    y = r * jnp.sin(theta) + center[1]

    result = jnp.stack([x, y], axis=-1)
    return result if chart_coords.shape[0] > 1 else result[0]

