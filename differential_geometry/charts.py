from typing import Callable, Optional, Union, Tuple
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from shapely.geometry import Polygon

class Chart:
    def __init__(
        self,
        name: str,
        manifold,
        boundary: Union[Tuple[float, float], np.ndarray],
        chart_map: Callable[[jnp.ndarray], jnp.ndarray],
        region_sampler: Optional[Callable[[int], jnp.ndarray]] = None,
    ):
        """
        Initialize a coordinate chart on a manifold.

        Args:
            name: Chart name (for reference).
            manifold: A Manifold object.
            boundary: 
                - If param_dim == 1: tuple (t_min, t_max) for interval in parameter space.
                - If param_dim == 2: array of shape (N, 2), closed non-self-intersecting curve.
            chart_map: Function from parameter space to chart coordinates.
            region_sampler: Optional sampling function; otherwise a default is selected.
        """
        self.name = name
        self.manifold = manifold
        self.chart_map = chart_map

        param_dim = manifold.param_dim
        ambient_dim = manifold.ambient_dim

        # Enforce supported dimensionalities
        supported_dims = {
            (1, 2),
            (1, 3),
            (2, 3),
        }
        if (param_dim, ambient_dim) not in supported_dims:
            raise ValueError(
                f"Unsupported chart configuration: (param_dim={param_dim}, ambient_dim={ambient_dim})\n"
                f"Allowed: {supported_dims}"
            )

        # Check boundary object
        if param_dim == 1:
            if not (isinstance(boundary, tuple) and len(boundary) == 2):
                raise ValueError(
                    "For 1D charts, boundary must be a tuple (t_min, t_max)"
                )
            self.boundary_interval = boundary
        elif param_dim == 2:
            boundary = np.asarray(boundary)
            if boundary.ndim != 2 or boundary.shape[1] != 2:
                raise ValueError(
                    "For 2D charts, boundary must be a 2D array of shape (N, 2)"
                )
            if not np.allclose(boundary[0], boundary[-1], atol=1e-6):
                raise ValueError(
                    "Boundary curve for 2D chart must be closed (first and last points must match)"
                )
            self.boundary_curve = boundary
        else:
            raise NotImplementedError("Only param_dim = 1 or 2 are currently supported.")

        # Select appropriate region sampler
        if region_sampler is not None:
            self.region_sampler = region_sampler
        else:
            self.region_sampler = (
                self._default_region_sampler_1d()
                if param_dim == 1
                else self._default_region_sampler_2d()
            )

    def map_to_chart(self, param_points: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the chart map to points in parameter space.

        Args:
            param_points: Array of shape (..., param_dim)

        Returns:
            Array of shape (..., chart_dim)
        """
        param_points = jnp.asarray(param_points)
        if param_points.shape[-1] != self.manifold.param_dim:
            raise ValueError(
                f"Expected param_points with last dim = {self.manifold.param_dim}, got {param_points.shape[-1]}"
            )
        return self.chart_map(param_points)

    def boundary_in_param_space(self, lambdas: jnp.ndarray) -> np.ndarray:
        """
        Return the boundary in parameter space, ensuring closure.

        → Only defined for 2D parameter space (closed curve).

        Returns:
            Array of shape (N+1, 2) with the first point repeated at the end.
        """
        if self.manifold.param_dim != 2:
            raise RuntimeError(
                "boundary_in_param_space is only defined for 2D parameter spaces."
            )
        boundary = np.array(self.boundary_curve)
        if not np.allclose(boundary[0], boundary[-1]):
            boundary = np.vstack([boundary, boundary[0]])
        return boundary

    def _default_region_sampler_2d(self) -> Callable[[int], jnp.ndarray]:
        """
        Region sampler for param_dim = 2 using Delaunay triangulation of the boundary.

        Returns:
            Function taking n_points and returning (N, 2) samples in parameter space.
        """
        def sampler(n_points):
            boundary = self.boundary_in_param_space(None)  # No lambdas needed anymore
            tri = Delaunay(boundary)

            simplices = tri.simplices
            vertices = boundary[simplices]  # shape (N_tri, 3, 2)

            n_tri = vertices.shape[0]
            points_per_triangle = max(n_points // n_tri, 1)

            def sample_triangle(tri_vertices):
                u = np.random.rand(points_per_triangle, 1)
                v = np.random.rand(points_per_triangle, 1)
                mask = (u + v) > 1
                u[mask] = 1 - u[mask]
                v[mask] = 1 - v[mask]
                w = 1 - (u + v)
                return (
                    w * tri_vertices[0]
                    + u * tri_vertices[1]
                    + v * tri_vertices[2]
                )

            sampled = np.vstack([sample_triangle(tri) for tri in vertices])
            return jnp.array(sampled)

        return sampler

    def _default_region_sampler_1d(self) -> Callable[[int], jnp.ndarray]:
        """
        Uniform sampler over a 1D interval in parameter space.

        Returns:
            Function taking n_points and returning array of shape (n_points, 1)
        """
        def sampler(n_points):
            t_min, t_max = self.boundary_interval
            samples = jnp.linspace(t_min, t_max, n_points)
            return samples.reshape(-1, 1)

        return sampler

    # ================================================================
    # === Inverse map (numerical) ====================================
    # ================================================================

    def sample_region_in_param_space(
        self, n_points: int = 2000
    ) -> jnp.ndarray:
        """
        Sample a set of points inside the region U in parameter space.

        Args:
            n_points: Approximate number of points to sample.

        Returns:
            Array of shape (N, param_dim)
        """
        return self.region_sampler(n_points)

    def inverse_map(self, x_target, initial_guess=None):
        """
        Numerically invert the chart map to find param_point such that:
            chart_map(param_point) ≈ x_target

        Args:
            x_target: Array of shape (..., chart_dim) — point(s) in chart coordinates.
            initial_guess: Optional array of shape (..., param_dim)

        Returns:
            param_points: Array of shape (..., param_dim)
        """
        x_target = jnp.atleast_2d(jnp.array(x_target))

        param_dim = self.manifold.param_dim
        chart_dim = x_target.shape[-1]

        # Initial guess
        if initial_guess is None:
            initial_guess = jnp.mean(self.sample_region_in_param_space(500), axis=0)
            initial_guess = jnp.broadcast_to(initial_guess, (x_target.shape[0], param_dim))

        results = []
        for xt, ig in zip(x_target, initial_guess):
            def objective(u):
                x = jnp.array(self.map_to_chart(jnp.array(u)))
                return jnp.sum((x - xt) ** 2)

            result = minimize(objective, x0=ig, method='L-BFGS-B')
            if not result.success:
                raise RuntimeError(f"Inversion failed at point {xt}: {result.message}")
            results.append(result.x)

        return jnp.array(results)  # shape (N, param_dim)


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

