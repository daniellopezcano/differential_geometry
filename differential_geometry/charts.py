from typing import Callable, Optional
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay


class Chart:
    """
    Represents a coordinate chart (U, X_map) for a manifold.

    - U is defined by a closed boundary curve in parameter space.
    - X_map is the coordinate map from U to R^d (chart coordinates).
    """

    def __init__(
        self,
        name: str,
        manifold,
        boundary_curve: Callable[[jnp.ndarray], jnp.ndarray],
        chart_map: Callable[[jnp.ndarray], jnp.ndarray],
        region_sampler: Optional[Callable[[int], jnp.ndarray]] = None,
    ):
        self.name = name
        self.manifold = manifold
        self.boundary_curve = boundary_curve
        self.chart_map = chart_map
        self.region_sampler = region_sampler or self._default_region_sampler()

    # ================================================================
    # === Core mappings ==============================================
    # ================================================================

    def map_to_chart(self, param_points: jnp.ndarray) -> jnp.ndarray:
        return self.chart_map(param_points)

    def boundary_in_param_space(self, lambdas: jnp.ndarray) -> jnp.ndarray:
        return self.boundary_curve(lambdas)

    # ================================================================
    # === New: Safe boundary (closed) ================================
    # ================================================================

    def boundary_in_param_space_closed(self, lambdas: jnp.ndarray) -> np.ndarray:
        """
        Return the boundary in parameter space, ensuring closure.

        → The returned array has shape (N+1, 2) with the first point repeated at the end.
        """
        boundary = np.array(self.boundary_in_param_space(lambdas))
        if not np.allclose(boundary[0], boundary[-1]):
            boundary = np.vstack([boundary, boundary[0]])
        return boundary

    def boundary_in_chart_coordinates_closed(self, lambdas: jnp.ndarray) -> np.ndarray:
        """
        Return the boundary mapped to chart coordinates, ensuring closure.

        → The returned array has shape (N+1, 2) with the first point repeated at the end.
        """
        boundary = self.boundary_in_param_space_closed(lambdas)
        chart_boundary = np.array(self.map_to_chart(jnp.array(boundary)))
        return chart_boundary

    # ================================================================
    # === Region sampling ============================================
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

    def _default_region_sampler(self) -> Callable[[int], jnp.ndarray]:
        """
        General region sampler based on boundary tessellation.

        → Works for arbitrary simply connected regions.
        → Uses Delaunay triangulation of the boundary.
        → Samples uniformly within triangles.

        Returns:
            Function taking (n_points) and returning (N, 2) samples.
        """
        def sampler(n_points):
            lambda_vals = jnp.linspace(0, 2 * jnp.pi, 600)
            boundary = np.array(self.boundary_in_param_space_closed(lambda_vals))

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

