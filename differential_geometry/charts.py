from typing import Callable, Optional, Union, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

import optax

from scipy.spatial import Delaunay


class Chart:
    def __init__(
        self,
        name: str,
        manifold,
        boundary: Union[Tuple[float, float], np.ndarray],
        chart_map: Callable[[jnp.ndarray], jnp.ndarray],
        region_sampler: Optional[Callable[[int], jnp.ndarray]] = None,
        inverse_chart_map: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ):
        """
        Initialize a coordinate chart on a manifold.

        Args:
            name: Chart name (for reference).
            manifold: A Manifold object.
            boundary:
                - If dim == 1: tuple (t_min, t_max) for interval in parameter space.
                - If dim == 2: array of shape (N, 2), closed non-self-intersecting curve.
            chart_map: Function from parameter space to chart coordinates.
            region_sampler: Optional sampling function; otherwise a default is selected.
            inverse_chart_map: Optional inverse of chart_map, if available.
        """
        self.name = name
        self.manifold = manifold
        self.chart_map = chart_map
        self.inverse_chart_map = inverse_chart_map

        dim = manifold.dim
        ambient_dim = manifold.ambient_dim

        supported_dims = {(1, 2), (1, 3), (2, 3)}
        if (dim, ambient_dim) not in supported_dims:
            raise ValueError(f"Unsupported chart configuration: {(dim, ambient_dim)} not in {supported_dims}")

        if dim == 1:
            if not (isinstance(boundary, tuple) and len(boundary) == 2):
                raise ValueError("For 1D charts, boundary must be a tuple (t_min, t_max)")
            self.boundary_interval = boundary
        elif dim == 2:
            boundary = np.asarray(boundary)
            if boundary.ndim != 2 or boundary.shape[1] != 2:
                raise ValueError("For 2D charts, boundary must be an array of shape (N, 2)")
            if not np.allclose(boundary[0], boundary[-1], atol=1e-6):
                raise ValueError("Boundary curve must be closed.")
            self.boundary_curve = boundary

        if region_sampler is not None:
            self.region_sampler = region_sampler
        else:
            self.region_sampler = (
                self._default_region_sampler_1d()
                if dim == 1
                else self._default_region_sampler_2d()
            )

    def map_to_chart(self, param_points: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the chart map to points in parameter space.

        Args:
            param_points: Array of shape (..., dim)

        Returns:
            Array of shape (..., chart_dim)
        """
        param_points = jnp.asarray(param_points)
        if param_points.shape[-1] != self.manifold.dim:
            raise ValueError(
                f"Expected param_points with last dim = {self.manifold.dim}, got {param_points.shape[-1]}"
            )
        return self.chart_map(param_points)

    def boundary_in_param_space(self, lambdas: jnp.ndarray) -> np.ndarray:
        """
        Return the boundary in parameter space, ensuring closure.

        → Only defined for 2D parameter space (closed curve).

        Returns:
            Array of shape (N+1, 2) with the first point repeated at the end.
        """
        if self.manifold.dim != 2:
            raise RuntimeError(
                "boundary_in_param_space is only defined for 2D parameter spaces."
            )
        boundary = np.array(self.boundary_curve)
        if not np.allclose(boundary[0], boundary[-1]):
            boundary = np.vstack([boundary, boundary[0]])
        return boundary

    def _default_region_sampler_2d(self) -> Callable[[int], jnp.ndarray]:
        """
        Region sampler for dim = 2 using Delaunay triangulation of the boundary.

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

    def sample_region_in_param_space(
        self, n_points: int = 2000
    ) -> jnp.ndarray:
        """
        Sample a set of points inside the region U in parameter space.

        Args:
            n_points: Approximate number of points to sample.

        Returns:
            Array of shape (N, dim)
        """
        return self.region_sampler(n_points)

    def inverse_map(self, x_target, initial_guess=None, n_iters=100, lr=0.05):
        """
        Inverse map via user-defined function or differentiable gradient descent.

        Args:
            x_target: Array (..., chart_dim) — target in chart coords
            initial_guess: Optional array (..., dim)
            n_iters: Number of optax steps
            lr: Learning rate for gradient descent

        Returns:
            Approximate preimages (..., dim)
        """
        x_target = jnp.atleast_2d(jnp.array(x_target))
        dim = self.manifold.dim

        if self.inverse_chart_map is not None:
            return jax.vmap(self.inverse_chart_map)(x_target)

        # Fallback to autodiff-based gradient descent inversion
        if initial_guess is None:
            ig = jnp.mean(self.sample_region_in_param_space(500), axis=0)
            initial_guess = jnp.broadcast_to(ig, (x_target.shape[0], dim))

        def loss_fn(u, x_target_single):
            x_est = self.map_to_chart(u)
            return jnp.sum((x_est - x_target_single) ** 2)

        @jit
        def optimize_single(x_target_single, u0):
            opt = optax.adam(lr)
            opt_state = opt.init(u0)

            def step_fn(u, opt_state):
                _, grad = value_and_grad(loss_fn)(u, x_target_single)
                updates, opt_state = opt.update(grad, opt_state)
                u = optax.apply_updates(u, updates)
                return u, opt_state

            u = u0
            for _ in range(n_iters):
                u, opt_state = step_fn(u, opt_state)
            return u

        results = jax.vmap(optimize_single)(x_target, initial_guess)
        return results

