from typing import Callable, Tuple
import jax
import jax.numpy as jnp


class Manifold:
    def __init__(
        self,
        param_dim: int,
        ambient_dim: int,
        embedding_func: Callable[[jnp.ndarray], jnp.ndarray],
    ):
        """
        Manifold embedded in R^n.

        Args:
            param_dim: Dimension of the parameter space (1D curve or 2D surface).
            ambient_dim: Dimension of the ambient space (2D or 3D).
            embedding_func: Function f: R^{param_dim} -> R^{ambient_dim}
                that defines the manifold embedding.
        """
        self.param_dim = param_dim
        self.ambient_dim = ambient_dim
        self.embedding_func = embedding_func

    def embed(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the embedding for a batch of parameter points.

        Args:
            params: Array of shape (..., param_dim) — parameter points.

        Returns:
            Array of shape (..., ambient_dim) — embedded points.
        """
        params = jnp.atleast_2d(params)
        return jax.vmap(self.embedding_func)(params)

    def tangent_plane_at(self, p_param: jnp.ndarray):
        """
        Compute the tangent plane basis vectors at a point p on the manifold.

        Args:
            p_param: Point in parameter space (shape (2,)).

        Returns:
            p_xyz: Embedded point on the manifold (shape (3,)).
            tangent_vec1: First tangent vector in R^3 (∂Φ/∂x).
            tangent_vec2: Second tangent vector in R^3 (∂Φ/∂y).
        """
        jacobian = jax.jacrev(self.embedding_func)(p_param)  # Shape (ambient_dim, param_dim)

        tangent_vec1 = jacobian[:, 0]  # ∂Φ/∂x
        tangent_vec2 = jacobian[:, 1]  # ∂Φ/∂y

        p_xyz = self.embed(p_param)
        if p_xyz.shape[0] == 1:
            p_xyz = p_xyz[0]

        return p_xyz, tangent_vec1, tangent_vec2
