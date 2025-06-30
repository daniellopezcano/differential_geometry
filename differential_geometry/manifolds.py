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