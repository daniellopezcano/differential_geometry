from typing import Callable, Tuple
import jax
import jax.numpy as jnp


class Manifold:
    def __init__(
        self,
        name: str,
        dim: int,
        ambient_dim: int,
        embedding_func: Callable[[jnp.ndarray], jnp.ndarray]
    ):
        """
        Manifold embedded in R^n.

        Args:
            name: Name of the manifold (for reference).
            dim: Dimension of the parameter space (e.g., 1D curve, 2D surface).
            ambient_dim: Dimension of the ambient space (e.g., 2D or 3D).
            embedding_func: Function f: R^{dim} -> R^{ambient_dim} that defines the manifold embedding.
        """
        self.name = name
        self.dim = dim
        self.ambient_dim = ambient_dim
        self.embedding_func = embedding_func

    def embed(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the embedding for a batch of parameter points.

        Args:
            params: Array of shape (..., dim) — parameter points.

        Returns:
            Array of shape (..., ambient_dim) — embedded points.
        """
        params = jnp.asarray(params)
        if params.shape[-1] != self.dim:
            raise ValueError(
                f"Expected params.shape[-1] == {self.dim}, but got {params.shape[-1]}"
            )

        # Reshape input for consistent vectorization
        params_ = params.reshape(-1, self.dim)
        embedded_ = jax.vmap(self.embedding_func)(params_)

        if embedded_.shape[-1] != self.ambient_dim:
            raise ValueError(
                f"Embedding function output has last dim {embedded_.shape[-1]}, "
                f"but expected ambient_dim = {self.ambient_dim}."
            )
        
        # Reshape embedded output to match original shape (..., ambient_dim)
        embedded = embedded_.reshape(params.shape[:-1] + (self.ambient_dim,))

        return embedded

    def derivatives_at_params(self, params: jnp.ndarray, jack_mode: str = "jacrev") -> jnp.ndarray:
        """
        Compute the Jacobian matrix ∂Φ/∂dim for a batch of parameter points.

        Args:
            params: Array of shape (..., dim).
            jack_mode: str, "jacfwd" or "jacrev"

        Returns:
            Array of shape (..., dim, ambient_dim) containing all directional derivatives
            at each point in parameter space.
        """
        params = jnp.asarray(params)
        if params.shape[-1] != self.dim:
            raise ValueError(
                f"Expected params.shape[-1] == {self.dim}, but got {params.shape[-1]}"
            )

        # Reshape input for consistent vectorization
        params_ = params.reshape(-1, self.dim)

        # Compute jacobian of shape (N, ambient_dim, dim)
        if jack_mode == "jacfwd":
            jacobian_ = jax.vmap(jax.jacfwd(self.embedding_func))(params_)
        elif jack_mode == "jacrev":
            jacobian_ = jax.vmap(jax.jacrev(self.embedding_func))(params_)

        # Transpose to (N, dim, ambient_dim)
        jacobian_ = jnp.transpose(jacobian_, axes=(0, 2, 1))
    
        # Reshape jacobian to match original shape (..., dim, ambient_dim)
        jacobian = jacobian_.reshape(params.shape[:-1] + (self.dim, self.ambient_dim))

        return jacobian

