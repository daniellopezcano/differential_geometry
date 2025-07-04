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
            param_dim: Dimension of the parameter space (e.g., 1D curve, 2D surface).
            ambient_dim: Dimension of the ambient space (e.g., 2D or 3D).
            embedding_func: Function f: R^{param_dim} -> R^{ambient_dim}
                that defines the manifold embedding.
        """
        valid_combinations = {
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 3),
        }
        if (param_dim, ambient_dim) not in valid_combinations:
            raise ValueError(
                f"Invalid (param_dim, ambient_dim) = ({param_dim}, {ambient_dim}). "
                "Allowed combinations: " + ", ".join(map(str, valid_combinations))
            )

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
        params = jnp.asarray(params)
        if params.shape[-1] != self.param_dim:
            raise ValueError(
                f"Expected params.shape[-1] == {self.param_dim}, but got {params.shape[-1]}"
            )

        # Ensure input is at least 2D for consistent vectorization
        params_2d = params.reshape(-1, self.param_dim)
        embedded = jax.vmap(self.embedding_func)(params_2d)

        if embedded.shape[-1] != self.ambient_dim:
            raise ValueError(
                f"Embedding function output has last dim {embedded.shape[-1]}, "
                f"but expected ambient_dim = {self.ambient_dim}."
            )

        return embedded.reshape(params.shape[:-1] + (self.ambient_dim,))

    def derivatives_at_params(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the Jacobian matrix ∂Φ/∂param_dim for a batch of parameter points.

        Args:
            params: Array of shape (..., param_dim).

        Returns:
            Array of shape (..., param_dim, ambient_dim) containing all directional derivatives
            at each point in parameter space.
        """
        params = jnp.asarray(params)
        if params.shape[-1] != self.param_dim:
            raise ValueError(
                f"Expected params.shape[-1] == {self.param_dim}, but got {params.shape[-1]}"
            )

        params_2d = params.reshape(-1, self.param_dim)

        # Compute jacobian of shape (N, ambient_dim, param_dim)
        jacobian = jax.vmap(jax.jacrev(self.embedding_func))(params_2d)

        # Transpose to (N, param_dim, ambient_dim)
        jacobian = jnp.transpose(jacobian, axes=(0, 2, 1))

        return jacobian.reshape(params.shape[:-1] + (self.param_dim, self.ambient_dim))


def surface_embedding_factory(
    amplitude=1.0,
    center=(0.0, 0.0),
    sigma=(jnp.pi, jnp.pi)
):
    """
    Creates an embedding function for the manifold.

    The manifold is shaped by an 'egg-carton' surface modulated by a Gaussian envelope.

    Args:
        amplitude: Amplitude of the Gaussian envelope.
        center: Tuple (mu_x, mu_y) — center of the Gaussian.
        sigma: Tuple (sigma_x, sigma_y) — standard deviations of the Gaussian.

    Returns:
        embedding function: params (x, y) → (x, y, z)
    """

    mu_x, mu_y = center
    sigma_x, sigma_y = sigma

    def embedding(params):
        x, y = params

        # Egg-carton surface
        z_eggs = (
            jnp.cos(x + y) / 2 +
            jnp.sin(x - y + jnp.pi / 2) / 2 +
            1
        )

        # Gaussian envelope
        z_gaussian = amplitude * jnp.exp(
            -(((x - mu_x)**2) / (2 * sigma_x**2) +
              ((y - mu_y)**2) / (2 * sigma_y**2))
        )

        z = z_eggs * z_gaussian

        return jnp.array([x, y, z])

    return embedding