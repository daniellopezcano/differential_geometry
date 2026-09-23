import jax.numpy as jnp

def factory_surface_embedding_gaussian_eggs(
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