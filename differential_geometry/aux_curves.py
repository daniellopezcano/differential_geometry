import jax.numpy as jnp

def factory_curve_wiggly(p_x, p_y, amplitude=jnp.pi/6, frequency=jnp.pi):
    """
    Returns a parametric curve function γ(λ) passing through (p_x, p_y) at λ = 0.

    Args:
        p_x, p_y: Base point in parameter space.
        amplitude: Amplitude of the oscillation in x-direction.
        frequency: Frequency of the sine oscillation.

    Returns:
        Function curve(λ) → (x(λ), y(λ))
    """
    def curve(lambda_values):
        x = p_x + amplitude * jnp.sin(frequency * lambda_values)
        y = p_y + lambda_values
        return jnp.stack([x, y], axis=-1)
    
    return curve

def factory_curve_arching(p_x, p_y, amplitude=jnp.pi/6, frequency=jnp.pi/3):
    """
    Returns a parametric curve function δ(λ) passing through (p_x, p_y) at λ = 0.

    Args:
        p_x, p_y: Base point in parameter space.
        amplitude: Amplitude of the cosine arch in y-direction.
        frequency: Frequency of the cosine oscillation.

    Returns:
        Function curve(λ) → (x(λ), y(λ))
    """
    def curve(lambda_values):
        x = p_x + lambda_values
        y = p_y + amplitude * (jnp.cos(frequency * lambda_values) - 1.0)  # Shifted cosine
        return jnp.stack([x, y], axis=-1)
    
    return curve

def factory_curve_vertical(p_x, p_y):
    """
    Returns a function representing the vertical line x = p_x.

    Args:
        p_x: x-coordinate (fixed).
        p_y: y-coordinate the curve passes through (lambda=0 reference point).

    Returns:
        A parametric function of lambda (λ) where y = λ + p_y, x = p_x.
    """
    def curve(lambda_values):
        x = jnp.full_like(lambda_values, p_x)
        y = p_y + lambda_values
        return jnp.stack([x, y], axis=-1)
    return curve

def factory_curve_circle(p_x, p_y):
    """
    Returns a function representing a circle centered at (0, 0) 
    with radius sqrt(p_x^2 + p_y^2), passing through (p_x, p_y).

    Args:
        p_x: x-coordinate.
        p_y: y-coordinate.

    Returns:
        A parametric function of lambda (angle parameter λ).
    """
    r = jnp.sqrt(p_x**2 + p_y**2)

    def curve(lambda_values):
        x = r * jnp.cos(lambda_values)
        y = r * jnp.sin(lambda_values)
        return jnp.stack([x, y], axis=-1)

    return curve