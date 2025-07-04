import jax
import jax.numpy as jnp


class Curve:
    """
    Represents a parametrized curve on the manifold.

    - Defined by a parametric function γ(λ) in parameter space.
    """

    def __init__(self, manifold, parametric_function):
        """
        Args:
            manifold: Manifold object with .embed().
            parametric_function: Function lambda → shape (param_dim,)
                mapping a scalar λ to a point in parameter space.
        """
        self.manifold = manifold
        self.parametric_function = parametric_function

        param_dim = manifold.param_dim
        ambient_dim = manifold.ambient_dim

        supported_dims = {(1, 2), (1, 3), (2, 2), (2, 3), (3, 3)}
        if (param_dim, ambient_dim) not in supported_dims:
            raise ValueError(f"Unsupported curve configuration: ({param_dim}, {ambient_dim}) not in {supported_dims}")

    def evaluate_in_param_space(self, lambdas):
        """
        Evaluate the curve in parameter space.

        Args:
            lambdas: Array of λ values (shape (N,)) or scalar.

        Returns:
            Array of shape (N, param_dim) if batched,
            or (param_dim,) if single lambda.
        """
        lambdas = jnp.atleast_1d(lambdas)
        param_points = jax.vmap(self.parametric_function)(lambdas)

        expected_dim = self.manifold.param_dim
        if param_points.shape[-1] != expected_dim:
            raise ValueError(
                f"parametric_function must return shape (..., {expected_dim}), got {param_points.shape}"
            )

        return param_points if lambdas.shape[0] > 1 else param_points[0]

    def evaluate_on_manifold(self, lambdas):
        """
        Evaluate the curve embedded in the manifold (ambient space).

        Args:
            lambdas: Array of λ values (shape (N,)) or scalar.

        Returns:
            Array of shape (N, ambient_dim) if batched,
            or (ambient_dim,) if single lambda.
        """
        param_points = self.evaluate_in_param_space(lambdas)
        embedded = self.manifold.embed(param_points)

        expected_dim = self.manifold.ambient_dim
        if embedded.shape[-1] != expected_dim:
            raise ValueError(
                f"manifold.embed must return shape (..., {expected_dim}), got {embedded.shape}"
            )

        return embedded if embedded.ndim == 2 else embedded[0]

    def tangent_vector_on_manifold(self, lambdas, method="autodiff", delta=1e-5):
        """
        Compute the tangent vector embedded in the manifold (ambient space).

        Args:
            lambdas: Array of λ values (shape (N,)) or scalar.
            method: 'autodiff' (default) or 'finite_difference'.
            delta: Step size for finite differences.

        Returns:
            Array of shape (N, ambient_dim) or (ambient_dim,) if single λ.
        """
        lambdas = jnp.atleast_1d(lambdas)

        if method == "finite_difference":
            plus = self.evaluate_on_manifold(lambdas + delta)
            minus = self.evaluate_on_manifold(lambdas - delta)
            tangent = (plus - minus) / (2 * delta)

        elif method == "autodiff":
            def gamma_fn(lmb):
                param = self.parametric_function(jnp.atleast_1d(lmb))
                return self.manifold.embed(param)

            jac_fn = jax.vmap(jax.jacrev(gamma_fn))
            tangent = jac_fn(lambdas).squeeze(-1)  # Remove trailing dim

        else:
            raise ValueError(f"Unknown method: {method}")

        return tangent if lambdas.shape[0] > 1 else tangent[0]

    def evaluate_in_chart(self, chart, lambdas):
        """
        Evaluate the curve in chart coordinates: X(gamma(lambda)).

        Args:
            chart: Chart object.
            lambdas: array-like

        Returns:
            Array of shape (N, chart_dim)
        """
        param_points = self.evaluate_in_param_space(lambdas)
        return chart.map_to_chart(param_points)
    
    def tangent_vector_components_in_chart(self, chart, lambdas, method="autodiff", delta=1e-5):
        lambdas = jnp.atleast_1d(lambdas)

        if method == "finite_difference":
            plus = self.evaluate_in_chart(chart, lambdas + delta)
            minus = self.evaluate_in_chart(chart, lambdas - delta)
            tangents = (plus - minus) / (2 * delta)

        elif method == "autodiff":
            def composed_fn(lmb):
                return chart.map_to_chart(self.parametric_function(jnp.atleast_1d(lmb)))
            jac_fn = jax.vmap(jax.jacrev(composed_fn))
            tangents = jnp.squeeze(jac_fn(lambdas), axis=1)

        else:
            raise ValueError(f"Unknown method: {method}")

        return tangents if lambdas.shape[0] > 1 else tangents[0]




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