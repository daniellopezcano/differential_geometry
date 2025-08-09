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

    def evaluate_in_chart(self, chart, lambdas: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the curve in chart coordinates: X(gamma(lambda)).

        Args:
            chart: Chart object.
            lambdas: Array of λ values.

        Returns:
            Array of shape (N, chart_dim)
        """
        if chart.manifold != self.manifold:
            raise ValueError("Chart must belong to the same manifold as the curve.")

        param_points = self.evaluate_in_param_space(lambdas)
        return chart.map_to_chart(param_points)

    def derivative_in_chart(self, chart, lambdas: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the derivative d/dλ of the chart-coordinate representation of the curve:
            d/dλ [ X(γ(λ)) ]

        Args:
            chart: Chart object.
            lambdas: Array of λ values.

        Returns:
            Array of shape (N, chart_dim): derivatives of each chart coordinate component
        """
        if chart.manifold != self.manifold:
            raise ValueError("Chart must belong to the same manifold as the curve.")

        def composed_func(lambda_scalar):
            # γ(λ)
            param_point = self.parametric_function(lambda_scalar)
            # X(γ(λ))
            return chart.map_to_chart(param_point)
        
        # Use `jax.jacrev` for vector-valued output, one derivative per component
        return jax.vmap(jax.jacrev(composed_func))(lambdas)

    def directional_derivative_on_manifold(self, lambdas: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the pushforward (directional derivative) of the curve at given λ values.

        This corresponds to:
            d/dλ [Φ(γ(λ))] = DΦ_{γ(λ)} · γ'(λ)

        Args:
            lambdas: Array of shape (N,) with scalar λ values.

        Returns:
            Array of shape (N, ambient_dim): the tangent vectors at each point on the manifold.
        """
        lambdas = jnp.asarray(lambdas)

        # Step 1: Evaluate parametric function γ(λ)
        param_points = jax.vmap(self.parametric_function)(lambdas)  # (N, param_dim)

        # Step 2: Compute γ'(λ) ∈ T_{γ(λ)}(param space)
        param_derivs = jax.vmap(jax.jacrev(self.parametric_function))(lambdas)  # (N, param_dim)

        # Step 3: Compute Jacobian of Φ at γ(λ): DΦ_{γ(λ)} ∈ ℝ^{param_dim × ambient_dim}
        jacobian = self.manifold.derivatives_at_params(param_points)  # (N, param_dim, ambient_dim)

        # Step 4: Pushforward: DΦ(γ(λ)) · γ'(λ)
        tangents = jnp.einsum("nij,ni->nj", jacobian, param_derivs)  # (N, ambient_dim)

        return tangents

    def compute_tangent_vectors_on_manifold(
        self,
        lambda_range=(0, 2 * jnp.pi),
        n_points=500,
        n_arrows=30,
    ):
        """
        Compute normalized tangent vectors of the curve in the ambient manifold space.

        Args:
            lambda_range: Tuple (min, max) defining the parameter interval.
            n_points: Number of lambda values for full resolution.
            n_arrows: Number of tangent vectors to subsample.

        Returns:
            Tuple (positions, tangents): both arrays of shape (n_arrows, ambient_dim).
        """
        lambdas = jnp.linspace(lambda_range[0], lambda_range[1], n_points)
        positions = self.evaluate_on_manifold(lambdas)
        tangents = self.directional_derivative_on_manifold(lambdas)

        # Subsample
        stride = max(1, len(lambdas) // n_arrows)
        sub_positions = positions[::stride]
        sub_tangents = tangents[::stride]

        # Normalize tangent vectors
        norms = jnp.linalg.norm(sub_tangents, axis=1, keepdims=True)
        normed_tangents = sub_tangents / norms

        return sub_positions, normed_tangents







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