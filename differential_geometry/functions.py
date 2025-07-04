from typing import Callable
import jax
import jax.numpy as jnp

class Function:
    """
    Represents a real-valued function defined on a manifold.

    The function is expressed in terms of the manifold's parameter space:
    f: R^{param_dim} → R
    """

    def __init__(
        self,
        manifold,
        function_parametric
    ):
        """
        Args:
            manifold: The manifold object (with embedding defined).
            function_parametric: A callable that takes parameter space
                coordinates (shape (..., param_dim)) and returns real values (shape (...,)).
        """
        self.manifold = manifold
        self.function_parametric = jax.vmap(function_parametric)

    def evaluate_in_param_space(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the function on the parameter space.

        Args:
            params: Array of shape (..., param_dim)

        Returns:
            Array of shape (...,) with function values.
        """
        params = jnp.atleast_2d(params)
        return self.function_parametric(params)

    def evaluate_on_manifold(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the function, but conceptually thought of as living on the manifold.
        This just maps parameter space to real numbers.

        Args:
            params: Array of shape (..., param_dim)

        Returns:
            Array of shape (...,) with function values.
        """
        return self.evaluate_in_param_space(params)

    def directional_derivative_along_curve(
        self,
        curve,
        lambda0,
        method="autodiff",
        delta=1e-5
    ):
        """
        Compute the directional derivative of the function along a curve γ at point γ(λ₀).

        Args:
            curve: A Curve object defined on the same manifold.
            lambda0: The parameter λ₀ at which to compute the derivative.
            method: 'autodiff' (default) or 'finite_difference'.
            delta: Step size if using finite differences.

        Returns:
            Scalar — the directional derivative (∂f along γ) at λ₀.
        """
        assert curve.manifold == self.manifold, "Curve must be defined on the same manifold as the function."

        if method == "finite_difference":
            f_plus = self.evaluate_in_param_space(curve.evaluate_in_param_space(lambda0 + delta))
            f_minus = self.evaluate_in_param_space(curve.evaluate_in_param_space(lambda0 - delta))
            derivative = (f_plus - f_minus) / (2 * delta)
            derivative = derivative.reshape(()) if hasattr(derivative, "reshape") else derivative
            return float(derivative)

        elif method == "autodiff":
            def composed_func(lmb):
                point = curve.evaluate_in_param_space(lmb)
                value = self.evaluate_in_param_space(point)
                return value.reshape(()) if hasattr(value, "reshape") else value

            derivative = jax.grad(composed_func)(lambda0)
            derivative = derivative.reshape(()) if hasattr(derivative, "reshape") else derivative
            return float(derivative)

        else:
            raise ValueError(f"Unknown method: {method}")

    def partial_derivatives_in_chart(
        self,
        chart,
        chart_points: jnp.ndarray,
        method="autodiff",
        delta=1e-5
    ) -> jnp.ndarray:
        """
        Compute ∂(f ◦ X^{-1})/∂x^i at given chart points.

        Args:
            chart: A Chart object defined on the same manifold.
            chart_points: Array of shape (N, chart_dim) with points in R^d (chart coordinates).
            method: "autodiff" or "finite_difference"
            delta: Step size for finite differences.

        Returns:
            Array of shape (N, chart_dim) with ∂f/∂x^i at each point.
        """
        assert chart.manifold == self.manifold, "Chart must belong to the same manifold as the function."

        chart_points = jnp.atleast_2d(chart_points)
        chart_dim = chart_points.shape[1]

        def pulled_back_function(x_chart):
            # x_chart ∈ ℝ^d → pull back to param space via X^{-1}
            x_chart = jnp.atleast_1d(x_chart)
            # Use numerical inverse via optimization or stored inverse if defined
            # Here we assume we can call chart.inverse_map(x_chart)
            # For simplicity we use a numerical Newton or fallback if not defined
            raise NotImplementedError("Need to implement or define chart.inverse_map.")

        # === Finite difference version ===
        if method == "finite_difference":
            def finite_diff_partial(x, i):
                dx = jnp.zeros_like(x).at[i].set(delta)
                f_plus = self.evaluate_in_param_space(chart.inverse_map(x + dx))
                f_minus = self.evaluate_in_param_space(chart.inverse_map(x - dx))
                return (f_plus - f_minus) / (2 * delta)

            grads = jnp.stack([
                jax.vmap(lambda x: jnp.array([finite_diff_partial(x, i) for i in range(chart_dim)]))(chart_points)
            ], axis=0)[0]

        # === Autodiff version ===
        elif method == "autodiff":
            def composed_f(x):
                param = chart.inverse_map(x)
                return self.evaluate_in_param_space(param).reshape(())

            jac_fn = jax.vmap(jax.grad(composed_f))
            grads = jac_fn(chart_points)

        else:
            raise ValueError(f"Unknown method: {method}")

        return grads  # shape (N, chart_dim)
