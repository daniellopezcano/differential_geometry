from typing import Callable
import jax
import jax.numpy as jnp

class Function:
    """
    Represents a real-valued function defined on a manifold.
    The function is defined on the parameter space: f: ℝ^param_dim → ℝ.
    """

    def __init__(self, manifold, function_parametric):
        """
        Args:
            manifold: Manifold object with .param_dim and .ambient_dim.
            function_parametric: Callable mapping (..., param_dim) → (...,)
        """
        self.manifold = manifold
        self.param_dim = manifold.param_dim
        self.ambient_dim = manifold.ambient_dim

        self.valid_combinations = {
            (1, 1), (1, 2), (1, 3),
            (2, 2), (2, 3),
            (3, 3)
        }

        if (self.param_dim, self.ambient_dim) not in self.valid_combinations:
            raise ValueError(f"Unsupported (param_dim, ambient_dim): ({self.param_dim}, {self.ambient_dim})")

        self._f = jax.jit(function_parametric)

    def evaluate_in_param_space(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate f on parameter space: ℝ^{param_dim} → ℝ.

        Args:
            params: Array (..., param_dim)

        Returns:
            Array (...,) of function values.
        """
        params = jnp.asarray(params)
        if params.shape[-1] != self.param_dim:
            raise ValueError(f"Expected shape (..., {self.param_dim}), got {params.shape}")
        return jax.vmap(self._f)(params)

    def evaluate_on_manifold(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Alias of evaluate_in_param_space for conceptual clarity.

        Args:
            params: (..., param_dim)

        Returns:
            (...,) real values.
        """
        return self.evaluate_in_param_space(params)

    # def partial_derivatives_in_chart(
    #     self,
    #     chart,
    #     chart_points: jnp.ndarray,
    #     method="autodiff",
    #     delta=1e-5
    # ) -> jnp.ndarray:
    #     """
    #     Compute ∂(f ◦ X^{-1})/∂x^i at given chart points.

    #     Args:
    #         chart: A Chart object defined on the same manifold.
    #         chart_points: Array of shape (N, chart_dim) with points in R^d (chart coordinates).
    #         method: "autodiff" or "finite_difference"
    #         delta: Step size for finite differences.

    #     Returns:
    #         Array of shape (N, chart_dim) with ∂f/∂x^i at each point.
    #     """
    #     assert chart.manifold == self.manifold, "Chart must belong to the same manifold as the function."

    #     chart_points = jnp.atleast_2d(chart_points)
    #     chart_dim = chart_points.shape[1]

    #     def pulled_back_function(x_chart):
    #         # x_chart ∈ ℝ^d → pull back to param space via X^{-1}
    #         x_chart = jnp.atleast_1d(x_chart)
    #         # Use numerical inverse via optimization or stored inverse if defined
    #         # Here we assume we can call chart.inverse_map(x_chart)
    #         # For simplicity we use a numerical Newton or fallback if not defined
    #         raise NotImplementedError("Need to implement or define chart.inverse_map.")

    #     # === Finite difference version ===
    #     if method == "finite_difference":
    #         def finite_diff_partial(x, i):
    #             dx = jnp.zeros_like(x).at[i].set(delta)
    #             f_plus = self.evaluate_in_param_space(chart.inverse_map(x + dx))
    #             f_minus = self.evaluate_in_param_space(chart.inverse_map(x - dx))
    #             return (f_plus - f_minus) / (2 * delta)

    #         grads = jnp.stack([
    #             jax.vmap(lambda x: jnp.array([finite_diff_partial(x, i) for i in range(chart_dim)]))(chart_points)
    #         ], axis=0)[0]

    #     # === Autodiff version ===
    #     elif method == "autodiff":
    #         def composed_f(x):
    #             param = chart.inverse_map(x)
    #             return self.evaluate_in_param_space(param).reshape(())

    #         jac_fn = jax.vmap(jax.grad(composed_f))
    #         grads = jac_fn(chart_points)

    #     else:
    #         raise ValueError(f"Unknown method: {method}")

    #     return grads  # shape (N, chart_dim)

    # def directional_derivative_along_curve(
    #     self,
    #     curve,
    #     lambda0,
    #     method="autodiff",
    #     delta=1e-5
    # ):
    #     """
    #     Compute the directional derivative of the function along a curve γ at point γ(λ₀).

    #     Args:
    #         curve: A Curve object defined on the same manifold.
    #         lambda0: The parameter λ₀ at which to compute the derivative.
    #         method: 'autodiff' (default) or 'finite_difference'.
    #         delta: Step size if using finite differences.

    #     Returns:
    #         Scalar — the directional derivative (∂f along γ) at λ₀.
    #     """
    #     assert curve.manifold == self.manifold, "Curve must be defined on the same manifold as the function."

    #     if method == "finite_difference":
    #         f_plus = self.evaluate_in_param_space(curve.evaluate_in_param_space(lambda0 + delta))
    #         f_minus = self.evaluate_in_param_space(curve.evaluate_in_param_space(lambda0 - delta))
    #         derivative = (f_plus - f_minus) / (2 * delta)
    #         derivative = derivative.reshape(()) if hasattr(derivative, "reshape") else derivative
    #         return float(derivative)

    #     elif method == "autodiff":
    #         def composed_func(lmb):
    #             point = curve.evaluate_in_param_space(lmb)
    #             value = self.evaluate_in_param_space(point)
    #             return value.reshape(()) if hasattr(value, "reshape") else value

    #         derivative = jax.grad(composed_func)(lambda0)
    #         derivative = derivative.reshape(()) if hasattr(derivative, "reshape") else derivative
    #         return float(derivative)

    #     else:
    #         raise ValueError(f"Unknown method: {method}")