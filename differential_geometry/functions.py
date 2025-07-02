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


