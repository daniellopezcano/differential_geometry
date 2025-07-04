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
