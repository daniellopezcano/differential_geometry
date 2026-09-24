import jax
import jax.numpy as jnp

import differential_geometry.functions as functions


class Covector:
    """
    Represents a covector at a point p of the manifold, i.e. an element of the
    cotangent space T_p^*M = Hom(T_pM, R).

    Covectors are built here as differentials (gradients) of smooth scalar fields,

        (df)_p : T_pM ---> R ,      (df)_p( v_{gamma,p} ) := v_{gamma,p}(f),

    which is the natural generating family of T_p^*M.

    The components w.r.t. the dual chart-induced basis {(dX^i)_p} of a chart
    (U, X) are

        x_i^{(p)} = d_i [ f o X^{-1} ] ( X(p) ).
    """

    def __init__(self, function, param_point):
        """
        Args:
            function: Function object (scalar field) defined on the manifold.
            param_point: Array of shape (dim,) — the point p in parameter space.
        """
        self.function = function
        self.manifold = function.manifold
        self.dim = function.dim

        param_point = jnp.asarray(param_point)
        if param_point.shape != (self.dim,):
            raise ValueError(
                f"Expected param_point of shape ({self.dim},), got {param_point.shape}"
            )
        self.param_point = param_point

        # Single-point evaluation of the scalar field, f: R^dim -> R.
        # (Function.evaluate_in_param_space is vmapped and therefore expects a batch.)
        self._f = function._f

    # ======================================================
    # === Components in a chart-induced dual basis
    # ======================================================

    def components_in_chart(self, chart, method: str = "auto") -> jnp.ndarray:
        """
        Components x_i^{(p)} of the covector in the dual chart-induced basis
        {(dX^i)_p} of the chart (U, X).

        Args:
            chart: Chart object defined on the same manifold.
            method:
                "chart_inverse"  — literal definition, differentiating f o X^{-1}
                                   at X(p) (requires chart.inverse_chart_map).
                "param_jacobian" — equivalent route that only needs the chart map:
                                   solves  sum_i x_i * (dX^i/dtheta^j) = d_j f
                                   at the point p of parameter space.
                "auto"           — the first one if an inverse chart map is
                                   available, the second one otherwise.

        Returns:
            Array of shape (dim,) with the covector components.
        """
        if chart.manifold != self.manifold:
            raise ValueError("Chart must belong to the same manifold as the covector.")

        if method == "auto":
            method = (
                "chart_inverse" if chart.inverse_chart_map is not None else "param_jacobian"
            )

        if method == "chart_inverse":
            if chart.inverse_chart_map is None:
                raise ValueError(
                    "Chart has no inverse_chart_map; use method='param_jacobian'."
                )
            chart_point = chart.map_to_chart(self.param_point)

            def f_in_chart(chart_coords):
                return self._f(chart.inverse_chart_map(chart_coords))

            components = jax.grad(f_in_chart)(chart_point)

        elif method == "param_jacobian":
            # d_j f at p (derivatives w.r.t. the parameter-space coordinates)
            grad_param = jax.grad(self._f)(self.param_point)                  # (dim,)
            # A^i_j = dX^i / dtheta^j at p
            jac_chart = jax.jacrev(chart.chart_map)(self.param_point)         # (dim, dim)
            # x_i A^i_j = d_j f   -->   A^T x = grad_param
            components = jnp.linalg.solve(jac_chart.T, grad_param)

        else:
            raise ValueError(f"Unknown method: {method}")

        return components

    def level_spacing_in_chart(self, chart, level_step: float = 1.0) -> float:
        """
        Euclidean distance (in chart coordinates) between two consecutive level
        lines of the covector separated by `level_step` units, i.e.
        level_step / ||x_i||. Useful to tune the covector "stack" plots.
        """
        components = self.components_in_chart(chart)
        return float(level_step / jnp.linalg.norm(components))

    # ======================================================
    # === Action on tangent vectors
    # ======================================================

    def evaluate_on_velocity(self, curve, lambda_0, atol: float = 1e-6) -> float:
        """
        Chart-free action of the covector on the velocity of a curve:

            (df)_p( v_{gamma,p} ) := v_{gamma,p}(f) = ( f o gamma )'(lambda_0).

        Args:
            curve: Curve object on the same manifold, with gamma(lambda_0) = p.
            lambda_0: Curve parameter at which the curve passes through p.
            atol: Tolerance used to check that gamma(lambda_0) = p.

        Returns:
            Scalar — the directional derivative of f along the curve at p.
        """
        if curve.manifold != self.manifold:
            raise ValueError("Curve must belong to the same manifold as the covector.")

        lambda_0 = jnp.asarray(lambda_0, dtype=float)
        gamma_lambda_0 = curve.parametric_function(lambda_0)
        if not bool(jnp.allclose(gamma_lambda_0, self.param_point, atol=atol)):
            raise ValueError(
                f"The curve does not pass through the base point at lambda_0={lambda_0}: "
                f"gamma(lambda_0) = {gamma_lambda_0}, p = {self.param_point}."
            )

        def f_along_curve(lmb):
            return self._f(curve.parametric_function(lmb))

        return jax.grad(f_along_curve)(lambda_0)

    def evaluate_on_vector_in_chart(self, chart, vector_components) -> float:
        """
        Action of the covector on a tangent vector given through its components
        in the chart-induced basis of the same chart:

            X_p( v ) = sum_i x_i^{(p)} * v^i.

        Args:
            chart: Chart object defined on the same manifold.
            vector_components: Array of shape (dim,) — components v^i.

        Returns:
            Scalar — the value of the pairing (chart-independent).
        """
        vector_components = jnp.asarray(vector_components).reshape(self.dim)
        return jnp.sum(self.components_in_chart(chart) * vector_components)


# ==========================================================
# === Chart-transition utilities
# ==========================================================

def chart_transition_jacobian(chart_target, chart_source, param_point) -> jnp.ndarray:
    """
    Jacobian of the chart transition map T_UV = X o Y^{-1} at the point p, i.e.

        (J_UV)^i_j (p) = ( dX^i / dY^j )_p ,

    where X is the chart map of `chart_target` (U) and Y that of `chart_source` (V).

    It is computed without inverting any chart map, as
    (dX/dtheta) * (dY/dtheta)^{-1} at p, with theta the parameter-space coordinates.

    Args:
        chart_target: Chart (U, X) — the chart the transition map maps *into*.
        chart_source: Chart (V, Y) — the chart the transition map maps *from*.
        param_point: Array of shape (dim,) — the point p in parameter space.

    Returns:
        Array of shape (dim, dim) with entries (J_UV)^i_j, i = row, j = column.
    """
    if chart_target.manifold != chart_source.manifold:
        raise ValueError("Both charts must belong to the same manifold.")

    param_point = jnp.asarray(param_point)
    jac_target = jax.jacrev(chart_target.chart_map)(param_point)   # dX^i / dtheta^j
    jac_source = jax.jacrev(chart_source.chart_map)(param_point)   # dY^i / dtheta^j

    return jac_target @ jnp.linalg.inv(jac_source)


def velocity_components_in_chart(curve, chart, lambda_0) -> jnp.ndarray:
    """
    Components of the velocity of a curve in the chart-induced basis:

        gamma^i_U(lambda_0) = ( X^i o gamma )'(lambda_0).

    Thin wrapper around Curve.derivative_in_chart for a single parameter value.

    Args:
        curve: Curve object.
        chart: Chart object defined on the same manifold.
        lambda_0: Curve parameter.

    Returns:
        Array of shape (dim,).
    """
    lambdas = jnp.atleast_1d(jnp.asarray(lambda_0, dtype=float))
    return curve.derivative_in_chart(chart, lambdas)[0]


# ==========================================================
# === Covector fields
# ==========================================================

class CovectorField:
    """
    Represents a smooth covector field, i.e. a smooth section of the cotangent bundle

        Theta : M ---> T*M ,     p ---> Theta(p) = X_p ,     with  pi* o Theta = Id_M .

    The field is specified by its components w.r.t. the basis d(theta^i) induced by the
    parameter space, which is the representation used internally. The component
    functions X_i in C^inf(U) of

        Theta|_U = X_i * dX^i

    in a chart (U, X) are obtained with components_in_chart; they transform covariantly.
    """

    def __init__(self, manifold, covector_function):
        """
        Args:
            manifold: Manifold object.
            covector_function: Function mapping a point in parameter space (shape (dim,))
                to the components (shape (dim,)) w.r.t. the basis d(theta^i).
        """
        self.manifold = manifold
        self.dim = manifold.dim
        self.covector_function = covector_function

    @classmethod
    def differential(cls, function):
        """
        The differential df of a scalar field, as a covector field (an exact 1-form):
        its parameter-space components are the partial derivatives of f.

        Args:
            function: Function object.

        Returns:
            CovectorField.
        """
        return cls(function.manifold, jax.grad(function._f))

    def evaluate_in_param_space(self, param_points):
        """
        Components w.r.t. the basis d(theta^i) at the given base points.

        Args:
            param_points: Array of shape (N, dim) or (dim,)

        Returns:
            Array of shape (N, dim) or (dim,)
        """
        param_points = jnp.atleast_2d(param_points)
        single_input = param_points.shape[0] == 1

        covectors = jax.vmap(self.covector_function)(param_points)

        return covectors[0] if single_input else covectors

    def components_in_chart(self, chart, param_points):
        """
        Component functions X_i of the field in the dual chart-induced basis {dX^i}:

            X_i(p) = Theta(p)( (d/dX^i)_p ) = (A^{-1})^j_i(p) * X_j^param(p),

        with A^i_j = dX^i / dtheta^j the Jacobian of the chart map.

        Args:
            chart: Chart object defined on the same manifold.
            param_points: Array of shape (N, dim) or (dim,) — base points in parameter space.

        Returns:
            Array of shape (N, dim) or (dim,).
        """
        if chart.manifold != self.manifold:
            raise ValueError("Chart must belong to the same manifold as the field.")

        param_points = jnp.atleast_2d(param_points)
        single_input = param_points.shape[0] == 1

        jacobians = jax.vmap(jax.jacrev(chart.chart_map))(param_points)           # (N, d, d)
        covectors_param = jnp.atleast_2d(self.evaluate_in_param_space(param_points))
        components = jnp.einsum('nji,nj->ni', jnp.linalg.inv(jacobians), covectors_param)

        return components[0] if single_input else components

    def action_on_vector_field(self, vector_field):
        """
        Action of the covector field on a vector field,

            Theta|X|(p) := Theta(p)( X(p) ),

        which is an element of C^inf(M). Computed chart-free, by contracting the
        parameter-space components of both fields.

        Args:
            vector_field: VectorField on the same manifold.

        Returns:
            Function object representing the scalar field Theta|X|.
        """
        if vector_field.manifold != self.manifold:
            raise ValueError("Both fields must be defined on the same manifold.")

        def action_parametric(params):
            return jnp.dot(self.covector_function(params), vector_field.vector_function(params))

        return functions.Function(manifold=self.manifold, function_parametric=action_parametric)

    def add(self, other):
        """Addition of covector fields, defined pointwise."""
        if other.manifold != self.manifold:
            raise ValueError("Both fields must be defined on the same manifold.")

        def summed(params):
            return self.covector_function(params) + other.covector_function(params)

        return CovectorField(self.manifold, summed)

    def multiply_by_function(self, function):
        """s-multiplication by an element of the ring C^inf(M), defined pointwise."""
        scalar_function = getattr(function, "_f", function)

        def scaled(params):
            return scalar_function(params) * self.covector_function(params)

        return CovectorField(self.manifold, scaled)
