import jax
import jax.numpy as jnp

import differential_geometry.functions as functions


class VectorField:
    """
    Represents a smooth vector field, i.e. a smooth section of the tangent bundle

        X : M ---> TM ,     p ---> X(p) = v_{gamma, p} ,     with  pi o X = Id_M .

    The field is specified by its components in parameter space, which is the
    representation used internally. The components in the chart-induced basis of
    a chart (U, X), i.e. the component functions X^i in C^inf(U) of

        X|_U = X^i * dX_i ,

    are obtained with components_in_chart, and the pushforward to the ambient
    space with evaluate_on_manifold.
    """

    def __init__(self, manifold, vector_function):
        """
        Args:
            manifold: Manifold object with .embed().
            vector_function: Function mapping a point in parameter space (shape (dim,))
                             to a vector (shape (dim,)) in parameter space coordinates.
        """
        self.manifold = manifold
        self.dim = manifold.dim
        self.vector_function = vector_function

    # ======================================================
    # === Field Evaluation in Parameter Space
    # ======================================================

    def evaluate_in_param_space(self, param_points):
        """
        Evaluate the vector field in parameter space.

        Args:
            param_points: Array of shape (N, dim) or (dim,)

        Returns:
            Array of shape (N, dim) or (dim,)
        """
        param_points = jnp.atleast_2d(param_points)
        single_input = param_points.shape[0] == 1

        vectors = jax.vmap(self.vector_function)(param_points)

        return vectors[0] if single_input else vectors

    # ======================================================
    # === Field Evaluation on Manifold (Ambient Space)
    # ======================================================

    def evaluate_on_manifold(self, param_points):
        """
        Pushforward of the vector field to the manifold (ambient space).

        Args:
            param_points: Array of shape (N, dim) or (dim,)

        Returns:
            Array of shape (N, ambient_dim) or (ambient_dim,)
        """
        param_points = jnp.atleast_2d(param_points)
        single_input = param_points.shape[0] == 1

        # Compute Jacobians
        def embed_jacobian(point):
            return jax.jacrev(self.manifold.embedding_func)(point)  # (ambient_dim, dim)

        jacobians = jax.vmap(embed_jacobian)(param_points)  # (N, ambient_dim, dim)

        # Field in parameter space
        vectors_param = jnp.atleast_2d(self.evaluate_in_param_space(param_points))  # (N, dim)

        vector_ambient = jnp.einsum('nij,nj->ni', jacobians, vectors_param)  # (N, ambient_dim)

        return vector_ambient[0] if single_input else vector_ambient

    # ======================================================
    # === Component functions in a chart-induced basis
    # ======================================================

    def components_in_chart(self, chart, param_points):
        """
        Component functions X^i of the field in the chart-induced basis {dX_i}:

            X^i(p) = (dX^i)_p( X(p) ) = d_j[X^i](p) * X^j_param(p).

        Args:
            chart: Chart object defined on the same manifold.
            param_points: Array of shape (N, dim) or (dim,) — base points, given
                in parameter space (the chart map is applied internally).

        Returns:
            Array of shape (N, dim) or (dim,) with the component functions
            evaluated at those base points.
        """
        if chart.manifold != self.manifold:
            raise ValueError("Chart must belong to the same manifold as the field.")

        param_points = jnp.atleast_2d(param_points)
        single_input = param_points.shape[0] == 1

        # A^i_j = dX^i / dtheta^j at each base point
        jacobians = jax.vmap(jax.jacrev(chart.chart_map))(param_points)          # (N, dim, dim)
        vectors_param = jnp.atleast_2d(self.evaluate_in_param_space(param_points))

        components = jnp.einsum('nij,nj->ni', jacobians, vectors_param)          # (N, dim)

        return components[0] if single_input else components

    def base_points_in_chart(self, chart, param_points):
        """
        Chart coordinates X^i(p) of the base points of the field, i.e. the first
        half of the tangent-bundle chart xi_X(v_{gamma,p}) = ( X^i(p) ; gamma^j ).

        Args:
            chart: Chart object defined on the same manifold.
            param_points: Array of shape (N, dim) or (dim,)

        Returns:
            Array of shape (N, dim) or (dim,)
        """
        param_points = jnp.atleast_2d(param_points)
        single_input = param_points.shape[0] == 1

        chart_points = jax.vmap(chart.map_to_chart)(param_points)

        return chart_points[0] if single_input else chart_points

    # ======================================================
    # === Action on scalar fields:  X|_f_| in C^inf(M)
    # ======================================================

    def action_on_function(self, function):
        """
        Action of the vector field on a scalar field,

            X|_f_|(p) := v_{gamma, p}(f),

        which is again an element of C^inf(M). It is computed chart-free, as the
        directional derivative of f along the field in parameter space.

        Args:
            function: Function object (or a plain callable R^dim -> R).

        Returns:
            Function object representing the scalar field X|_f_|.
        """
        scalar_function = getattr(function, "_f", function)

        def action_parametric(params):
            return jnp.dot(jax.grad(scalar_function)(params), self.vector_function(params))

        return functions.Function(manifold=self.manifold, function_parametric=action_parametric)

    # ======================================================
    # === C^inf(M)-module operations on Gamma TM
    # ======================================================

    def add(self, other):
        """
        Addition of vector fields, X_1 (+)_Gamma X_2, defined pointwise by

            (X_1 (+)_Gamma X_2)|_f_|(p) := X_1|_f_|(p) + X_2|_f_|(p).

        Args:
            other: VectorField on the same manifold.

        Returns:
            VectorField — the sum.
        """
        if other.manifold != self.manifold:
            raise ValueError("Both fields must be defined on the same manifold.")

        def summed_vector_function(params):
            return self.vector_function(params) + other.vector_function(params)

        return VectorField(self.manifold, summed_vector_function)

    def multiply_by_function(self, function):
        """
        s-multiplication of a vector field by an element of the ring C^inf(M),
        f (.)_Gamma X, defined pointwise by

            (f (.)_Gamma X)|_g_|(p) := f(p) * X|_g_|(p).

        Note that the scaling factor is a *function*, not a real number: this is
        precisely what makes Gamma TM a C^inf(M)-module rather than a vector space.

        Args:
            function: Function object (or a plain callable R^dim -> R).

        Returns:
            VectorField — the rescaled field.
        """
        scalar_function = getattr(function, "_f", function)

        def scaled_vector_function(params):
            return scalar_function(params) * self.vector_function(params)

        return VectorField(self.manifold, scaled_vector_function)


# Backwards-compatible alias (the class used to be called Field)
Field = VectorField


# ==========================================================
# === Chart-induced frame fields
# ==========================================================

def chart_induced_frame(chart):
    """
    The d chart-induced basis fields  dX_i := ( d/dX^i )  over the chart domain U.

    They provide the local frame in which any field is expanded as X|_U = X^i * dX_i.
    Such a frame exists over a chart domain, but in general not globally
    (colloquially, the "hairy ball theorem").

    Args:
        chart: Chart object.

    Returns:
        List of `dim` VectorField objects, [dX_1, ..., dX_d].
    """
    manifold = chart.manifold

    def make_frame_field(index):
        def frame_vector_function(params):
            # A^i_j = dX^i/dtheta^j  -->  d/dX^i has parameter components (A^{-1})^j_i
            jacobian = jax.jacrev(chart.chart_map)(params)
            return jnp.linalg.inv(jacobian)[:, index]

        return VectorField(manifold, frame_vector_function)

    return [make_frame_field(index) for index in range(manifold.dim)]
