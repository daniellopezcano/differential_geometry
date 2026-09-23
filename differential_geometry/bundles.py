from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

import differential_geometry.manifolds as manifolds


class TangentBundle:
    """
    The tangent bundle  TM --pi--> M  of a manifold, in the concrete representation
    used for computations and figures.

    A point of the total space is represented by the pair

        (theta^1, ..., theta^d ; u^1, ..., u^d)  in  R^{2d},

    where the first block are the parameter-space coordinates of the base point p and
    the second block are the components of the tangent vector at p w.r.t. the
    parameter-space basis. The projection simply forgets the second block.

    Bundle charts are induced by charts of the base, exactly as in the notes:

        xi_X (v_{gamma,p}) = ( X^i(p) ; (dX^j)_p(v_{gamma,p}) ),

    i.e. the base block is mapped with the chart map and the fibre block with the
    Jacobian of the chart map (see chart_map below).

    Drawing the total space requires one extra, purely visual, choice: an ambient
    direction along which each fibre is displayed. For a curve embedded in R^3 -- the
    case of the figures -- the fibre over p is drawn along a unit vector n(p)
    transverse to the curve, so that the total space of a ring becomes the familiar
    cylinder. That drawing is stored in `self.total_space`, which is an ordinary
    Manifold object whose parameter space is the bundle parameter space above, so all
    the existing plotting machinery applies to it.
    """

    def __init__(
        self,
        manifold,
        fibre_direction: Optional[Union[Callable, jnp.ndarray]] = None,
        fibre_scale: float = 1.0,
        name: str = r"$T\mathcal{M}$",
    ):
        """
        Args:
            manifold: Manifold object (the base space).
            fibre_direction: Ambient direction along which the fibres are drawn. Either
                a callable mapping a parameter point (shape (dim,)) to an ambient vector
                (shape (ambient_dim,)), or a fixed ambient vector. Defaults to the last
                ambient axis, e.g. (0, 0, 1) for a curve in R^3.
            fibre_scale: Scaling of the fibre coordinate in the drawing of the total space.
            name: Name of the bundle (for reference and labels).
        """
        self.manifold = manifold
        self.base_dim = manifold.dim
        self.dim = 2 * manifold.dim
        self.ambient_dim = manifold.ambient_dim
        self.fibre_scale = fibre_scale
        self.name = name

        self.fibre_direction = self._resolve_fibre_direction(fibre_direction)
        self.total_space = self._build_total_space()

    # ======================================================
    # === Points of the total space
    # ======================================================

    def bundle_points(self, param_points, vector_components) -> jnp.ndarray:
        """
        Assemble points of the total space out of base points and fibre components.

        Args:
            param_points: Array (..., base_dim) — base points in parameter space.
            vector_components: Array (..., base_dim) — components w.r.t. the
                parameter-space basis.

        Returns:
            Array (..., 2 * base_dim).
        """
        param_points = jnp.atleast_2d(jnp.asarray(param_points))
        vector_components = jnp.atleast_2d(jnp.asarray(vector_components))
        return jnp.concatenate([param_points, vector_components], axis=-1)

    def projection(self, bundle_points) -> jnp.ndarray:
        """
        The projection map pi: (theta ; u) -> theta.

        Args:
            bundle_points: Array (..., 2 * base_dim).

        Returns:
            Array (..., base_dim) with the base points in parameter space.
        """
        bundle_points = jnp.atleast_2d(jnp.asarray(bundle_points))
        return bundle_points[..., : self.base_dim]

    def fibre_components(self, bundle_points) -> jnp.ndarray:
        """
        The fibre block of a point of the total space, i.e. the components of the
        tangent vector w.r.t. the parameter-space basis.
        """
        bundle_points = jnp.atleast_2d(jnp.asarray(bundle_points))
        return bundle_points[..., self.base_dim :]

    def fibre(self, param_point, fibre_values) -> jnp.ndarray:
        """
        Sample the fibre T_pM = preim_pi({p}) over a base point.

        Args:
            param_point: Array (base_dim,) — the base point p in parameter space.
            fibre_values: Array (N,) (1-dimensional base) or (N, base_dim) — the fibre
                components to sample.

        Returns:
            Array (N, 2 * base_dim) with points of the total space, all projecting to p.
        """
        fibre_values = jnp.asarray(fibre_values)
        if fibre_values.ndim == 1:
            fibre_values = fibre_values.reshape(-1, self.base_dim)

        param_point = jnp.asarray(param_point).reshape(1, self.base_dim)
        param_points = jnp.broadcast_to(param_point, (fibre_values.shape[0], self.base_dim))

        return self.bundle_points(param_points, fibre_values)

    def zero_section(self, param_points) -> jnp.ndarray:
        """
        The zero section, p -> (p ; 0), whose image is the copy of the base manifold
        sitting inside the total space.
        """
        param_points = jnp.atleast_2d(jnp.asarray(param_points))
        return self.bundle_points(param_points, jnp.zeros_like(param_points))

    def section_of_field(self, field, param_points) -> jnp.ndarray:
        """
        The image of a vector field, seen as a section of this bundle:

            X : M ---> TM ,   p ---> ( p ; X^i_param(p) ).

        Args:
            field: VectorField object on the same manifold.
            param_points: Array (N, base_dim) — base points in parameter space.

        Returns:
            Array (N, 2 * base_dim) with points of the total space.
        """
        if field.manifold != self.manifold:
            raise ValueError("The field must be defined on the base manifold of the bundle.")

        param_points = jnp.atleast_2d(jnp.asarray(param_points))
        components = jnp.atleast_2d(field.evaluate_in_param_space(param_points))

        return self.bundle_points(param_points, components)

    # ======================================================
    # === Bundle charts induced by charts of the base
    # ======================================================

    def chart_map(self, chart, bundle_points) -> jnp.ndarray:
        """
        The bundle chart xi_X induced by a chart (U, X) of the base:

            xi_X( v_{gamma,p} ) = ( X^i(p) ; (dX^j)_p(v_{gamma,p}) ),

        i.e. the base block is mapped with the chart map and the fibre block with the
        Jacobian of the chart map at the base point.

        Args:
            chart: Chart object on the base manifold.
            bundle_points: Array (N, 2 * base_dim).

        Returns:
            Array (N, 2 * base_dim) with the bundle-chart coordinates.
        """
        if chart.manifold != self.manifold:
            raise ValueError("The chart must belong to the base manifold of the bundle.")

        bundle_points = jnp.atleast_2d(jnp.asarray(bundle_points))
        param_points = self.projection(bundle_points)
        components = self.fibre_components(bundle_points)

        chart_coordinates = jax.vmap(chart.map_to_chart)(param_points)              # (N, d)
        jacobians = jax.vmap(jax.jacrev(chart.chart_map))(param_points)             # (N, d, d)
        chart_components = jnp.einsum('nij,nj->ni', jacobians, components)          # (N, d)

        return jnp.concatenate([chart_coordinates, chart_components], axis=-1)

    def inverse_chart_map(self, chart, chart_points) -> jnp.ndarray:
        """
        The inverse bundle chart xi_X^{-1}, which reads the base point off the first
        block and the fibre components off the second one.

        Args:
            chart: Chart object on the base manifold, with an inverse_chart_map.
            chart_points: Array (N, 2 * base_dim) — bundle-chart coordinates.

        Returns:
            Array (N, 2 * base_dim) with points of the total space.
        """
        if chart.inverse_chart_map is None:
            raise ValueError("The chart needs an inverse_chart_map to invert the bundle chart.")

        chart_points = jnp.atleast_2d(jnp.asarray(chart_points))
        chart_coordinates = chart_points[..., : self.base_dim]
        chart_components = chart_points[..., self.base_dim :]

        param_points = jax.vmap(chart.inverse_chart_map)(chart_coordinates)         # (N, d)
        jacobians = jax.vmap(jax.jacrev(chart.chart_map))(param_points)             # (N, d, d)
        components = jnp.einsum('nij,nj->ni', jnp.linalg.inv(jacobians), chart_components)  # (N, d)

        return self.bundle_points(param_points, components)

    def chart_transition(self, chart_target, chart_source, chart_points) -> jnp.ndarray:
        """
        The transition map between two bundle charts,
        xi_Y o xi_X^{-1}, which splits into a base block and a fibre-wise linear block:

            ( alpha ; beta ) ---> ( T_VU(alpha) ; (J_VU)^j_k (p) * beta^k ).

        Args:
            chart_target: Chart (V, Y) — the bundle chart mapped *into*.
            chart_source: Chart (U, X) — the bundle chart mapped *from*.
            chart_points: Array (N, 2 * base_dim) — coordinates w.r.t. xi_X.

        Returns:
            Array (N, 2 * base_dim) — coordinates w.r.t. xi_Y.
        """
        bundle_points = self.inverse_chart_map(chart_source, chart_points)
        return self.chart_map(chart_target, bundle_points)

    # ======================================================
    # === Drawing of the total space
    # ======================================================

    def _resolve_fibre_direction(self, fibre_direction):
        """Normalize the fibre-direction argument into a callable of the base point."""
        if fibre_direction is None:
            default_direction = jnp.zeros(self.ambient_dim).at[-1].set(1.0)
            return lambda params: default_direction

        if callable(fibre_direction):
            return fibre_direction

        fixed_direction = jnp.asarray(fibre_direction)
        return lambda params: fixed_direction

    def _build_total_space(self):
        """
        Build the Manifold object that *draws* the total space: its parameter space is
        the bundle parameter space (theta ; u) and its embedding is

            Phi_TM(theta ; u) = Phi(theta) + fibre_scale * u * n(theta).

        Only implemented for a 1-dimensional base embedded in R^3 (the case of the
        figures), where the result is a ruled surface -- a cylinder, for a planar ring.
        """
        if not (self.base_dim == 1 and self.ambient_dim == 3):
            raise NotImplementedError(
                "The drawing of the total space is only implemented for a 1-dimensional "
                f"base in R^3, got (dim, ambient_dim) = ({self.base_dim}, {self.ambient_dim})."
            )

        base_embedding = self.manifold.embedding_func
        fibre_direction = self.fibre_direction
        fibre_scale = self.fibre_scale

        def total_space_embedding(bundle_point):
            param_point = bundle_point[: self.base_dim]
            fibre_component = bundle_point[self.base_dim]

            base = base_embedding(param_point)
            direction = fibre_direction(param_point)
            direction = direction / jnp.linalg.norm(direction)

            return base + fibre_scale * fibre_component * direction

        return manifolds.Manifold(
            name=self.name,
            dim=self.dim,
            ambient_dim=self.ambient_dim,
            embedding_func=total_space_embedding,
        )

    def embed(self, bundle_points) -> jnp.ndarray:
        """
        Embed points of the total space in the ambient space, using the drawing above.

        Args:
            bundle_points: Array (..., 2 * base_dim).

        Returns:
            Array (..., ambient_dim).
        """
        return self.total_space.embed(jnp.atleast_2d(jnp.asarray(bundle_points)))
