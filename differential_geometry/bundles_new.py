from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

import differential_geometry.manifolds as manifolds


class VectorBundle:
    """
    Common machinery of the rank-d vector bundles built over a manifold, TM and T*M.

    A point of the total space is represented by the pair

        (theta^1, ..., theta^d ; u_1, ..., u_d)  in  R^{2d},

    where the first block are the parameter-space coordinates of the base point p and
    the second block are the components of the fibre element w.r.t. the basis induced
    by the parameter space: d/d(theta^i) for TM, d(theta^i) for T*M. The projection
    forgets the second block.

    Bundle charts are induced by charts of the base, as in the notes: the base block is
    mapped with the chart map, the fibre block with a matrix built from the Jacobian of
    the chart map (the Jacobian itself for TM, its inverse transpose for T*M), which is
    the only thing subclasses have to provide (see _fibre_block_matrices).

    Drawing the total space requires one purely visual choice: an ambient direction
    along which each fibre is displayed. That drawing is stored in `self.total_space`,
    an ordinary Manifold object whose parameter space is the bundle parameter space
    above, so that all the existing plotting machinery applies to it.
    """

    def __init__(
        self,
        manifold,
        fibre_direction: Optional[Union[Callable, jnp.ndarray]] = None,
        fibre_scale: float = 1.0,
        name: str = r"$E$",
    ):
        """
        Args:
            manifold: Manifold object (the base space).
            fibre_direction: Ambient direction along which the fibres are drawn. Either
                a callable mapping a parameter point (shape (dim,)) to an ambient vector
                (shape (ambient_dim,)), or a fixed ambient vector. If None, each
                subclass provides its own default.
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

    def bundle_points(self, param_points, fibre_values) -> jnp.ndarray:
        """
        Assemble points of the total space out of base points and fibre components.

        Args:
            param_points: Array (..., base_dim) — base points in parameter space.
            fibre_values: Array (..., base_dim) — fibre components w.r.t. the basis
                induced by the parameter space.

        Returns:
            Array (..., 2 * base_dim).
        """
        param_points = jnp.atleast_2d(jnp.asarray(param_points))
        fibre_values = jnp.atleast_2d(jnp.asarray(fibre_values))
        return jnp.concatenate([param_points, fibre_values], axis=-1)

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
        """The fibre block of a point of the total space."""
        bundle_points = jnp.atleast_2d(jnp.asarray(bundle_points))
        return bundle_points[..., self.base_dim :]

    def fibre(self, param_point, fibre_values) -> jnp.ndarray:
        """
        Sample the fibre preim_pi({p}) over a base point.

        Args:
            param_point: Array (base_dim,) — the base point p in parameter space.
            fibre_values: Array (N,) (1-dimensional base) or (N, base_dim).

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
        The image of a field, seen as a section of this bundle:

            p ---> ( p ; field components at p ).

        Args:
            field: Field object on the base manifold, with evaluate_in_param_space
                (a VectorField for TM, a CovectorField for T*M).
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

    def _fibre_block_matrices(self, chart, param_points) -> jnp.ndarray:
        """
        Matrices M(p) such that (chart fibre components) = M(p) @ (parameter-space
        fibre components). To be provided by each subclass.
        """
        raise NotImplementedError

    def chart_map(self, chart, bundle_points) -> jnp.ndarray:
        """
        The bundle chart induced by a chart (U, X) of the base:
        base block mapped with the chart map, fibre block with _fibre_block_matrices.

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

        chart_coordinates = jax.vmap(chart.map_to_chart)(param_points)            # (N, d)
        matrices = self._fibre_block_matrices(chart, param_points)                # (N, d, d)
        chart_components = jnp.einsum('nij,nj->ni', matrices, components)         # (N, d)

        return jnp.concatenate([chart_coordinates, chart_components], axis=-1)

    def inverse_chart_map(self, chart, chart_points) -> jnp.ndarray:
        """
        The inverse bundle chart, which reads the base point off the first block and
        the fibre components off the second one.

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

        param_points = jax.vmap(chart.inverse_chart_map)(chart_coordinates)       # (N, d)
        matrices = self._fibre_block_matrices(chart, param_points)                # (N, d, d)
        components = jnp.einsum('nij,nj->ni', jnp.linalg.inv(matrices), chart_components)

        return self.bundle_points(param_points, components)

    def chart_transition(self, chart_target, chart_source, chart_points) -> jnp.ndarray:
        """
        The transition map between two bundle charts, which splits into the base
        transition and a fibre-wise linear block.

        Args:
            chart_target: Chart (V, Y) — the bundle chart mapped *into*.
            chart_source: Chart (U, X) — the bundle chart mapped *from*.
            chart_points: Array (N, 2 * base_dim) — coordinates w.r.t. the source chart.

        Returns:
            Array (N, 2 * base_dim) — coordinates w.r.t. the target chart.
        """
        bundle_points = self.inverse_chart_map(chart_source, chart_points)
        return self.chart_map(chart_target, bundle_points)

    # ======================================================
    # === Drawing of the total space
    # ======================================================

    def _default_fibre_direction(self, params):
        """Default ambient direction along which the fibres are drawn."""
        return jnp.zeros(self.ambient_dim).at[-1].set(1.0)

    def _resolve_fibre_direction(self, fibre_direction):
        """Normalize the fibre-direction argument into a callable of the base point."""
        if fibre_direction is None:
            return self._default_fibre_direction

        if callable(fibre_direction):
            return fibre_direction

        fixed_direction = jnp.asarray(fibre_direction)
        return lambda params: fixed_direction

    def _build_total_space(self):
        """
        Build the Manifold object that *draws* the total space:

            Phi_E(theta ; u) = Phi(theta) + fibre_scale * u * n(theta).

        Only implemented for a 1-dimensional base embedded in R^3 (the case of the
        figures).
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


class TangentBundle(VectorBundle):
    """
    The tangent bundle  TM --pi--> M.

    The fibre block holds the components of a tangent vector w.r.t. d/d(theta^i), and
    the bundle chart induced by (U, X) is

        xi_X( v_{gamma,p} ) = ( X^i(p) ; (dX^j)_p(v_{gamma,p}) ),

    i.e. its fibre block is the Jacobian of the chart map (contravariant components).
    By default the fibres are drawn along the last ambient axis, so that the total
    space of a planar ring becomes a vertical cylinder.
    """

    def __init__(self, manifold, fibre_direction=None, fibre_scale: float = 1.0,
                 name: str = r"$T\mathcal{M}$"):
        super().__init__(manifold, fibre_direction=fibre_direction,
                         fibre_scale=fibre_scale, name=name)

    def _fibre_block_matrices(self, chart, param_points):
        # A^i_j = dX^i / dtheta^j
        return jax.vmap(jax.jacrev(chart.chart_map))(param_points)


class CotangentBundle(VectorBundle):
    """
    The cotangent bundle  T*M --pi*--> M.

    The fibre block holds the components of a covector w.r.t. d(theta^i), and the
    bundle chart induced by (U, X) is

        xi*_X( X_p ) = ( X^i(p) ; X_p( (d/dX^j)_p ) ),

    i.e. its fibre block is the inverse transpose of the Jacobian of the chart map
    (covariant components).

    By default the fibres are drawn along the unit vector orthogonal both to the curve
    and to the last ambient axis, i.e. radially for a ring lying in a horizontal plane.
    Together with the vertical fibres of TangentBundle, this lets both bundles be drawn
    in the same ambient space with transverse fibres (see embed_fibre_pairs).
    """

    def __init__(self, manifold, fibre_direction=None, fibre_scale: float = 1.0,
                 name: str = r"$T^*\mathcal{M}$"):
        super().__init__(manifold, fibre_direction=fibre_direction,
                         fibre_scale=fibre_scale, name=name)

    def _default_fibre_direction(self, params):
        tangent = jax.jacrev(self.manifold.embedding_func)(params)[:, 0]     # (ambient,)
        vertical = jnp.zeros(self.ambient_dim).at[-1].set(1.0)
        direction = jnp.cross(tangent, vertical)
        return direction / jnp.linalg.norm(direction)

    def _fibre_block_matrices(self, chart, param_points):
        # (A^{-1})^T, with A^i_j = dX^i / dtheta^j
        jacobians = jax.vmap(jax.jacrev(chart.chart_map))(param_points)
        return jnp.transpose(jnp.linalg.inv(jacobians), axes=(0, 2, 1))


# ==========================================================
# === Drawing TM and T*M together
# ==========================================================

def embed_fibre_pairs(
    tangent_bundle,
    cotangent_bundle,
    param_points,
    vector_components,
    covector_components,
) -> jnp.ndarray:
    """
    Embed pairs (v, X) in T_pM x T*_pM, drawing the two fibres over each base point
    along their own (transverse) directions:

        Phi(p) + [ offset of v in the drawing of TM ] + [ offset of X in the drawing of T*M ].

    For a planar ring with the default directions, the pairs over p fill the vertical
    half-plane spanned by the vertical tangent fibre and the radial cotangent fibre, and
    a pair of fields p -> (X_p, Theta_p) traces a curve whose vertical displacement is
    the vector field and whose radial displacement is the covector field.

    Args:
        tangent_bundle: TangentBundle object.
        cotangent_bundle: CotangentBundle object over the same manifold.
        param_points: Array (N, base_dim) — base points in parameter space.
        vector_components: Array (N, base_dim) — components w.r.t. d/d(theta^i).
        covector_components: Array (N, base_dim) — components w.r.t. d(theta^i).

    Returns:
        Array (N, ambient_dim).
    """
    if tangent_bundle.manifold != cotangent_bundle.manifold:
        raise ValueError("Both bundles must be defined over the same manifold.")

    param_points = jnp.atleast_2d(jnp.asarray(param_points))
    base = tangent_bundle.manifold.embed(param_points)

    tangent_offset = tangent_bundle.embed(
        tangent_bundle.bundle_points(param_points, vector_components)) - base
    cotangent_offset = cotangent_bundle.embed(
        cotangent_bundle.bundle_points(param_points, covector_components)) - base

    return base + tangent_offset + cotangent_offset
