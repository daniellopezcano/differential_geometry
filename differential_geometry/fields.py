import jax
import jax.numpy as jnp


class Field:
    """
    Represents a vector field on the manifold.

    The field is defined by specifying its components in parameter space.
    """

    def __init__(self, manifold, vector_function):
        """
        Args:
            manifold: Manifold object with .embed().
            vector_function: Function mapping a point in parameter space (shape (param_dim,))
                             to a vector (shape (param_dim,)) in parameter space coordinates.
        """
        self.manifold = manifold
        self.vector_function = vector_function

    # ======================================================
    # === Field Evaluation in Parameter Space
    # ======================================================

    def evaluate_in_param_space(self, param_points):
        """
        Evaluate the vector field in parameter space.

        Args:
            param_points: Array of shape (N, param_dim) or (param_dim,)

        Returns:
            Array of shape (N, param_dim) or (param_dim,)
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
            param_points: Array of shape (N, param_dim) or (param_dim,)

        Returns:
            Array of shape (N, ambient_dim) or (ambient_dim,)
        """
        param_points = jnp.atleast_2d(param_points)
        single_input = param_points.shape[0] == 1

        # Compute Jacobians
        def embed_jacobian(point):
            return jax.jacrev(self.manifold.embedding_func)(point)  # (ambient_dim, param_dim)

        jacobians = jax.vmap(embed_jacobian)(param_points)  # (N, ambient_dim, param_dim)

        # Field in parameter space
        vectors_param = self.evaluate_in_param_space(param_points)  # (N, param_dim)

        if single_input:
            jacobian = jacobians[0]        # (ambient_dim, param_dim)
            vector = vectors_param         # (param_dim,)
            vector_ambient = jacobian @ vector  # (ambient_dim,)
        else:
            vector_ambient = jnp.einsum('nij,nj->ni', jacobians, vectors_param)  # (N, ambient_dim)

        return vector_ambient
