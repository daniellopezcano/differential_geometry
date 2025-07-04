import jax
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Polygon, MultiPolygon


def circular_vector_field_parametric(point):
    """
    Radial vector field in parameter space: v = (x, y)
    """
    x, y = point
    return jnp.array([-y, x])

def constant_direction_vector_field_parametric(direction=(1.0, 0.0)):
    """
    Returns a function representing a constant vector field
    in the specified direction in parameter space.

    Args:
        direction: Tuple (dx, dy) specifying the constant direction.

    Returns:
        A function point -> jnp.array([dx, dy]) that ignores the point input.
    """
    direction = jnp.array(direction)

    def vector_field(point):
        return direction

    return vector_field