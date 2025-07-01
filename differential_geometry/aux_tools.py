import jax
import jax.numpy as jnp
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

def surface_embedding_factory(
    amplitude=1.0,
    center=(0.0, 0.0),
    sigma=(jnp.pi, jnp.pi)
):
    """
    Creates an embedding function for the manifold.

    The manifold is shaped by an 'egg-carton' surface modulated by a Gaussian envelope.

    Args:
        amplitude: Amplitude of the Gaussian envelope.
        center: Tuple (mu_x, mu_y) — center of the Gaussian.
        sigma: Tuple (sigma_x, sigma_y) — standard deviations of the Gaussian.

    Returns:
        embedding function: params (x, y) → (x, y, z)
    """

    mu_x, mu_y = center
    sigma_x, sigma_y = sigma

    def embedding(params):
        x, y = params

        # Egg-carton surface
        z_eggs = (
            jnp.cos(x + y) / 2 +
            jnp.sin(x - y + jnp.pi / 2) / 2 +
            1
        )

        # Gaussian envelope
        z_gaussian = amplitude * jnp.exp(
            -(((x - mu_x)**2) / (2 * sigma_x**2) +
              ((y - mu_y)**2) / (2 * sigma_y**2))
        )

        z = z_eggs * z_gaussian

        return jnp.array([x, y, z])

    return embedding

def cartesian_chart_map(params):
    """
    Chart map for U: Identity map (x, y) → (x, y).
    """
    return params

def polar_chart_map(params, center=(0.0, 0.0)):
    """
    Chart map for V: Maps (x, y) in param space to (r, θ) in polar coordinates.

    Args:
        params: Array of shape (N, 2) or (2,) — parameter points.
        center: (x_center, y_center) — pole of the polar chart.

    Returns:
        Array of shape (N, 2) or (2,) with (r, θ) in chart space.
        Returns NaNs for points that are ill-defined.
    """
    params = jnp.atleast_2d(params)

    x = params[:, 0] - center[0]
    y = params[:, 1] - center[1]

    # Check validity: finite x and y
    valid = jnp.isfinite(x) & jnp.isfinite(y)

    # Compute r and theta normally
    r = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(y, x)

    # Where invalid, set r and theta to NaN
    r = jnp.where(valid, r, jnp.nan)
    theta = jnp.where(valid, theta, jnp.nan)

    result = jnp.stack([r, theta], axis=-1)

    return result if params.shape[0] > 1 else result[0]

def boundary_ellipse_in_param_space(lambdas, center=(1.0, 0.0), axes=(1.5, 1.0)):
    """
    Parametric boundary of an ellipse in parameter space.

    Args:
        lambdas (array): Angles [0, 2π] parameterizing the ellipse.
        center (tuple): (x_center, y_center) of the ellipse.
        axes (tuple): (a, b) semi-axes along x and y directions.

    Returns:
        Array of shape (N, 2) with (x, y) boundary points.
    """
    x_center, y_center = center
    a, b = axes
    x = a * jnp.cos(lambdas) + x_center
    y = b * jnp.sin(lambdas) + y_center
    return jnp.stack([x, y], axis=-1)

def wiggly_curve_factory(p_x, p_y, amplitude=jnp.pi/6, frequency=jnp.pi):
    """
    Returns a parametric curve function γ(λ) passing through (p_x, p_y) at λ = 0.

    Args:
        p_x, p_y: Base point in parameter space.
        amplitude: Amplitude of the oscillation in x-direction.
        frequency: Frequency of the sine oscillation.

    Returns:
        Function curve(λ) → (x(λ), y(λ))
    """
    def curve(lambda_values):
        x = p_x + amplitude * jnp.sin(frequency * lambda_values)
        y = p_y + lambda_values
        return jnp.stack([x, y], axis=-1)
    
    return curve

def arching_curve_factory(p_x, p_y, amplitude=jnp.pi/6, frequency=jnp.pi/3):
    """
    Returns a parametric curve function δ(λ) passing through (p_x, p_y) at λ = 0.

    Args:
        p_x, p_y: Base point in parameter space.
        amplitude: Amplitude of the cosine arch in y-direction.
        frequency: Frequency of the cosine oscillation.

    Returns:
        Function curve(λ) → (x(λ), y(λ))
    """
    def curve(lambda_values):
        x = p_x + lambda_values
        y = p_y + amplitude * (jnp.cos(frequency * lambda_values) - 1.0)  # Shifted cosine
        return jnp.stack([x, y], axis=-1)
    
    return curve

def vertical_line_curve_factory(p_x, p_y):
    """
    Returns a function representing the vertical line x = p_x.

    Args:
        p_x: x-coordinate (fixed).
        p_y: y-coordinate the curve passes through (lambda=0 reference point).

    Returns:
        A parametric function of lambda (λ) where y = λ + p_y, x = p_x.
    """
    def curve(lambda_values):
        x = jnp.full_like(lambda_values, p_x)
        y = p_y + lambda_values
        return jnp.stack([x, y], axis=-1)
    return curve

def circle_curve_factory(p_x, p_y):
    """
    Returns a function representing a circle centered at (0, 0) 
    with radius sqrt(p_x^2 + p_y^2), passing through (p_x, p_y).

    Args:
        p_x: x-coordinate.
        p_y: y-coordinate.

    Returns:
        A parametric function of lambda (angle parameter λ).
    """
    r = jnp.sqrt(p_x**2 + p_y**2)

    def curve(lambda_values):
        x = r * jnp.cos(lambda_values)
        y = r * jnp.sin(lambda_values)
        return jnp.stack([x, y], axis=-1)

    return curve



def compute_tangent_plane(manifold, p_param):
    """
    Compute the tangent plane basis vectors at a point p on the manifold.

    Args:
        manifold: Manifold object.
        p_param: Point in parameter space (shape (2,)).

    Returns:
        p_xyz: Embedded point on the manifold (shape (3,)).
        tangent_vec1: First tangent vector in R^3 (∂Φ/∂x).
        tangent_vec2: Second tangent vector in R^3 (∂Φ/∂y).
    """
    def embedding_fn(param):
        return manifold.embedding_func(param)

    jacobian = jax.jacrev(embedding_fn)(p_param)  # Shape (3, 2)

    tangent_vec1 = jacobian[:, 0]  # ∂Φ/∂x
    tangent_vec2 = jacobian[:, 1]  # ∂Φ/∂y

    p_xyz = manifold.embed(p_param)

    if p_xyz.shape[0] == 1:
        p_xyz = p_xyz[0]

    return p_xyz, tangent_vec1, tangent_vec2

def compute_chart_intersection(
    chart1_boundary, chart2_boundary
):
    """
    Compute the intersection polygon(s) of chart1_boundary and chart2_boundary.

    Args:
        chart1_boundary: (N, 2) array-like
        chart2_boundary: (M, 2) array-like

    Returns:
        A list of (K_i, 2) numpy arrays, each one corresponding to one polygon in the intersection.
        If the intersection is a single polygon, it's still returned as a list with one element.
    """
    poly1 = Polygon(np.array(chart1_boundary)).buffer(0)
    poly2 = Polygon(np.array(chart2_boundary)).buffer(0)

    intersection = poly1.intersection(poly2)

    if intersection.is_empty:
        raise ValueError("Charts do not intersect!")

    # Handle both Polygon and MultiPolygon
    if isinstance(intersection, Polygon):
        intersection_coords_list = [np.array(intersection.exterior.coords)]
    elif isinstance(intersection, MultiPolygon):
        intersection_coords_list = [
            np.array(part.exterior.coords) for part in intersection.geoms
        ]
    else:
        raise TypeError(f"Unexpected geometry type: {type(intersection)}")

    return intersection_coords_list

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