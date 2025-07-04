# Differential Geometry Tools
==========================

A Python package for differential geometry utilities, manifolds, and visualization tools.

## Overview
-----------

This package provides a collection of tools and utilities for working with differential geometry, including manifolds, charts, and curves. It also includes visualization tools for plotting and exploring these geometric objects.

## Features
------------

*   **Manifolds**: Define and manipulate manifolds, including charts and transition functions.
*   **Curves**: Define and manipulate curves on manifolds, including tangent vectors and geodesics.
*   **Visualization**: Plot and visualize manifolds, curves, and other geometric objects using matplotlib and plotly.
*   **Utilities**: Various utility functions for working with differential geometry, including numerical computations and data structures.

## Installation
------------

To install this package, run the following command:

```bash
pip install -e .
```

## Usage
-----

### Manifolds

To define a manifold, use the `Manifold` class:

```python
from differential_geometry.manifolds import Manifold

# Define a 2D manifold
manifold = Manifold(dim=2)
```

You can then define charts and transition functions on the manifold:

```python
from differential_geometry.charts import Chart

# Define a chart on the manifold
chart = Chart(manifold, dim=2)

# Define a transition function between charts
transition_function = chart.transition_function(another_chart)
```

The `Manifold` class has several methods for working with manifolds, including:

*   `add_chart`: Add a new chart to the manifold.
*   `get_chart`: Get a chart from the manifold by name or index.
*   `transition_function`: Compute the transition function between two charts.
*   `coordinate_transform`: Transform coordinates between two charts.

### Charts

The `Chart` class represents a chart on a manifold. It has several methods for working with charts, including:

*   `add_coordinate`: Add a new coordinate to the chart.
*   `get_coordinate`: Get a coordinate from the chart by name or index.
*   `transition_function`: Compute the transition function between two charts.

### Curves

To define a curve on a manifold, use the `Curve` class:

```python
from differential_geometry.curves import Curve

# Define a curve on the manifold
curve = Curve(manifold, dim=1)
```

You can then compute tangent vectors and geodesics on the curve:

```python
from differential_geometry.tangent_vectors import TangentVector

# Compute the tangent vector at a point on the curve
tangent_vector = TangentVector(curve, point)
```

The `Curve` class has several methods for working with curves, including:

*   `add_point`: Add a new point to the curve.
*   `get_point`: Get a point from the curve by index.
*   `tangent_vector`: Compute the tangent vector at a point on the curve.
*   `geodesic`: Compute the geodesic between two points on the curve.

### Tangent Vectors

The `TangentVector` class represents a tangent vector at a point on a curve. It has several methods for working with tangent vectors, including:

*   `add_component`: Add a new component to the tangent vector.
*   `get_component`: Get a component from the tangent vector by name or index.
*   `norm`: Compute the norm of the tangent vector.

### Geodesics

The `Geodesic` class represents a geodesic between two points on a curve. It has several methods for working with geodesics, including:

*   `add_point`: Add a new point to the geodesic.
*   `get_point`: Get a point from the geodesic by index.
*   `length`: Compute the length of the geodesic.

### Visualization

To visualize a manifold or curve, use the `plot` function:

```python
from differential_geometry.plotting_tools import plot

# Plot the manifold
plot(manifold)

# Plot the curve
plot(curve)
```

The `plot` function has several options for customizing the plot, including:

*   `axis`: Specify the axis to plot on.
*   `title`: Specify the title of the plot.
*   `xlabel`: Specify the x-axis label.
*   `ylabel`: Specify the y-axis label.
*   `zlabel`: Specify the z-axis label.
*   `color`: Specify the color of the plot.
*   `marker`: Specify the marker type for the plot.
*   `linestyle`: Specify the line style for the plot.

You can also use the `plotly` library to create interactive plots:

```python
import plotly.graph_objs as go

# Create a plotly figure
fig = go.Figure(data=[go.Surface(x=manifold.x, y=manifold.y, z=manifold.z)])

# Show the plot