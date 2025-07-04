import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import jax.numpy as jnp
import numpy as np

from plotting_tools.basic_plotting_tools import Arrow3D

def plot_manifold(
    ax,
    manifold,
    resolution=100,
    xmin=-jnp.pi, xmax=jnp.pi,
    ymin=-jnp.pi, ymax=jnp.pi,
    color="skyblue",
    alpha=1.0,
    edgecolor="none",
    label=None,
    label_position="center",
    label_fontsize=20,
    function=None,
    cmap="viridis",
    label_function=None,
):
    """
    Clean-style manifold plotter for supported embeddings:
      - 1→2: curve in 2D
      - 1→3: curve in 3D
      - 2→2: surface in 2D (via contours)
      - 2→3: surface in 3D
    3→3 is not implemented.

    Args:
        ax: Matplotlib axis (2D or 3D).
        manifold: Manifold object.
        resolution: Number of grid/line points.
        xmin, xmax, ymin, ymax: Parameter space limits.
        color: Surface or line color.
        alpha: Transparency.
        edgecolor: Grid mesh line color.
        label: Optional text label for the manifold.
        label_position: Label placement on plot.
        label_fontsize: Font size for label text.
        function: Optional scalar function on parameter space.
        cmap: Colormap for scalar shading.
        label_function: Label for colorbar.
    """
    param_dim = manifold.param_dim
    ambient_dim = manifold.ambient_dim

    if (param_dim, ambient_dim) == (3, 3):
        raise NotImplementedError("3D volume rendering is not implemented (TODO).")

    def clean_3d_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.grid(False)
        ax.set_box_aspect([1, 1, 1])  # Force equal aspect

        # Remove panes
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        # Remove the edges of the box
        ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # New (ensure outer lines also invisible)
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
            axis.label.set_color((1.0, 1.0, 1.0, 0.0))
            axis.major_ticks = []


    def clean_2d_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # === 1D domains ===
    if param_dim == 1:
        t = jnp.linspace(xmin, xmax, resolution).reshape(-1, 1)
        embedded = manifold.embed(t)

        if ambient_dim == 2:
            ax.plot(embedded[:, 0], embedded[:, 1], color=color, alpha=alpha)
            clean_2d_axes(ax)

        elif ambient_dim == 3:
            ax.plot3D(embedded[:, 0], embedded[:, 1], embedded[:, 2], color=color, alpha=alpha)
            clean_3d_axes(ax)

        if label:
            idx = resolution // 2
            pos = embedded[idx]
            ax.text(*pos, label, fontsize=label_fontsize, color=color)

    # === 2D domains ===
    elif param_dim == 2:
        x = jnp.linspace(xmin, xmax, resolution)
        y = jnp.linspace(ymin, ymax, resolution)
        XX, YY = jnp.meshgrid(x, y)
        param_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)
        embedded = manifold.embed(param_points)

        if ambient_dim == 2:
            X = embedded[:, 0].reshape(XX.shape)
            Y = embedded[:, 1].reshape(XX.shape)
            if function is not None:
                F_vals = function.evaluate_in_param_space(param_points).reshape(XX.shape)
                cmap_obj = cm.get_cmap(cmap)
                norm = mcolors.Normalize(vmin=float(F_vals.min()), vmax=float(F_vals.max()))
                ax.contourf(X, Y, F_vals, levels=50, cmap=cmap_obj)
                cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_obj), ax=ax)
                cbar.set_label(label_function or "Function value", fontsize=label_fontsize)
            else:
                ax.contour(X, Y, levels=10, colors=color, linewidths=1)
            clean_2d_axes(ax)

        elif ambient_dim == 3:
            X = embedded[:, 0].reshape(XX.shape)
            Y = embedded[:, 1].reshape(XX.shape)
            Z = embedded[:, 2].reshape(XX.shape)

            if function is not None:
                F_vals = function.evaluate_in_param_space(param_points).reshape(XX.shape)
                norm = mcolors.Normalize(vmin=float(F_vals.min()), vmax=float(F_vals.max()))
                cmap_obj = cm.get_cmap(cmap)
                face_colors = cmap_obj(norm(F_vals))

                surface = ax.plot_surface(
                    X, Y, Z,
                    facecolors=face_colors,
                    edgecolor=edgecolor,
                    linewidth=0.,
                    antialiased=False,
                    shade=False,
                    alpha=alpha
                )

                mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
                mappable.set_array(F_vals)
                cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
                cbar.set_label(label_function or "Function value", fontsize=label_fontsize)

            else:
                surface = ax.plot_surface(
                    X, Y, Z,
                    color=color,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    antialiased=True,
                    linewidth=0.2 if edgecolor != "none" else 0
                )

            clean_3d_axes(ax)

            if label is not None:
                X_flat, Y_flat, Z_flat = X.ravel(), Y.ravel(), Z.ravel()
                if label_position == "center":
                    pos = (jnp.mean(X_flat), jnp.mean(Y_flat), jnp.mean(Z_flat))
                elif label_position == "top":
                    idx = jnp.argmax(Z_flat)
                    pos = (X_flat[idx], Y_flat[idx], Z_flat[idx])
                elif label_position == "bottom":
                    idx = jnp.argmin(Z_flat)
                    pos = (X_flat[idx], Y_flat[idx], Z_flat[idx])
                elif label_position == "right":
                    idx = jnp.argmax(X_flat)
                    pos = (X_flat[idx], Y_flat[idx], Z_flat[idx])
                elif label_position == "left":
                    idx = jnp.argmin(X_flat)
                    pos = (X_flat[idx], Y_flat[idx], Z_flat[idx])
                else:
                    raise ValueError(f"Invalid label_position: {label_position}")

                ax.text(*pos, label, fontsize=label_fontsize, color="black" if function else color)

    else:
        raise NotImplementedError(f"Plotting for (param_dim={param_dim}, ambient_dim={ambient_dim}) is not implemented.")

def plot_tangent_vectors(
    ax,
    embedded_points,
    jacobians,
    scale=0.5,
    colors=None,
    arrowstyle='-|>',
    mutation_scale=15,
    linewidth=1.5,
    alpha=1.0,
    draw_points=True,
    point_color="black",
    point_size=20,
    labels=None,
    label_fontsize=20,
    label_offset=0.05,
):
    """
    Plot tangent vectors in 2D or 3D using arrows and optional labels.

    Args:
        ax: Matplotlib axis (2D or 3D).
        embedded_points: Array (N, D) — manifold points in ambient space.
        jacobians: Array (N, param_dim, D) — tangent vectors in ambient space.
        scale: Scaling factor for arrow lengths.
        colors: Optional (N, param_dim) list of colors.
        arrowstyle: Arrow style for FancyArrowPatch (3D only).
        mutation_scale: Arrowhead size for FancyArrowPatch (3D only).
        linewidth: Line width of arrows.
        alpha: Opacity.
        draw_points: If True, mark base points with scatter.
        point_color: Color for base points.
        point_size: Marker size for base points.
        labels: Optional list of (N, param_dim) strings to label each arrow.
        label_fontsize: Font size for labels.
        label_offset: Offset multiplier for text position along arrow direction.
    """
    import numpy as np

    N, D = embedded_points.shape
    _, param_dim, D2 = jacobians.shape
    assert D == D2, f"Ambient dim mismatch: got {D} vs {D2}"

    for i in range(N):
        base = np.array(embedded_points[i])
        for j in range(param_dim):
            direction = np.array(jacobians[i, j]) * scale
            tip = base + direction
            color = colors[i][j] if colors else "black"

            if D == 2:
                ax.quiver(
                    base[0], base[1],
                    direction[0], direction[1],
                    angles='xy', scale_units='xy', scale=1.0,
                    color=color, width=0.008, alpha=alpha
                )
                if labels:
                    offset_pos = tip + label_offset * direction
                    ax.text(
                        offset_pos[0], offset_pos[1],
                        labels[i][j],
                        fontsize=label_fontsize,
                        color=color
                    )

            elif D == 3:
                arrow = Arrow3D(
                    [base[0], tip[0]],
                    [base[1], tip[1]],
                    [base[2], tip[2]],
                    mutation_scale=mutation_scale,
                    lw=linewidth,
                    arrowstyle=arrowstyle,
                    color=color,
                    alpha=alpha
                )
                ax.add_artist(arrow)

                if labels:
                    offset_pos = tip + label_offset * direction
                    ax.text(
                        offset_pos[0], offset_pos[1], offset_pos[2],
                        labels[i][j],
                        fontsize=label_fontsize,
                        color=color
                    )

            else:
                raise NotImplementedError(f"Only D=2 or D=3 supported, got D={D}")

    if draw_points:
        if D == 2:
            ax.scatter(embedded_points[:, 0], embedded_points[:, 1],
                       color=point_color, s=point_size, zorder=3)
        elif D == 3:
            ax.scatter(embedded_points[:, 0], embedded_points[:, 1], embedded_points[:, 2],
                       color=point_color, s=point_size)

def plot_tangent_spaces(
    ax,
    embedded_points,
    jacobians,
    size=1.0,
    color="lightgray",
    colors=None,
    alpha=0.5,
    resolution=10,
    line_width=2.0,
    labels=None,
    label_fontsize=10,
    label_offset=0.1,
):
    """
    Plot tangent lines (for 1D) or planes (for 2D) in ambient space.

    Args:
        ax: Matplotlib axis (2D or 3D).
        embedded_points: (N, D) — manifold points in R^D.
        jacobians: (N, param_dim, D) — tangent vectors at those points.
        size: Length/extent of tangent lines or planes.
        color: Default color if `colors` is not provided.
        colors: Optional list of shape (N,) with custom color per tangent space.
        alpha: Transparency for planes.
        resolution: Plane resolution (grid).
        line_width: Width of tangent lines.
        labels: Optional list (N,) with one label per tangent space.
        label_fontsize: Font size for labels.
        label_offset: Multiplier for text label offset along direction.
    """
    import numpy as np

    N, D = embedded_points.shape
    _, param_dim, D2 = jacobians.shape
    assert D == D2, f"Ambient dim mismatch: got {D} vs {D2}"
    assert param_dim in [1, 2], "Only param_dim 1 or 2 supported."

    for i in range(N):
        base = np.array(embedded_points[i])
        local_color = colors[i] if colors else color

        if param_dim == 1:
            v = np.array(jacobians[i, 0]) / np.linalg.norm(jacobians[i, 0])
            t = np.linspace(-size, size, 10)
            points = base[np.newaxis, :] + t[:, np.newaxis] * v[np.newaxis, :]

            if D == 2:
                ax.plot(points[:, 0], points[:, 1], color=local_color, lw=line_width)
                if labels:
                    tip = base + size * v
                    offset = tip + label_offset * v
                    ax.text(offset[0], offset[1], labels[i],
                            fontsize=label_fontsize, color=local_color)
            elif D == 3:
                ax.plot(points[:, 0], points[:, 1], points[:, 2], color=local_color, lw=line_width)
                if labels:
                    tip = base + size * v
                    offset = tip + label_offset * v
                    ax.text(offset[0], offset[1], offset[2], labels[i],
                            fontsize=label_fontsize, color=local_color)

        elif param_dim == 2:
            assert D == 3, "Tangent planes only supported in 3D"
            v1 = np.array(jacobians[i, 0])
            v2 = np.array(jacobians[i, 1])

            s = np.linspace(-size, size, resolution)
            t = np.linspace(-size, size, resolution)
            S, T = np.meshgrid(s, t)

            plane = (
                base.reshape(3, 1, 1)
                + S * v1.reshape(3, 1, 1)
                + T * v2.reshape(3, 1, 1)
            )

            X, Y, Z = plane[0], plane[1], plane[2]
            ax.plot_surface(X, Y, Z, color=local_color, alpha=alpha, edgecolor="none")

            if labels:
                center = base + 0.5 * v1 + 0.5 * v2 + label_offset * (v1 + v2)
                ax.text(center[0], center[1], center[2], labels[i],
                        fontsize=label_fontsize, color=local_color)

