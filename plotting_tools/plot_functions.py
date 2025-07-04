# Re-import necessary libraries after kernel reset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import Delaunay

# Define the updated function
def plot_parametric_function_in_chart_coordinates(
    ax,
    chart,
    param_space_data: np.ndarray,
    function_values: np.ndarray,
    cmap="viridis",
    alpha=0.8,
    linewidth=0.1,
    label=None,
    label_position="center",
    label_fontsize=14,
    simplices=None,
    inset_colorbar=True,
    inset_position=(0.02, 0.02),
    inset_size=(0.25, 0.02),
    colorbar_orientation="horizontal",
    label_colorbar=r"$f$",
    label_colorbar_position="top",
    inset_box_alpha=0.8,
    inset_box_color="white",
    inset_border_color="black"
):
    chart_map = chart.chart_map
    param_space_data = np.asarray(param_space_data)
    function_values = np.asarray(function_values)

    if chart.manifold.param_dim != 2:
        raise ValueError("Only implemented for 2D parameter spaces.")

    if simplices is None:
        tri = Delaunay(param_space_data)
        simplices = tri.simplices

    triangles_param = param_space_data[simplices]
    triangles_chart = jax.vmap(chart_map)(jnp.array(triangles_param))
    triangles_chart = np.array(triangles_chart)

    valid_mask = ~np.isnan(triangles_chart).any(axis=(1, 2))
    triangles_chart = triangles_chart[valid_mask]
    triangles_values = function_values[simplices][valid_mask]

    norm = mcolors.Normalize(vmin=function_values.min(), vmax=function_values.max())
    cmap_func = cm.get_cmap(cmap)

    for tri_pts, tri_vals in zip(triangles_chart, triangles_values):
        color = cmap_func(norm(np.mean(tri_vals)))
        ax.fill(
            tri_pts[:, 0], tri_pts[:, 1],
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            linewidth=linewidth
        )

    if label is not None and len(triangles_chart) > 0:
        all_pts = triangles_chart.reshape(-1, 2)
        x, y = all_pts[:, 0], all_pts[:, 1]

        if label_position == "center":
            pos = (jnp.mean(x), jnp.mean(y))
        elif label_position == "top":
            idx = jnp.argmax(y)
            pos = (x[idx], y[idx])
        elif label_position == "bottom":
            idx = jnp.argmin(y)
            pos = (x[idx], y[idx])
        elif label_position == "right":
            idx = jnp.argmax(x)
            pos = (x[idx], y[idx])
        elif label_position == "left":
            idx = jnp.argmin(x)
            pos = (x[idx], y[idx])
        else:
            raise ValueError(f"Invalid label_position: {label_position}")

        ax.text(
            pos[0], pos[1], label,
            fontsize=label_fontsize,
            color="black",
            ha="center", va="center"
        )

    if inset_colorbar:
        cbax = inset_axes(
            ax,
            width=inset_size[0],
            height=inset_size[1],
            loc='lower left',
            bbox_to_anchor=(inset_position[0], inset_position[1], 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_func)
        mappable.set_array(function_values)

        cbar = plt.colorbar(
            mappable,
            cax=cbax,
            orientation=colorbar_orientation
        )

        cbar.ax.tick_params(labelsize=8)
        cbar.outline.set_visible(False)

        for spine in cbax.spines.values():
            spine.set_edgecolor(inset_border_color)
            spine.set_linewidth(1.0)

        cbax.set_facecolor(inset_box_color)
        cbax.patch.set_alpha(inset_box_alpha)

        if label_colorbar:
            if colorbar_orientation == "horizontal":
                if label_colorbar_position == "top":
                    cbax.set_title(label_colorbar, fontsize=10, pad=4)
                elif label_colorbar_position == "bottom":
                    cbax.set_xlabel(label_colorbar, fontsize=10, labelpad=4)
                else:
                    raise ValueError("Label position for horizontal colorbar must be 'top' or 'bottom'.")
            else:
                if label_colorbar_position == "right":
                    cbax.set_ylabel(label_colorbar, fontsize=10, rotation=-90, labelpad=10)
                elif label_colorbar_position == "left":
                    cbax.yaxis.set_label_position("left")
                    cbax.set_ylabel(label_colorbar, fontsize=10, rotation=90, labelpad=10)
                else:
                    raise ValueError("Label position for vertical colorbar must be 'left' or 'right'.")
