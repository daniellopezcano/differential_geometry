import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, tri
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import proj3d

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import Delaunay

import tol_colors as tc


# ---------------------------------------------------------------------------
# Style and colormaps
# ---------------------------------------------------------------------------

def _resolve_cmap(cmap):
    """Accept None (use rc default), a colormap name, or a Colormap object."""
    if cmap is None:
        return mpl.colormaps[mpl.rcParams["image.cmap"]]
    if isinstance(cmap, mcolors.Colormap):
        return cmap                        # e.g. tc.sunset, tc.colormaps["sunset"]
    if isinstance(cmap, str):
        return mpl.colormaps[cmap]         # e.g. "viridis"
    raise TypeError(f"cmap must be None, a str or a matplotlib Colormap, got {type(cmap)}")


def set_plot_style(usetex=False, font_size=16, color_cycle=None, default_cmap=None):
    """Apply the project-wide Matplotlib style and return the color cycle.

    Parameters
    ----------
    usetex : bool
        Render text with LaTeX (requires a working LaTeX install).
    font_size : int
        Base font size.
    color_cycle : iterable of colors, optional
        Line color cycle. Defaults to Paul Tol's 'bright' scheme.
    default_cmap : str or Colormap, optional
        e.g. tc.sunset or tc.PRGn. Sets the default for imshow/pcolormesh/
        scatter and for plot_parametric_function_in_chart_coordinates.

    Returns
    -------
    list
        The colors in the cycle, so specific ones can be picked by index.
    """
    colors = list(tc.bright) if color_cycle is None else list(color_cycle)

    rc = {
        "axes.prop_cycle": plt.cycler(color=colors),
        "text.usetex": usetex,
        "font.size": font_size,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    }

    if default_cmap is not None:
        cmap = _resolve_cmap(default_cmap)
        # rcParams needs a registered name; the prefix avoids clashing with
        # Matplotlib's own maps of the same name (e.g. its built-in "PRGn").
        name = cmap.name if cmap.name.startswith("tol_") else f"tol_{cmap.name}"
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=name)
        rc["image.cmap"] = name

    mpl.rcParams.update(rc)
    return colors

class Arrow3D(FancyArrowPatch):
    """A 3D arrow for Matplotlib 3D axes."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = np.asarray(xs), np.asarray(ys), np.asarray(zs)

    def _project(self):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return zs

    def draw(self, renderer):
        self._project()
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        return np.mean(self._project())
