from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm, matplotlib.colors
import numpy as np

def plot_spatial_values(iterable, 
                              x="x", y="y", value="value", category="category",
                              x_lim=(0, 100), y_lim=(0, 100),
                              bin_width=None, figsize=(20, 10),
                              cmap="bwr", interpolation="bicubic", metric="mean",
                              verbose=False, alpha=None, orient="v", save_path=None
                             ):
    """Takes an iterable providing positions and values and plots a heatmap of the values.
    Can take multiple different categories (e.g. cameras) in one iterable.
    
    Arguments:
        iterable: iterable yielding dict
            Returns a dict with the positions and values (default keys: x, y, value, category).
        x, y, value, category: string
            Keys for accessing the dicts returned by the iterable.
        x_lim, y_lim: tuple(number, number)
            The bounds of the heatmap (values lying outside will be clipped).
        bin_width: number
            The size of the single bins in the heatmap.
        figsize: tuple(int, int)
            figsize in inches passed to matplotlib.pyplot.
        cmap: matplotlib.colors.Colormap or string
            Colormap to be used for the heatmap.
        interpolation: string
            Interpolation passed to pyplot.imshow.
        metric: one of ("mean", "count", "sum", "var", "std", "sample_var")
            Specifies how to aggregate the values from the iterable.
        verbose: bool
            Whether to print additional output (e.g. the min/max coordinates observed).
        alpha: None or string (see metric)
            If given, this specifies the alpha value of the heatmap.
            E.g. metric="mean", alpha="count"
        orient: "h" or "v"
            Whether the different categories in the data will be horizontally or vertically aligned.
    """
    if bin_width is None:
        bin_width = max((y_lim[1] - y_lim[0]) / 20, (x_lim[1] - x_lim[0]) / 20)
    
    def make_empty_container():
        w = int((x_lim[1] - x_lim[0]) / bin_width) + 1
        h = int((y_lim[1] - y_lim[0]) / bin_width) + 1
        return (np.zeros(shape=(h, w), dtype=np.float32), # Running mean
                np.zeros(shape=(h, w), dtype=np.int), # Running count
                np.zeros(shape=(h, w), dtype=np.float32)) # M2 aggregator (see Welford)
    accumulators = defaultdict(make_empty_container)
    
    true_x_lim = +np.inf, -np.inf
    true_y_lim = +np.inf, -np.inf
    
    for row in iterable:
        s_x, s_y, s_value = row[x], row[y], row[value]
        true_x_lim = min(true_x_lim[0], s_x), max(true_x_lim[1], s_x)
        true_y_lim = min(true_y_lim[0], s_y), max(true_y_lim[1], s_y)
                                                  
        s_cat = None
        if category is not None:
            s_cat = row[category]
        
        s_x = min(max(x_lim[0], s_x), x_lim[1])
        s_y = min(max(y_lim[0], s_y), y_lim[1])
        bin_x = int((s_x - x_lim[0]) / bin_width)
        bin_y = int((s_y - y_lim[0]) / bin_width)
        
        mean, count, M2 = accumulators[s_cat]
        count[bin_y, bin_x] += 1
        delta = s_value - mean[bin_y, bin_x]
        mean[bin_y, bin_x] += delta / count[bin_y, bin_x]
        delta2 = s_value - mean[bin_y, bin_x]
        M2[bin_y, bin_x] += delta * delta2
    
    if verbose:
        print("Observed min/max coordinates: X ({}), Y ({})".format(true_x_lim, true_y_lim))
    
    def raw_to_metric(mean, count, M2, metric):
        if metric == "mean":
            return mean
        elif metric == "sum":
            return mean * count
        elif metric == "count":
            return count
        elif metric == "var":
            return M2 / count
        elif metric == "sample_var":
            return M2 / (count - 1)
        elif metric == "std":
            return np.sqrt(M2 / count)
        raise ValueError("Unknown metric.")
    
    n_categories = len(accumulators.keys())
    grid = (n_categories, 1) if orient == "v" else (1, n_categories)
    fig, axes = plt.subplots(*grid, figsize=figsize)
    cmap = matplotlib.cm.get_cmap(cmap)
    
    # Split the iteration over the data from the plotting so we can get common limits.
    colorbar_limits = +np.inf, -np.inf
    results_to_plot = []

    for idx, (category, (mean, count, M2)) in enumerate(sorted(accumulators.items())):
        ax = axes[idx]
        if category is not None:
            ax.set_title(category)
        
        result = raw_to_metric(mean, count, M2, metric)
        result_alpha = None
        if alpha is not None and alpha != "none":
            result_alpha = raw_to_metric(mean, count, M2, alpha).copy().astype(np.float32)
            result_alpha[np.isnan(result_alpha)] = 0.0
            result_alpha -= result_alpha.min()
            result_alpha /= result_alpha.max()
            if alpha in ("var", "sample_var", "std"):
                result_alpha = 1.0 - result_alpha
        vmin, vmax = np.percentile(result.flatten(), (1, 99))
        colorbar_limits = min(colorbar_limits[0], vmin), max(colorbar_limits[1], vmax)
        results_to_plot.append((ax, result, result_alpha))

    for (ax, result, result_alpha) in results_to_plot:
        # Common limits.
        vmin, vmax = colorbar_limits
        normalizer = matplotlib.colors.Normalize(vmin, vmax, clip=False)
        colormapper = matplotlib.cm.ScalarMappable(norm=normalizer, cmap=cmap)
        colormapper.set_array([])
        colormapper.set_clim(vmin, vmax)
        result = cmap(normalizer(result))
        
        if result_alpha is not None:
            result[:, :, -1] = result_alpha
        ax.imshow(result, interpolation=interpolation)
        plt.colorbar(colormapper, ax=ax)
        ax.set_axis_off()
        
        ax.set_aspect("equal")
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()