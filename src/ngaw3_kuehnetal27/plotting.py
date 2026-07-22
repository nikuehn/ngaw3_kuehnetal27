"""
"""

from plotnine import *
import numpy as np

# ggplot/plotnine theme
size_title = 20
size_st = 15

fig_width = 8
fig_ratio = 0.8

custom_theme = (
    theme_bw()
    + theme(
        axis_title=element_text(size=size_title),
        axis_text=element_text(size=size_st),
        plot_title=element_text(size=size_title),
        plot_subtitle=element_text(size=size_st),
        legend_text=element_text(size=size_st),
        legend_title=element_text(size=size_st),
        legend_key_width=20,
        legend_box_background=element_rect(color="black"),
        panel_grid_major=element_line(color="gray", linewidth=0.75),
        panel_grid_minor=element_line(color="gray", linewidth=0.75),  # add this if you want minor gridlines
        legend_key_spacing_y=0,
        strip_text=element_text(size=size_st, face='bold')
    )
)

def log_breaks(limits):
    """Generate major breaks at powers of 10"""
    min_log = np.floor(np.log10(limits[0]))
    max_log = np.ceil(np.log10(limits[1]))
    breaks = 10 ** np.arange(min_log, max_log + 1)
    # Filter to only include breaks within the data range
    breaks = breaks[(breaks >= limits[0]) & (breaks <= limits[1])]
    return breaks

def log_minor_breaks(limits):
    """Generate minor breaks at 2,3,4,5,6,7,8,9 times powers of 10"""
    min_log = np.floor(np.log10(limits[0]))
    max_log = np.ceil(np.log10(limits[1]))

    minor = []
    for exp in range(int(min_log), int(max_log) + 1):
        base = 10 ** exp
        minor.extend([base * i for i in range(2, 10)])

    # Filter to only include minor breaks within the data range
    minor = [m for m in minor if limits[0] <= m <= limits[1]]
    return minor
