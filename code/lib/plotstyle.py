"""Common matplotlib style: print-friendly, colour-blind safe, vector PDF output."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f5fa8"
ORANGE = "#d9741a"
GREY = "#8a8a8a"
DARK = "#222222"
GREEN = "#2a8a5b"
RED = "#b3302f"


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save(fig, path):
    fig.savefig(path, format="pdf")
    plt.close(fig)
