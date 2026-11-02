"""Estilo común de gráficos (matplotlib)."""
import matplotlib.pyplot as plt

from src.config import FIGURES

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#a3a29c"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": GRAY,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "lines.linewidth": 1.2,
        "axes.prop_cycle": plt.cycler(color=[BLUE, ORANGE, AQUA]),
    })


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
