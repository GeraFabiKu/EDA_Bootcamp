"""Paleta, estilos y utilidades de dibujo compartidas por todo el proyecto.

Criterio de color (para que el público no se confunda):
- verde-azulado  -> vehículo eléctrico / avance
- coral          -> emisiones y contaminación
- ámbar          -> destacados
- gris           -> referencia o dato sin información
Cada país mantiene el mismo color en todas las diapositivas en las que aparece.
"""
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

RAIZ = Path(__file__).resolve().parents[1]
RUTA_GEO = RAIZ / "data" / "geo" / "paises.geojson"

COLORES = {
    "electrico": "#00A896",
    "emision": "#F25F5C",
    "destacado": "#FFBA08",
    "referencia": "#8D99AE",
    "azul": "#3A86FF",
    "morado": "#7B2CBF",
    "texto": "#1F2937",
    "fondo": "#F8FAFC",
    "mapa_vacio": "#E5E7EB",
}

# Un color fijo por país para los gráficos de evolución
COLOR_PAIS = {
    "Norway": "#00A896", "Sweden": "#3A86FF", "Germany": "#7B2CBF",
    "China": "#F25F5C", "USA": "#FFBA08", "Spain": "#FB5607",
    "Italy": "#38B000", "Korea": "#8D99AE",
}


def estilo_presentacion():
    """Tema de seaborn para las diapositivas (fuentes grandes, fondo claro)."""
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    plt.rcParams.update({
        "figure.facecolor": COLORES["fondo"],
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.labelcolor": COLORES["texto"],
        "xtick.color": COLORES["texto"],
        "ytick.color": COLORES["texto"],
        "savefig.facecolor": COLORES["fondo"],
    })


def estilo_analisis():
    """Estilo sobrio de matplotlib para el notebook de análisis."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })


def _poligonos(geometria):
    if geometria["type"] == "Polygon":
        return [geometria["coordinates"]]
    if geometria["type"] == "MultiPolygon":
        return geometria["coordinates"]
    return []


def mapa_coropletico(ax, valores, cmap="BuGn", vmin=None, vmax=None):
    """Dibuja un mapa mundial coloreado por país con matplotlib (usa el GeoJSON local).

    `valores` es una Serie indexada por código ISO3. Devuelve el ScalarMappable
    para poder añadir la barra de color.
    """
    with open(RUTA_GEO, encoding="utf-8") as f:
        geo = json.load(f)
    valores = valores.dropna()
    vmin = valores.min() if vmin is None else vmin
    vmax = valores.max() if vmax is None else vmax
    norma = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmapa = plt.get_cmap(cmap)

    parches, colores = [], []
    for feat in geo["features"]:
        iso = feat["properties"].get("ISO3166-1-Alpha-3")
        if feat["geometry"] is None:
            continue
        color = cmapa(norma(valores[iso])) if iso in valores.index else COLORES["mapa_vacio"]
        for poligono in _poligonos(feat["geometry"]):
            parches.append(MplPolygon(np.array(poligono[0]), closed=True))
            colores.append(color)

    ax.add_collection(PatchCollection(parches, facecolors=colores, edgecolors="white", linewidths=0.3))
    ax.set_xlim(-170, 190)
    ax.set_ylim(-58, 84)
    ax.set_aspect("auto")
    ax.axis("off")
    return plt.cm.ScalarMappable(norm=norma, cmap=cmapa)
