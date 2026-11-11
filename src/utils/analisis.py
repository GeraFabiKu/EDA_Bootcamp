"""Panel país-año y pruebas de las hipótesis del EDA.

Todas las cifras de la presentación, la memoria y el notebook salen de estas
funciones, así que los números coinciden en todos los entregables.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from datos import (RUTA_PROCESADOS, cargar_afir, cargar_fua_2025, cargar_gni,
                   cargar_iea, combinar_panel)

ANIO_BASE = 2015     # inicio del periodo para medir el cambio de emisiones (H1)
ANIO_FIN = 2023      # último año con emisiones de transporte completas (H1)
ANIO_CORTE = 2024    # año de comparación entre países (H2 y H3)


def construir_panel():
    """Une IEA, Banco Mundial, AFIR y renta en un panel país-año con variables derivadas."""
    iea = cargar_iea()
    wdi = pd.read_csv(RUTA_PROCESADOS / "wdi_indicadores.csv")
    afir = cargar_afir()
    gni = cargar_gni()
    p = combinar_panel(iea, wdi, afir, gni)
    p["co2_transporte_pc"] = p["co2_transporte_mt"] * 1e6 / p["poblacion"]   # t CO2 por habitante
    p["log_pib"] = np.log(p["pib_per_capita"])
    p["puntos_por_1000_hab"] = p["puntos_recarga_publicos"] / p["poblacion"] * 1000
    p["grupo_renta"] = pd.cut(
        p["pib_per_capita"], bins=[0, 15_000, 40_000, np.inf],
        labels=["Renta media-baja", "Renta alta", "Renta muy alta"])
    return p


def cargar_calidad_aire():
    return cargar_fua_2025()


def h1_emisiones_transporte(p):
    """H1: los países con más VE redujeron más sus emisiones de CO2 del transporte por habitante."""
    a = p[p.anio == ANIO_BASE].set_index("iso3")
    b = p[p.anio == ANIO_FIN].set_index("iso3")
    d = pd.DataFrame({
        "pais": b["pais"], "cuota_2023": b["cuota_parque_pct"],
        "co2_pc_2015": a["co2_transporte_pc"], "co2_pc_2023": b["co2_transporte_pc"],
        "log_pib_2023": b["log_pib"], "pib_2023": b["pib_per_capita"],
    }).dropna(subset=["cuota_2023", "co2_pc_2015", "co2_pc_2023", "log_pib_2023"])
    d["cambio_co2_pc_pct"] = (d["co2_pc_2023"] / d["co2_pc_2015"] - 1) * 100
    m_simple = smf.ols("cambio_co2_pc_pct ~ cuota_2023", d).fit(cov_type="HC3")
    m_control = smf.ols("cambio_co2_pc_pct ~ cuota_2023 + log_pib_2023", d).fit(cov_type="HC3")
    return d, m_simple, m_control


def h2_infraestructura(p, anio=ANIO_CORTE, n_boot=5000, semilla=42):
    """H2: los países con más puntos de recarga por habitante tienen más VE en su parque."""
    q = p[p.anio == anio].set_index("iso3")
    d = q[["pais", "cuota_parque_pct", "puntos_por_1000_hab", "log_pib"]].dropna()
    r = np.corrcoef(d["puntos_por_1000_hab"], d["cuota_parque_pct"])[0, 1]

    # Intervalo de confianza al 95 % por bootstrap (remuestreo con numpy)
    rng = np.random.default_rng(semilla)
    x, y = d["puntos_por_1000_hab"].to_numpy(), d["cuota_parque_pct"].to_numpy()
    indices = rng.integers(0, len(d), size=(n_boot, len(d)))
    r_boot = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in indices])
    ic = np.percentile(r_boot, [2.5, 97.5])

    r_renta = np.corrcoef(d["log_pib"], d["cuota_parque_pct"])[0, 1]
    m_pts = smf.ols("cuota_parque_pct ~ puntos_por_1000_hab", d).fit(cov_type="HC3")
    m_renta = smf.ols("cuota_parque_pct ~ log_pib", d).fit(cov_type="HC3")
    m_ambas = smf.ols("cuota_parque_pct ~ puntos_por_1000_hab + log_pib", d).fit(cov_type="HC3")
    return {"datos": d, "r": r, "ic": ic, "r_renta": r_renta,
            "m_pts": m_pts, "m_renta": m_renta, "m_ambas": m_ambas}


def h2_ue(p, afir, anio=ANIO_CORTE):
    """Contraste en la UE (fuente AFIR): ¿los países con más ventas de VE tienen más puntos de recarga por habitante?

    Se usan solo los países con datos completos en ambas fuentes, sin rellenar con ceros.
    """
    a = afir[afir["anio"] == anio].set_index("iso3")["afir_puntos_total"]
    q = p[p.anio == anio].set_index("iso3")[["pais", "cuota_ventas_pct", "poblacion"]]
    d = q.join(a.rename("puntos_afir"), how="inner").dropna()
    d["puntos_por_1000_hab"] = d["puntos_afir"] / d["poblacion"] * 1000
    r = np.corrcoef(d["cuota_ventas_pct"], d["puntos_por_1000_hab"])[0, 1]
    return d, r


def h3_calidad_aire(p, fua, anio=ANIO_CORTE):
    """H3: los países con más VE tienen menos NO2 en sus ciudades (2025).

    Muestra: países del panel con cuota del parque en 2024, NO2 medio de sus ciudades y renta.
    Errores estándar clásicos (OLS): son los que corresponden al contraste de la correlación de Pearson
    (misma observación, mismo p). Los errores robustos HC1 y HC3 se calculan en sensibilidad.py.
    """
    q = (p[p.anio == anio].set_index("iso3")[["pais", "cuota_parque_pct", "log_pib"]]
         .join(fua.set_index("iso3"), how="inner")
         .dropna(subset=["cuota_parque_pct", "no2_2025", "log_pib"]))
    r = np.corrcoef(q["cuota_parque_pct"], q["no2_2025"])[0, 1]
    m_simple = smf.ols("no2_2025 ~ cuota_parque_pct", q).fit(cov_type="HC3")
    m_control = smf.ols("no2_2025 ~ cuota_parque_pct + log_pib", q).fit(cov_type="HC3")
    return q, r, m_simple, m_control


def resumen_numeros(p, fua, afir):
    """Diccionario con las cifras clave, usado por la presentación y la memoria."""
    h1d, h1s, h1c = h1_emisiones_transporte(p)
    h2 = h2_infraestructura(p)
    ue, r_ue = h2_ue(p, afir)
    h3q, r3, h3s, h3c = h3_calidad_aire(p, fua)
    return {
        "paises_panel": p["iso3"].nunique(),
        "anios": f"{int(p['anio'].min())}–{int(p['anio'].max())}",
        "h1_n": len(h1d),
        "h1_b_simple": h1s.params["cuota_2023"], "h1_p_simple": h1s.pvalues["cuota_2023"], "h1_r2_simple": h1s.rsquared,
        "h1_b_control": h1c.params["cuota_2023"], "h1_p_control": h1c.pvalues["cuota_2023"],
        "h1_b_pib": h1c.params["log_pib_2023"], "h1_p_pib": h1c.pvalues["log_pib_2023"], "h1_r2_control": h1c.rsquared,
        "h2_n": len(h2["datos"]), "h2_r": h2["r"], "h2_ic_lo": h2["ic"][0], "h2_ic_hi": h2["ic"][1],
        "h2_r2_pts": h2["m_pts"].rsquared, "h2_r2_renta": h2["m_renta"].rsquared,
        "h2_p_pts_control": h2["m_ambas"].pvalues["puntos_por_1000_hab"],
        "h2_p_renta_control": h2["m_ambas"].pvalues["log_pib"],
        "h2_r_ue": r_ue, "h2_n_ue": len(ue),
        "h3_n": len(h3q), "h3_r": r3,
        "h3_p_r": stats.pearsonr(h3q["cuota_parque_pct"], h3q["no2_2025"])[1],
        "h3_ciudades": int(h3q["ciudades_no2"].sum()),
        "h3_p_simple_hc1": smf.ols("no2_2025 ~ cuota_parque_pct", h3q).fit(cov_type="HC1").pvalues["cuota_parque_pct"],
        "h3_p_simple_clasico": smf.ols("no2_2025 ~ cuota_parque_pct", h3q).fit().pvalues["cuota_parque_pct"],
        "h3_p_control_hc1": smf.ols("no2_2025 ~ cuota_parque_pct + log_pib", h3q).fit(cov_type="HC1").pvalues["cuota_parque_pct"],
        "h3_p_control_clasico": smf.ols("no2_2025 ~ cuota_parque_pct + log_pib", h3q).fit().pvalues["cuota_parque_pct"],
        "h3_b_simple": h3s.params["cuota_parque_pct"], "h3_p_simple": h3s.pvalues["cuota_parque_pct"],
        "h3_b_control": h3c.params["cuota_parque_pct"], "h3_p_control": h3c.pvalues["cuota_parque_pct"],
        "h3_p_pib": h3c.pvalues["log_pib"],
    }
