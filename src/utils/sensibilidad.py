"""Comprobaciones de robustez y factores de contexto del EDA.

Amplían el análisis principal (analisis.py) con:
- H1 con otras ventanas temporales (2000–2023) y con variables de contexto;
- H2 y H3 con controles de urbanización, densidad y renta;
- el precio medio de los coches eléctricos frente a los de combustión (IEA);
- una tabla de factores de contexto por país.

Las variables de contexto se miden en el año base de cada análisis (2015 en H1,
2024 en H2 y H3) para no mezclar la causa con el efecto.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from datos import RUTA_PROCESADOS, IEA_A_ISO3, RUTA_DATOS, cargar_afir, cargar_fua_2025
import analisis
from analisis import ANIO_BASE, ANIO_FIN, ANIO_CORTE, construir_panel, h2_ue, h3_calidad_aire

CONTEXTO = {
    "SP.URB.TOTL.IN.ZS": "urbanizacion_pct",
    "EN.POP.DNST": "densidad_hab_km2",
    "EG.ELC.RNEW.ZS": "renovables_elec_pct",
    "EG.USE.COMM.FO.ZS": "fosiles_energia_pct",
    "AG.LND.TOTL.K2": "superficie_km2",
    "SP.POP.TOTL": "poblacion",
    "NY.GDP.PCAP.KD": "pib_per_capita",
    "EN.GHG.CO2.TR.MT.CE.AR5": "co2_transporte_mt",
}


def cargar_contexto():
    """Indicadores de contexto 2000–2025 (Banco Mundial) en formato ancho, solo los 61 países del estudio."""
    l = pd.read_csv(RUTA_PROCESADOS / "wdi_extendido_2000_2025.csv")
    l = l[l["Indicator Code"].isin(CONTEXTO) & l["Country Code"].isin(IEA_A_ISO3.values())]
    l["indicador"] = l["Indicator Code"].map(CONTEXTO)
    ancho = l.pivot_table(index=["Country Code", "anio"], columns="indicador", values="valor", aggfunc="first")
    ancho = ancho.reset_index().rename(columns={"Country Code": "iso3"})
    ancho.columns.name = None
    return ancho


# Inferencia principal del estudio: errores HC3 (ver AUDITORIA_FINAL_EDA.md). Clásicos y HC1 solo como sensibilidad.
def _ols(formula, datos, cov="HC3"):
    return smf.ols(formula, datos).fit() if cov == "clasico" else smf.ols(formula, datos).fit(cov_type=cov)


def h1_ventana(p, ctx, anio_base=ANIO_BASE, anio_fin=ANIO_FIN, controles=False):
    """H1 con un año base y final arbitrarios. Con controles=True añade densidad, urbanización y renovables del año base."""
    c = ctx.set_index(["iso3", "anio"])
    base = c.xs(anio_base, level="anio")
    fin = c.xs(anio_fin, level="anio")
    cuota = p[p.anio == anio_fin].set_index("iso3")["cuota_parque_pct"]
    d = pd.DataFrame({
        "cuota": cuota,
        "co2_pc_base": base["co2_transporte_mt"] * 1e6 / base["poblacion"],
        "co2_pc_fin": fin["co2_transporte_mt"] * 1e6 / fin["poblacion"],
        "log_pib": np.log(fin["pib_per_capita"]),
        "log_densidad_base": np.log(base["densidad_hab_km2"]),
        "urbanizacion_base": base["urbanizacion_pct"],
        "renovables_base": base["renovables_elec_pct"],
    }).dropna()
    d["cambio"] = (d["co2_pc_fin"] / d["co2_pc_base"] - 1) * 100
    m1 = _ols("cambio ~ cuota", d)
    m2 = _ols("cambio ~ cuota + log_pib", d)
    out = {"n": len(d), "b_sin_renta": m1.params["cuota"], "p_sin_renta": m1.pvalues["cuota"],
           "b_renta": m2.params["cuota"], "p_renta": m2.pvalues["cuota"],
           "b_pib": m2.params["log_pib"], "p_pib": m2.pvalues["log_pib"], "r2_renta": m2.rsquared}
    if controles:
        m3 = _ols("cambio ~ cuota + log_pib + log_densidad_base + urbanizacion_base + renovables_base", d)
        out.update({"b_contexto": m3.params["cuota"], "p_contexto": m3.pvalues["cuota"],
                    "p_densidad": m3.pvalues["log_densidad_base"], "p_urbanizacion": m3.pvalues["urbanizacion_base"],
                    "p_renovables": m3.pvalues["renovables_base"], "r2_contexto": m3.rsquared})
    return d, out


def h2_con_controles(p, ctx, anio=ANIO_CORTE):
    """H2: ¿la recarga explica la cuota de VE una vez controlada la renta, la urbanización y la densidad?"""
    q = p[p.anio == anio].set_index("iso3")[["pais", "cuota_parque_pct", "puntos_por_1000_hab", "log_pib"]]
    c = ctx[ctx.anio == anio].set_index("iso3")[["urbanizacion_pct", "densidad_hab_km2"]]
    d = q.join(c, how="inner").dropna()
    d["log_densidad"] = np.log(d["densidad_hab_km2"])
    m = _ols("cuota_parque_pct ~ puntos_por_1000_hab + log_pib + urbanizacion_pct + log_densidad", d)
    return d, m


def h3_con_controles(p, fua, ctx, anio=ANIO_CORTE):
    """H3: ¿el parque eléctrico se asocia con el NO2 una vez controlados la urbanización y la densidad?"""
    q, r, _, _ = h3_calidad_aire(p, fua, anio)
    c = ctx[ctx.anio == anio].set_index("iso3")[["urbanizacion_pct", "densidad_hab_km2"]]
    d = q.join(c, how="inner").dropna(subset=["urbanizacion_pct", "densidad_hab_km2"])
    d["log_densidad"] = np.log(d["densidad_hab_km2"])
    m = _ols("no2_2025 ~ cuota_parque_pct + urbanizacion_pct + log_densidad", d)
    return d, m


def precios_prima(anio_max=2025):
    """Prima de precio BEV frente a ICEV (precio medio de catálogo en USD 2025, IEA).

    Se usa el segmento 'Medium car' y, por país, el año más reciente con datos de ambos.
    El precio es de catálogo: incluye IVA e impuestos de compra, pero excluye subvenciones.
    """
    ev = pd.read_excel(RUTA_DATOS / "EV data by country 2026.xlsx", sheet_name="GEVO_EV_2026")
    s = ev[(ev["parameter"].eq("price_mean_2025USD")) & (ev["mode"].eq("Medium car"))
           & (ev["powertrain"].isin(["BEV", "ICEV"])) & (ev["year"] <= anio_max)].copy()
    s["iso3"] = s["region_country"].map(IEA_A_ISO3)
    s = s.dropna(subset=["iso3"])
    piv = s.pivot_table(index=["iso3", "year"], columns="powertrain", values="value", aggfunc="first").dropna()
    piv = piv.reset_index().sort_values("year").groupby("iso3").tail(1).set_index("iso3")
    piv["prima_bev_pct"] = (piv["BEV"] / piv["ICEV"] - 1) * 100
    piv = piv.rename(columns={"year": "anio_precio", "BEV": "precio_bev_usd", "ICEV": "precio_icev_usd"})
    return piv[["anio_precio", "precio_bev_usd", "precio_icev_usd", "prima_bev_pct"]]


def tabla_contexto(p, ctx, afir, fua):
    """Tabla de factores de contexto por país (año de comparación ANIO_CORTE)."""
    q = p[p.anio == ANIO_CORTE].set_index("iso3")[["pais", "cuota_parque_pct", "puntos_por_1000_hab",
                                                    "pib_per_capita", "grupo_renta"]]
    c = ctx[ctx.anio == ANIO_CORTE].set_index("iso3")[["urbanizacion_pct", "densidad_hab_km2",
                                                        "fosiles_energia_pct", "superficie_km2"]]
    r = ctx[ctx.anio == ANIO_BASE].set_index("iso3")[["renovables_elec_pct"]].rename(
        columns={"renovables_elec_pct": "renovables_elec_2015_pct"})
    precio = precios_prima()
    t = q.join(c, how="left").join(r, how="left").join(precio, how="left")
    t = t.join(fua.set_index("iso3")[["no2_2025"]], how="left")
    return t


def h3_completo(p, fua, ctx, anio=ANIO_CORTE):
    """H3 con los tres controles a la vez: renta, urbanización y densidad."""
    q, r, _, _ = h3_calidad_aire(p, fua, anio)
    c = ctx[ctx.anio == anio].set_index("iso3")[["urbanizacion_pct", "densidad_hab_km2"]]
    d = q.join(c, how="inner").dropna(subset=["urbanizacion_pct", "densidad_hab_km2", "log_pib"])
    d["log_densidad"] = np.log(d["densidad_hab_km2"])
    return d, _ols("no2_2025 ~ cuota_parque_pct + log_pib + urbanizacion_pct + log_densidad", d)


def tabla_estimadores(p, fua, ctx, anio=ANIO_CORTE):
    """Sensibilidad a los errores estándar: clásicos, HC1 (usado en H1 y H2) y HC3, con las mismas observaciones."""
    d, _ = h3_completo(p, fua, ctx, anio)
    dh, _ = h2_con_controles(p, ctx, anio)
    h1d = analisis.h1_emisiones_transporte(p)[0]
    modelos = [
        ("H1 sin controles", "cambio_co2_pc_pct ~ cuota_2023", h1d),
        ("H1 con renta", "cambio_co2_pc_pct ~ cuota_2023 + log_pib_2023", h1d),
        ("H3 sin controles", "no2_2025 ~ cuota_parque_pct", d),
        ("H3 con renta", "no2_2025 ~ cuota_parque_pct + log_pib", d),
        ("H3 con urbanización y densidad", "no2_2025 ~ cuota_parque_pct + urbanizacion_pct + log_densidad", d),
        ("H3 con los tres controles", "no2_2025 ~ cuota_parque_pct + log_pib + urbanizacion_pct + log_densidad", d),
        ("H2 sin controles", "cuota_parque_pct ~ puntos_por_1000_hab", dh),
        ("H2 con renta", "cuota_parque_pct ~ puntos_por_1000_hab + log_pib", dh),
        ("H2 con los tres controles", "cuota_parque_pct ~ puntos_por_1000_hab + log_pib + urbanizacion_pct + log_densidad", dh),
    ]
    filas = []
    for nombre, f, datos_m in modelos:
        var = f.split("~")[1].strip().split(" ")[0]
        for cov in ["clasico", "HC1", "HC3"]:
            m = smf.ols(f, datos_m).fit() if cov == "clasico" else smf.ols(f, datos_m).fit(cov_type=cov)
            filas.append({"modelo": nombre, "n": int(m.nobs), "variable": var, "errores": cov,
                          "coef": m.params[var], "EE": m.bse[var], "p": m.pvalues[var]})
    return pd.DataFrame(filas)


def descriptivos(p, fua):
    """Cifras descriptivas usadas en los documentos (muestras, valores ausentes, extremos)."""
    d = {}
    d["filas_panel"] = int(len(p))
    d["paises_panel"] = int(p["iso3"].nunique())
    d["anios_panel"] = f"{int(p['anio'].min())}–{int(p['anio'].max())}"
    claves = {"cuota_parque_pct": "cuota", "puntos_recarga_publicos": "recarga", "co2_transporte_mt": "co2",
              "pib_per_capita": "pib", "poblacion": "poblacion"}
    d["ausentes"] = {v: int(p[k].isna().sum()) for k, v in claves.items()}
    d["ausentes_afir_2024"] = int(p.loc[p.anio == ANIO_CORTE, "cuota_parque_pct"].isna().sum())
    q = p[p.anio == ANIO_CORTE].set_index("iso3")
    d["cuota_2024"] = {k: float(q.loc[k, "cuota_parque_pct"]) for k in ["NOR", "ISL", "SWE", "DNK", "CHN", "USA", "MYS", "IDN", "SGP", "ISR"]
                       if k in q.index}
    top = q["cuota_parque_pct"].dropna().sort_values(ascending=False).head(3)
    d["top3_cuota_2024"] = [[str(q.loc[i, "pais"]), float(v)] for i, v in top.items()]
    f = fua.set_index("iso3")
    f = f[f.index.isin(p["iso3"].unique())]  # solo países del panel (Argentina figura en el archivo y no en el estudio)
    d["no2_paises"] = int(f["no2_2025"].notna().sum())
    d["no2_media"] = float(f["no2_2025"].mean())
    d["no2_min"] = float(f["no2_2025"].min())
    d["no2_max"] = float(f["no2_2025"].max())
    d["no2_ciudades_total"] = int(f["ciudades_no2"].sum())
    q3, _, _, _ = h3_calidad_aire(p, fua)
    d["h3_n"] = int(len(q3))
    d["no2_ciudades_h3"] = int(q3["ciudades_no2"].sum())
    res_f = RUTA_PROCESADOS / "fua_no2_resumen.json"
    if res_f.exists():
        d.update({"no2_archivo_" + k: v for k, v in json.loads(res_f.read_text(encoding="utf-8")).items()})
    d["no2_ciudades_min"] = [str(f["ciudades_no2"].idxmin()), int(f["ciudades_no2"].min())]
    d["no2_ciudades_max"] = [str(f["ciudades_no2"].idxmax()), int(f["ciudades_no2"].max())]
    for k in ["MYS", "IDN", "SGP", "ISR", "CHN"]:
        if k in f.index:
            d[f"no2_{k}"] = float(f.loc[k, "no2_2025"])
            d[f"ciudades_no2_{k}"] = int(f.loc[k, "ciudades_no2"])
    return d


def ejecutar_todo():
    """Calcula todas las comprobaciones y devuelve un diccionario de tablas y cifras."""
    p = construir_panel()
    ctx = cargar_contexto()
    fua = cargar_fua_2025()
    afir = cargar_afir()
    res = {}
    res["h1_2015"], res["h1_2015_stats"] = h1_ventana(p, ctx, 2015, 2023, controles=True)
    res["h1_2000"], res["h1_2000_stats"] = h1_ventana(p, ctx, 2000, 2023, controles=True)
    res["h1_prepandemia"], res["h1_prepandemia_stats"] = h1_ventana(p, ctx, 2015, 2019, controles=False)
    res["h2_ctrl_d"], res["h2_ctrl_m"] = h2_con_controles(p, ctx)
    res["h3_ctrl_d"], res["h3_ctrl_m"] = h3_con_controles(p, fua, ctx)
    res["precio"] = precios_prima()
    res["contexto"] = tabla_contexto(p, ctx, afir, fua)
    res["h3_completo_d"], res["h3_completo_m"] = h3_completo(p, fua, ctx)
    res["descriptivos"] = descriptivos(p, fua)
    res["estimadores"] = tabla_estimadores(p, fua, ctx)
    return res
