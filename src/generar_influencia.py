"""Genera las tablas de estimación y de influencia que usan los documentos.

Estimador principal: HC3. Clásico y HC1 son sensibilidad. Noruega se mantiene en el análisis principal;
la exclusión de cada país (incluida Noruega) se presenta solo como sensibilidad.

Salidas (src/outputs/tablas/):
- comparacion_estimadores.csv   coef, EE, t, p, R², IC95 de cada especificación con los tres estimadores
- influencia_resumen.csv        p-valor con todas las observaciones, sin Noruega y rango al excluir cada país (HC3, clásico, HC1)
- influencia_observaciones.csv  países cuya exclusión cambia el signo o la significación al 5 %
- correlaciones.csv             Pearson y Spearman de H1, H2 y H3
- diagnosticos.csv              Breusch-Pagan, apalancamiento máximo y distancia de Cook por especificación
Ejecutar después de generar_sensibilidad.py.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings("ignore")
RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "utils"))
import analisis as a  # noqa: E402
import sensibilidad as s  # noqa: E402

T = RAIZ / "outputs" / "tablas"
T.mkdir(parents=True, exist_ok=True)


def ajustar(formula, datos, cov):
    return smf.ols(formula, datos).fit() if cov == "clasico" else smf.ols(formula, datos).fit(cov_type=cov)


def especificaciones():
    p = s.construir_panel()
    ctx = s.cargar_contexto()
    fua = s.cargar_fua_2025()
    d1, _, _ = a.h1_emisiones_transporte(p)
    d2, _ = s.h2_con_controles(p, ctx)
    q, *_ = s.h3_calidad_aire(p, fua)
    c = ctx[ctx.anio == 2024].set_index("iso3")[["urbanizacion_pct", "densidad_hab_km2"]]
    d3 = q.join(c, how="inner").dropna(subset=["cuota_parque_pct", "no2_2025", "log_pib", "urbanizacion_pct", "densidad_hab_km2"]).copy()
    d3["log_densidad"] = np.log(d3["densidad_hab_km2"])
    return [
        ("H1", "sin controles", "cambio_co2_pc_pct ~ cuota_2023", d1, "cuota_2023", "ninguno"),
        ("H1", "con renta", "cambio_co2_pc_pct ~ cuota_2023 + log_pib_2023", d1, "cuota_2023", "log PIB 2023"),
        ("H2", "sin controles", "cuota_parque_pct ~ puntos_por_1000_hab", d2, "puntos_por_1000_hab", "ninguno"),
        ("H2", "con renta", "cuota_parque_pct ~ puntos_por_1000_hab + log_pib", d2, "puntos_por_1000_hab", "log PIB"),
        ("H2", "con los tres controles", "cuota_parque_pct ~ puntos_por_1000_hab + log_pib + urbanizacion_pct + log_densidad", d2, "puntos_por_1000_hab", "log PIB, urbanización, log densidad"),
        ("H3", "simple", "no2_2025 ~ cuota_parque_pct", d3, "cuota_parque_pct", "ninguno"),
        ("H3", "con renta", "no2_2025 ~ cuota_parque_pct + log_pib", d3, "cuota_parque_pct", "log PIB"),
        ("H3", "con urbanización y densidad", "no2_2025 ~ cuota_parque_pct + urbanizacion_pct + log_densidad", d3, "cuota_parque_pct", "urbanización, log densidad"),
        ("H3", "con los tres controles", "no2_2025 ~ cuota_parque_pct + log_pib + urbanizacion_pct + log_densidad", d3, "cuota_parque_pct", "log PIB, urbanización, log densidad"),
    ], d1, d2, d3


def main():
    SP, d1, d2, d3 = especificaciones()
    ESTIM = ["HC3", "clasico", "HC1"]
    comp, loo, diag = [], [], []
    for hip, mod, f, d, var, ctrl in SP:
        for est in ESTIM:
            m = ajustar(f, d, est)
            ci = m.conf_int().loc[var]
            comp.append({"hipotesis": hip, "modelo": mod, "n": int(m.nobs), "variable": var, "controles": ctrl,
                         "estimador": est, "coef": m.params[var], "error_estandar": m.bse[var], "t": m.tvalues[var],
                         "p": m.pvalues[var], "ic95_lo": ci[0], "ic95_hi": ci[1], "R2": m.rsquared})
            for excl in d.index:
                dd = d.drop(excl)
                mm = ajustar(f, dd, est)
                loo.append({"hipotesis": hip, "modelo": mod, "estimador": est, "excluido": excl, "n": int(mm.nobs),
                            "coef": mm.params[var], "p": mm.pvalues[var]})
        base = ajustar(f, d, "clasico")
        lv = base.get_influence().hat_matrix_diag
        cook = base.get_influence().cooks_distance[0]
        lm, lmp, _, _ = het_breuschpagan(base.resid, base.model.exog)
        k = len(base.params)
        diag.append({"hipotesis": hip, "modelo": mod, "n": int(base.nobs), "BP_LM": lm, "BP_p": lmp,
                     "h_max": lv.max(), "pais_h_max": d.index[lv.argmax()], "umbral_h_2k_n": 2 * k / len(d),
                     "n_h_sobre_umbral": int((lv > 2 * k / len(d)).sum()),
                     "cook_max": cook.max(), "pais_cook_max": d.index[cook.argmax()], "n_cook_sobre_4_n": int((cook > 4 / len(d)).sum())})
    C = pd.DataFrame(comp); C.to_csv(T / "comparacion_estimadores.csv", index=False)
    L = pd.DataFrame(loo); L.to_csv(T / "influencia_loo.csv", index=False)
    D = pd.DataFrame(diag); D.to_csv(T / "diagnosticos.csv", index=False)

    # resumen de influencia por especificación y estimador
    res, alertas = [], []
    for (hip, mod, est), g in C.groupby(["hipotesis", "modelo", "estimador"], sort=False):
        b = g.iloc[0]
        Lg = L[(L.hipotesis == hip) & (L.modelo == mod) & (L.estimador == est)]
        nor = Lg[Lg.excluido == "NOR"].iloc[0]
        sig0 = b.p < 0.05
        cambios = Lg[((Lg.p < 0.05) != sig0) | (np.sign(Lg.coef) != np.sign(b.coef))]
        res.append({"hipotesis": hip, "modelo": mod, "estimador": est, "n": b.n, "coef": b.coef, "p": b.p,
                    "p_sin_NOR": nor.p, "coef_sin_NOR": nor.coef, "p_min_excluyendo_un_pais": Lg.p.min(),
                    "p_max_excluyendo_un_pais": Lg.p.max(), "n_exclusiones_que_cambian_significacion_5": int(((Lg.p < 0.05) != sig0).sum()),
                    "n_exclusiones_que_cambian_signo": int((np.sign(Lg.coef) != np.sign(b.coef)).sum())})
        for _, r in cambios.iterrows():
            alertas.append({"hipotesis": hip, "modelo": mod, "estimador": est, "pais_excluido": r.excluido,
                            "p_con_todos": b.p, "p_sin_pais": r.p, "coef_con_todos": b.coef, "coef_sin_pais": r.coef,
                            "cambia_significacion_5": (r.p < 0.05) != sig0, "cambia_signo": np.sign(r.coef) != np.sign(b.coef)})
    pd.DataFrame(res).to_csv(T / "influencia_resumen.csv", index=False)
    pd.DataFrame(alertas).to_csv(T / "influencia_observaciones.csv", index=False)

    # correlaciones (descriptivas; no sustituyen a la pendiente HC3)
    cor = []
    for hip, x, y, d in [("H1", "cuota_2023", "cambio_co2_pc_pct", d1), ("H2", "puntos_por_1000_hab", "cuota_parque_pct", d2),
                         ("H3", "cuota_parque_pct", "no2_2025", d3)]:
        r, pr = stats.pearsonr(d[x], d[y]); rho, ps = stats.spearmanr(d[x], d[y]); n = len(d)
        cor.append({"hipotesis": hip, "n": n, "pearson_r": r, "pearson_t": r * np.sqrt(n - 2) / np.sqrt(1 - r * r),
                    "pearson_p": pr, "spearman_rho": rho, "spearman_p": ps})
    pd.DataFrame(cor).to_csv(T / "correlaciones.csv", index=False)
    print("tablas de estimación e influencia escritas en", T)


if __name__ == "__main__":
    main()
