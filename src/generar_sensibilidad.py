"""Genera las comprobaciones de robustez y los factores de contexto.

Salidas (en src/outputs/):
- tablas/sensibilidad_h1.csv, sensibilidad_h2_h3.csv, precio_prima_bev.csv, contexto_paises_2024.csv
- cifras_sensibilidad.json  (todas las cifras de sensibilidad usadas en los documentos)

Ejecutar antes de generar_influencia.py y generar_contrastes_ajustados.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent          # carpeta src/
sys.path.insert(0, str(RAIZ / "utils"))
import sensibilidad as s  # noqa: E402

SALIDA = RAIZ / "outputs"
(SALIDA / "tablas").mkdir(parents=True, exist_ok=True)


def a_float(x):
    return float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None


def main():
    R = s.ejecutar_todo()

    h1 = {"2015_2023": R["h1_2015_stats"], "2000_2023": R["h1_2000_stats"], "2015_2019": R["h1_prepandemia_stats"]}
    filas = []
    for clave, st in h1.items():
        filas.append({"especificacion": f"H1 {clave.replace('_', '–')}", **{k: a_float(v) for k, v in st.items()}})
    import pandas as pd
    pd.DataFrame(filas).to_csv(SALIDA / "tablas" / "sensibilidad_h1.csv", index=False)

    m2, m3 = R["h2_ctrl_m"], R["h3_ctrl_m"]
    h23 = pd.DataFrame([
        {"hipotesis": "H2 con controles", "n": int(m2.nobs), "coef": m2.params["puntos_por_1000_hab"],
         "p": m2.pvalues["puntos_por_1000_hab"], "r2": m2.rsquared},
        {"hipotesis": "H3 con controles", "n": int(m3.nobs), "coef": m3.params["cuota_parque_pct"],
         "p": m3.pvalues["cuota_parque_pct"], "r2": m3.rsquared},
    ])
    h23.to_csv(SALIDA / "tablas" / "sensibilidad_h2_h3.csv", index=False)
    R["estimadores"].round(4).to_csv(SALIDA / "tablas" / "sensibilidad_estimadores_h2_h3.csv", index=False)
    import analisis, datos
    q3, _, _, _ = analisis.h3_calidad_aire(analisis.construir_panel(), datos.cargar_fua_2025())
    q3.reset_index().to_csv(SALIDA / "tablas" / "h3_no2_adopcion.csv", index=False)

    R["precio"].round(2).to_csv(SALIDA / "tablas" / "precio_prima_bev.csv")
    R["contexto"].round(3).to_csv(SALIDA / "tablas" / "contexto_paises_2024.csv")

    m4 = R["h3_completo_m"]
    cifras = {
        "h3_completo": {"n": int(m4.nobs), "coef_cuota": a_float(m4.params["cuota_parque_pct"]),
                         "p_cuota": a_float(m4.pvalues["cuota_parque_pct"]),
                         "p_renta": a_float(m4.pvalues["log_pib"]),
                         "p_urbanizacion": a_float(m4.pvalues["urbanizacion_pct"]),
                         "p_densidad": a_float(m4.pvalues["log_densidad"])},
        "descriptivos": R["descriptivos"],
        "h1_2015": {k: a_float(v) for k, v in R["h1_2015_stats"].items()},
        "h1_2000": {k: a_float(v) for k, v in R["h1_2000_stats"].items()},
        "h1_prepandemia": {k: a_float(v) for k, v in R["h1_prepandemia_stats"].items()},
        "h2_ctrl": {"n": int(m2.nobs), "coef_puntos": a_float(m2.params["puntos_por_1000_hab"]),
                    "p_puntos": a_float(m2.pvalues["puntos_por_1000_hab"]),
                    "p_renta": a_float(m2.pvalues["log_pib"]),
                    "p_urbanizacion": a_float(m2.pvalues["urbanizacion_pct"]),
                    "p_densidad": a_float(m2.pvalues["log_densidad"]), "r2": a_float(m2.rsquared)},
        "h3_ctrl": {"n": int(m3.nobs), "coef_cuota": a_float(m3.params["cuota_parque_pct"]),
                    "p_cuota": a_float(m3.pvalues["cuota_parque_pct"]),
                    "p_urbanizacion": a_float(m3.pvalues["urbanizacion_pct"]),
                    "p_densidad": a_float(m3.pvalues["log_densidad"]), "r2": a_float(m3.rsquared)},
        "precio": {"n": int(len(R["precio"])), "media": a_float(R["precio"]["prima_bev_pct"].mean()),
                   "mediana": a_float(R["precio"]["prima_bev_pct"].median()),
                   "min": a_float(R["precio"]["prima_bev_pct"].min()),
                   "max": a_float(R["precio"]["prima_bev_pct"].max()),
                   "anios": [int(x) for x in sorted(R["precio"]["anio_precio"].unique())]},
    }
    (SALIDA / "cifras_sensibilidad.json").write_text(json.dumps(cifras, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Sensibilidad generada:", SALIDA / "cifras_sensibilidad.json")


if __name__ == "__main__":
    main()
