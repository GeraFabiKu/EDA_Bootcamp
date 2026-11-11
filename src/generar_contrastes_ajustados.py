"""Ajuste por comparaciones múltiples de los contrastes principales y de sensibilidad.

Principal (Holm, complementario): una prueba por hipótesis, pendiente simple con HC3.
Sensibilidad (BH y Holm): familia de especificaciones HC3 y ventanas temporales de H1.
Escribe outputs/tablas/contrastes_ajustados.csv y outputs/cifras_ajustes.json.
Ejecutar después de generar_influencia.py (que escribe comparacion_estimadores.csv).
"""
import json
from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent
OUT = RAIZ / "outputs"
TAB = OUT / "tablas"


def holm(p):
    items = sorted(p.items(), key=lambda kv: kv[1]); m = len(items); out = {}; prev = 0.0
    for i, (k, v) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * v)); prev = adj; out[k] = adj
    return out


def bh(p):
    items = sorted(p.items(), key=lambda kv: kv[1]); m = len(items); out = {}; prev = 1.0
    for i in range(m - 1, -1, -1):
        k, v = items[i]; adj = min(prev, v * m / (i + 1)); prev = adj; out[k] = adj
    return out


def main():
    comp = pd.read_csv(TAB / "comparacion_estimadores.csv")
    hc3 = comp[comp["estimador"] == "HC3"]
    sens = json.loads((OUT / "cifras_sensibilidad.json").read_text(encoding="utf-8"))

    principal = {
        "H1": float(hc3[(hc3.hipotesis == "H1") & (hc3.modelo == "sin controles")]["p"].iloc[0]),
        "H2": float(hc3[(hc3.hipotesis == "H2") & (hc3.modelo == "sin controles")]["p"].iloc[0]),
        "H3": float(hc3[(hc3.hipotesis == "H3") & (hc3.modelo == "simple")]["p"].iloc[0]),
    }
    ventanas = {
        "H1 2015–2019 sin controles": sens["h1_prepandemia"]["p_sin_renta"],
        "H1 2015–2019 con renta": sens["h1_prepandemia"]["p_renta"],
        "H1 2000–2023 sin controles": sens["h1_2000"]["p_sin_renta"],
        "H1 2000–2023 con renta": sens["h1_2000"]["p_renta"],
        "H1 2015–2023 con contexto": sens["h1_2015"]["p_contexto"],
    }
    especif = {f"{r.hipotesis} · {r.modelo} · {r.variable}": float(r.p) for r in hc3.itertuples()}

    familias = {
        "P_principal": principal,
        "E_especificaciones": especif,
        "V_ventanas_H1": ventanas,
        "E+V_total": {**especif, **{f"V: {k}": v for k, v in ventanas.items()}},
    }
    filas = []
    for fam, pv in familias.items():
        h, b = holm(pv), bh(pv)
        for k, v in pv.items():
            filas.append({"familia": fam, "contraste": k, "p_nominal_HC3": v,
                          "p_holm": h[k], "p_bh": b[k],
                          "significativo_nominal": v < 0.05,
                          "significativo_holm": h[k] < 0.05,
                          "significativo_bh": b[k] < 0.05})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(TAB / "contrastes_ajustados.csv", index=False, float_format="%.6f")

    cifras = {
        "principal_nominal": {k: round(v, 6) for k, v in principal.items()},
        "principal_holm": {k: round(v, 6) for k, v in holm(principal).items()},
        "principal_bh": {k: round(v, 6) for k, v in bh(principal).items()},
        "ventana_h1_2015_2019_holm_familia_V": round(holm(ventanas)["H1 2015–2019 sin controles"], 6),
        "ventana_h1_2015_2019_holm_total": round(holm({**especif, **{f"V: {k}": v for k, v in ventanas.items()}})["V: H1 2015–2019 sin controles"], 6),
        "m_principal": len(principal),
        "m_total": len(especif) + len(ventanas),
    }
    (OUT / "cifras_ajustes.json").write_text(json.dumps(cifras, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Contrastes ajustados escritos en", TAB / "contrastes_ajustados.csv")
    print(json.dumps(cifras, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
