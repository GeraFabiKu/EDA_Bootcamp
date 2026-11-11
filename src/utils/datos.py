"""Carga, limpieza y combinación de los datasets del EDA.

Cada función `cargar_*` devuelve un DataFrame con la clave `iso3` (código de
país de 3 letras) y, cuando aplica, `anio`. Así todas las fuentes se pueden
unir con `pd.merge` sin depender de cómo escribe cada organismo el nombre
de los países (p. ej. "Turkiye", "Türkiye" o "Turkey").
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

RUTA_DATOS = Path(__file__).resolve().parents[1] / "data"
RUTA_PROCESADOS = RUTA_DATOS / "procesados"

# IEA (Global EV Data Explorer) -> ISO3
IEA_A_ISO3 = {
    "Argentina": "ARG", "Australia": "AUS", "Austria": "AUT", "Belgium": "BEL",
    "Brazil": "BRA", "Bulgaria": "BGR", "Cambodia": "KHM", "Canada": "CAN",
    "Chile": "CHL", "China": "CHN", "Colombia": "COL", "Costa Rica": "CRI",
    "Croatia": "HRV", "Cyprus": "CYP", "Czech Republic": "CZE", "Denmark": "DNK",
    "Estonia": "EST", "Finland": "FIN", "France": "FRA", "Germany": "DEU",
    "Greece": "GRC", "Hungary": "HUN", "Iceland": "ISL", "India": "IND",
    "Indonesia": "IDN", "Ireland": "IRL", "Israel": "ISR", "Italy": "ITA",
    "Japan": "JPN", "Jordan": "JOR", "Korea": "KOR", "Lao PDR": "LAO",
    "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX", "Malaysia": "MYS",
    "Mexico": "MEX", "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL",
    "Norway": "NOR", "Philippines": "PHL", "Poland": "POL", "Portugal": "PRT",
    "Romania": "ROU", "Russia": "RUS", "Seychelles": "SYC", "Singapore": "SGP",
    "Slovakia": "SVK", "Slovenia": "SVN", "South Africa": "ZAF", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "CHE", "Thailand": "THA", "Turkiye": "TUR",
    "USA": "USA", "United Arab Emirates": "ARE", "United Kingdom": "GBR",
    "Uruguay": "URY", "Uzbekistan": "UZB", "Viet Nam": "VNM",
}

# AFIR (Comisión Europea) -> ISO3
AFIR_A_ISO3 = {
    "Austria": "AUT", "Belgium": "BEL", "Bulgaria": "BGR", "Croatia": "HRV",
    "Cyprus": "CYP", "Czech Republic": "CZE", "Denmark": "DNK", "Estonia": "EST",
    "Finland": "FIN", "France": "FRA", "Germany": "DEU", "Greece": "GRC",
    "Hungary": "HUN", "Ireland": "IRL", "Italy": "ITA", "Latvia": "LVA",
    "Lithuania": "LTU", "Luxembourg": "LUX", "Malta": "MLT", "Netherlands": "NLD",
    "Poland": "POL", "Portugal": "PRT", "Romania": "ROU", "Slovakia": "SVK",
    "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE",
}

# Calidad del aire por áreas urbanas funcionales (FUA) -> ISO3.
# Criterio: todo país del panel (IEA) con ciudades FUA en el archivo de NO2 debe figurar aquí.
# Seychelles (SYC) no aparece en el archivo FUA y queda sin NO2.
FUA_A_ISO3 = {
    "Austria": "AUT", "Belgium": "BEL", "Bulgaria": "BGR", "Croatia": "HRV",
    "Cyprus": "CYP", "CzechRepublic": "CZE", "Denmark": "DNK", "Estonia": "EST",
    "Finland": "FIN", "France": "FRA", "Germany": "DEU", "Greece": "GRC",
    "Hungary": "HUN", "Iceland": "ISL", "Ireland": "IRL", "Italy": "ITA",
    "Latvia": "LVA", "Lithuania": "LTU", "Luxembourg": "LUX", "Netherlands": "NLD",
    "Norway": "NOR", "Poland": "POL", "Portugal": "PRT", "Romania": "ROU",
    "Slovakia": "SVK", "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE",
    "Switzerland": "CHE", "UnitedKingdom": "GBR", "Russia": "RUS", "Turkey": "TUR",
    "Chile": "CHL", "Argentina": "ARG", "Australia": "AUS", "Brazil": "BRA",
    "Canada": "CAN", "China": "CHN", "Colombia": "COL", "CostaRica": "CRI",
    "India": "IND", "Indonesia": "IDN", "Japan": "JPN", "SouthKorea": "KOR",
    "Malaysia": "MYS", "Mexico": "MEX", "NewZealand": "NZL", "Philippines": "PHL",
    "SouthAfrica": "ZAF", "Thailand": "THA", "UnitedStates": "USA",
    "Uruguay": "URY", "Vietnam": "VNM", "Israel": "ISR", "Singapore": "SGP",
    # Añadidos en la corrección: estaban en la IEA y en el archivo FUA, pero faltaban aquí.
    "UnitedArabEmirates": "ARE", "Jordan": "JOR", "Cambodia": "KHM", "Laos": "LAO",
    "Uzbekistan": "UZB", "Nepal": "NPL",
}

# Indicadores del Banco Mundial (WDI) que usamos
WDI_INDICADORES = {
    "EN.GHG.CO2.TR.MT.CE.AR5": "co2_transporte_mt",   # CO2 del transporte (Mt CO2e)
    "EN.GHG.ALL.PC.CE.AR5": "ghg_per_capita",        # GEI totales per cápita (t CO2e)
    "NY.GDP.PCAP.KD": "pib_per_capita",              # PIB per cápita (US$ 2015 constantes)
    "SP.POP.TOTL": "poblacion",
    "EN.ATM.PM25.MC.M3": "pm25_medio_wdi",           # PM2.5 medio (µg/m3)
}


def cargar_iea():
    """Ventas, parque y puntos de recarga de vehículos eléctricos (IEA, datos históricos)."""
    ev = pd.read_excel(RUTA_DATOS / "EV data by country 2026.xlsx", sheet_name="GEVO_EV_2026")
    ev = ev[ev["category"].eq("Historical")].copy()        # excluimos proyecciones
    ev["iso3"] = ev["region_country"].map(IEA_A_ISO3)
    ev = ev.dropna(subset=["iso3"])

    coches = ev[ev["mode"].eq("Cars") & ev["powertrain"].eq("EV")]
    mapa = {
        "EV stock": "parque_ev",
        "EV sales": "ventas_ev",
        "EV stock share": "cuota_parque_pct",
        "EV sales share": "cuota_ventas_pct",
    }
    partes = []
    for param, nombre in mapa.items():
        s = coches[coches["parameter"].eq(param)]
        s = s.groupby(["iso3", "year"], as_index=False)["value"].first()
        # Las cuotas de coches ya vienen en porcentaje (Noruega 2024 = 35 %); no se reescalan.
        partes.append(s.rename(columns={"value": nombre}).set_index(["iso3", "year"]))

    recarga = ev[ev["parameter"].eq("EV charging points") & ev["mode"].eq("EVSE")]
    recarga = recarga.groupby(["iso3", "year"], as_index=False)["value"].sum()
    partes.append(recarga.rename(columns={"value": "puntos_recarga_publicos"}).set_index(["iso3", "year"]))

    out = pd.concat(partes, axis=1).reset_index().rename(columns={"year": "anio"})
    out["pais"] = out["iso3"].map(
        ev.drop_duplicates("iso3").set_index("iso3")["region_country"])
    return out


def cargar_wdi(anio_min=2010, anio_max=2025):
    """Indicadores del Banco Mundial (hoja 'Data' de WDIEXCEL.xlsx), pasados de formato ancho a largo."""
    w = pd.read_excel(RUTA_DATOS / "WDIEXCEL.xlsx", sheet_name="Data", engine="calamine")
    w = w[w["Indicator Code"].isin(WDI_INDICADORES)]
    cols = [c for c in w.columns if str(c).isdigit() and anio_min <= int(c) <= anio_max]
    largo = w.melt(id_vars=["Country Code", "Indicator Code"], value_vars=cols,
                   var_name="anio", value_name="valor")
    largo["anio"] = largo["anio"].astype(int)
    largo["indicador"] = largo["Indicator Code"].map(WDI_INDICADORES)
    ancho = largo.pivot_table(index=["Country Code", "anio"], columns="indicador",
                              values="valor", aggfunc="first").reset_index()
    ancho = ancho.rename(columns={"Country Code": "iso3"})
    ancho.columns.name = None
    return ancho


def cargar_afir():
    """Puntos de recarga por país y año en la UE (AFIR). 2026 es un dato parcial (hasta mayo)."""
    af = pd.read_excel(RUTA_DATOS / "Recharging points per power level (AFIR) dataset (2020 - May 2026).xlsx")
    af["iso3"] = af["Country"].map(AFIR_A_ISO3)
    af = af.dropna(subset=["iso3"])
    af["es_dc"] = af["Category"].str.contains("DC")
    total = af.groupby(["iso3", "Year"], as_index=False)["Total Recharging Points"].sum()
    dc = af[af["es_dc"]].groupby(["iso3", "Year"], as_index=False)["Total Recharging Points"].sum()
    out = total.merge(dc, on=["iso3", "Year"], how="left", suffixes=("", "_dc"))
    out = out.rename(columns={"Year": "anio", "Total Recharging Points": "afir_puntos_total",
                              "Total Recharging Points_dc": "afir_puntos_dc"})
    return out


def cargar_gni():
    """Renta nacional bruta per cápita (método Atlas), tabla de ranking del Banco Mundial."""
    raw = pd.read_excel(RUTA_DATOS / "GNIPC.xlsx", header=None)
    tabla = raw.iloc[7:, [0, 3, 4]].copy()
    tabla.columns = ["iso3", "pais_gni", "gni_per_capita"]
    tabla = tabla[tabla["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")]
    tabla["gni_per_capita"] = pd.to_numeric(tabla["gni_per_capita"], errors="coerce")
    return tabla.dropna(subset=["gni_per_capita"]).drop_duplicates("iso3")


CONTAMINANTES_FUA = {"NO2": "no2_pred", "PM25": "pm25_pred", "O3": "o3_pred", "CO": "co_pred"}


def procesar_fua_contaminante(nombre):
    """Lee por bloques el CSV de un contaminante (2025) y calcula el valor por país.

    Agregación realmente implementada (un solo paso sobre las observaciones):
    1) por área urbana funcional (FUA), media de TODAS las observaciones disponibles
       (celda × día), calculada como suma / recuento acumulados por bloques;
    2) por país (ISO3), media simple de sus FUA, sin ponderar por población.
    No se calcula una media diaria por FUA antes de promediar: la alternativa de dos
    pasos se compara en outputs/tablas/control_agregacion_no2.csv (diferencia mediana
    por FUA de 0,003 en NO₂; el contraste H3 cambia en la tercera cifra decimal).

    Para NO₂, `no2_pred` es un valor predicho cuya unidad no está documentada en el
    archivo de origen. Guarda el resultado en data/procesados/fua_<contaminante>.csv.
    """
    col = CONTAMINANTES_FUA[nombre]
    ruta = RUTA_DATOS / f"FUA_{nombre}_TimeSeries_2025.csv"
    suma = None
    nombres = []
    cols = ["fua_cntry", "fua_id", col] + (["fua_name"] if nombre == "NO2" else [])
    for trozo in pd.read_csv(ruta, usecols=cols, chunksize=2_000_000):
        g = trozo.groupby(["fua_cntry", "fua_id"])[col].agg(["sum", "count"])
        suma = g if suma is None else suma.add(g, fill_value=0)
        if nombre == "NO2":
            nombres.append(trozo[["fua_cntry", "fua_id", "fua_name"]].drop_duplicates())
    if nombre == "NO2":
        nm = pd.concat(nombres)
        nm["con_nombre"] = nm["fua_name"].notna() & (nm["fua_name"].astype(str).str.strip() != "")
        nm = nm.sort_values("con_nombre", ascending=False).drop_duplicates(["fua_cntry", "fua_id"])
        resumen = {"nombres_distintos_fua_cntry": int(nm["fua_cntry"].nunique()),
                   "ciudades_totales": int(len(nm)),
                   "ciudades_sin_nombre": int((~nm["con_nombre"]).sum())}
        (RUTA_PROCESADOS / "fua_no2_resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    ciudad = (suma["sum"] / suma["count"]).rename("valor").reset_index()
    ciudad["iso3"] = ciudad["fua_cntry"].map(FUA_A_ISO3)
    ciudad = ciudad.dropna(subset=["iso3"])
    pais = ciudad.groupby("iso3").agg(**{f"{nombre.lower()}_2025": ("valor", "mean"),
                                         f"ciudades_{nombre.lower()}": ("fua_id", "count")}).reset_index()
    pais.to_csv(RUTA_PROCESADOS / f"fua_{nombre.lower()}.csv", index=False)
    return pais


def cargar_fua_2025():
    """Une los cuatro contaminantes ya procesados en una tabla por país (ISO3)."""
    tablas = [pd.read_csv(RUTA_PROCESADOS / f"fua_{n.lower()}.csv") for n in CONTAMINANTES_FUA]
    out = tablas[0]
    for t in tablas[1:]:
        out = out.merge(t, on="iso3", how="outer")
    return out


def combinar_panel(iea, wdi, afir, gni):
    """Panel país-año de los 61 países del estudio.

    Se usa la unión completa (outer) de la IEA y el Banco Mundial, no solo la IEA:
    así un dato del Banco Mundial (p. ej. CO2 de 2015) no se pierde porque el
    país no tenga fila en la IEA para ese año. Antes del filtro, el WDI incluye
    agregados regionales (Mundo, UE, etc.), que se eliminan aquí.
    """
    paises = iea["iso3"].unique()
    wdi = wdi[wdi["iso3"].isin(paises)]
    panel = iea.merge(wdi, on=["iso3", "anio"], how="outer")
    nombres = iea.drop_duplicates("iso3").set_index("iso3")["pais"]
    panel["pais"] = panel["pais"].fillna(panel["iso3"].map(nombres))
    panel = panel.merge(afir.rename(columns={"Year": "anio"}), on=["iso3", "anio"], how="left")
    panel = panel.merge(gni, on="iso3", how="left")
    return panel.sort_values(["iso3", "anio"]).reset_index(drop=True)
