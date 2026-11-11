# Electromovilidad, recarga, CO₂ del transporte y NO₂ urbano: EDA de 61 países

## Objetivo

Analizar si la adopción del vehículo eléctrico y la infraestructura de recarga se asocian con el CO₂ del transporte por habitante y con el NO₂ urbano predicho en 61 países (2010–2025). Es un estudio observacional: las asociaciones no implican causalidad.

## Requisitos

- Python 3.10 (verificado con 3.10.12).
- Dependencias fijadas en `requirements.txt`, más Jupyter (`jupyter` o `notebook`).
- No requiere conexión a internet.

## Datos

Para ejecutar el estudio se usan los ficheros de `src/data/` incluidos en el repositorio:

- Originales: `EV data by country 2026.xlsx` (IEA), `Recharging points per power level (AFIR) dataset (2020 - May 2026).xlsx` (AFIR), `GNIPC.xlsx` (Banco Mundial) y `geo/paises.geojson` (mapa).
- Procesados: todo `src/data/procesados/`.

No son necesarios para ejecutar el análisis y no están incluidos: `WDIEXCEL.xlsx` (Banco Mundial) y `FUA_*_TimeSeries_2025.csv`. Los procesados `wdi_indicadores.csv`, `wdi_extendido_2000_2025.csv` y `fua_*.csv` se generaron con un proceso anterior que no forma parte del repositorio, por lo que no pueden regenerarse desde cero. El análisis los utiliza tal cual.

## Instalación

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt jupyter
```

Linux o macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
```

Verificado solo en Linux.

## Ejecución

### Ejecutar el estudio

1. Entrar en `src/` y abrir el notebook:

   ```bash
   cd src
   jupyter notebook main.ipynb
   ```

2. En Jupyter, elegir *Kernel → Restart & Run All*. Tarda alrededor de un minuto.

El notebook principal lee `outputs/tablas/contrastes_ajustados.csv`, que ya viene incluido. No es necesario ejecutar los scripts siguientes para ejecutar el estudio.

### Regenerar resultados auxiliares (opcional)

Los tres scripts regeneran las tablas de robustez de `src/outputs/`. Ejecutarlos desde `src/` y en este orden:

```bash
python generar_sensibilidad.py
python generar_influencia.py
python generar_contrastes_ajustados.py
```

`generar_contrastes_ajustados.py` depende de los dos anteriores. Cada script sobrescribe sus ficheros de salida.

## Estructura

```
.
├── README.md, FUENTES.md, requirements.txt
├── documentacion/        Material auxiliar
├── entregables/          Memoria y presentación
└── src/
    ├── main.ipynb                       Notebook principal
    ├── utils/                           Carga de datos, análisis, gráficos y sensibilidad
    ├── generar_sensibilidad.py          ┐
    ├── generar_influencia.py            ├ Tablas de robustez (opcionales)
    ├── generar_contrastes_ajustados.py  ┘
    ├── data/                            Datos de entrada y procesados
    └── outputs/                         Resultados
```
