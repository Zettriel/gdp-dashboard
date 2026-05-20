"""
╔══════════════════════════════════════════════════════════════════╗
║    ANÁLISIS DE REGRESIÓN LINEAL MÚLTIPLE                        ║
║    Banco de Bogotá S.A. — BVC: BOGOTA                          ║
║    Variables: Tasa BanRep + COLCAP → Precio Acción             ║
╚══════════════════════════════════════════════════════════════════╝

Autor: Trabajo de clase — Finanzas / Econometría
Fuentes de datos: CSV local (datos reales 2018-2026) + Banco de la República

CÓMO EJECUTAR:
    streamlit run streamlit_app.py

LIBRERÍAS NECESARIAS (instalar una sola vez):
    pip install streamlit yfinance pandas numpy statsmodels plotly scikit-learn scipy
"""

# ─────────────────────────────────────────────
# 0. IMPORTACIONES
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Regresión — Banco de Bogotá",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;
        border-left: 5px solid #e94560;
    }
    .main-header h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { color: #a0aec0; margin: 0.4rem 0 0 0; font-size: 1rem; }
    .main-header .badge {
        display: inline-block; background: #e94560; color: white;
        padding: 3px 10px; border-radius: 20px; font-size: 0.75rem;
        font-weight: 600; margin-top: 0.5rem; letter-spacing: 1px;
    }
    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1.2rem 1.5rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .value  { font-size: 2rem; font-weight: 700; color: #1a1a2e; font-family: 'IBM Plex Mono', monospace; }
    .metric-card .label  { font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
    .metric-card .sublabel { font-size: 0.75rem; color: #a0aec0; margin-top: 0.1rem; }
    .diag-ok   { color: #38a169; font-weight: 700; }
    .diag-warn { color: #d69e2e; font-weight: 700; }
    .diag-bad  { color: #e53e3e; font-weight: 700; }
    .explain-box {
        background: #f7fafc; border-left: 4px solid #4299e1;
        padding: 1rem 1.2rem; border-radius: 0 8px 8px 0;
        margin: 1rem 0; font-size: 0.9rem; color: #2d3748;
    }
    .explain-box strong { color: #1a1a2e; }
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #1a1a2e;
        border-bottom: 2px solid #e94560; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3. DICCIONARIO DE TASAS BANREP (completo 2018-2026)
# ─────────────────────────────────────────────
TASAS_BANREP = {
    "2018-01": 4.75, "2018-02": 4.50, "2018-03": 4.50, "2018-04": 4.25,
    "2018-05": 4.25, "2018-06": 4.25, "2018-07": 4.25, "2018-08": 4.25,
    "2018-09": 4.25, "2018-10": 4.25, "2018-11": 4.25, "2018-12": 4.25,
    "2019-01": 4.25, "2019-02": 4.25, "2019-03": 4.25, "2019-04": 4.25,
    "2019-05": 4.25, "2019-06": 4.25, "2019-07": 4.25, "2019-08": 4.25,
    "2019-09": 4.25, "2019-10": 4.25, "2019-11": 4.25, "2019-12": 4.25,
    "2020-01": 4.25, "2020-02": 4.25, "2020-03": 3.75, "2020-04": 3.25,
    "2020-05": 3.00, "2020-06": 2.75, "2020-07": 2.50, "2020-08": 2.25,
    "2020-09": 2.00, "2020-10": 1.75, "2020-11": 1.75, "2020-12": 1.75,
    "2021-01": 1.75, "2021-02": 1.75, "2021-03": 1.75, "2021-04": 1.75,
    "2021-05": 1.75, "2021-06": 1.75, "2021-07": 1.75, "2021-08": 1.75,
    "2021-09": 2.00, "2021-10": 2.50, "2021-11": 3.00, "2021-12": 3.00,
    "2022-01": 4.00, "2022-02": 4.00, "2022-03": 5.00, "2022-04": 6.00,
    "2022-05": 7.50, "2022-06": 9.00, "2022-07": 10.00,"2022-08": 10.00,
    "2022-09": 11.00,"2022-10": 12.00,"2022-11": 12.00,"2022-12": 12.00,
    "2023-01": 12.75,"2023-02": 12.75,"2023-03": 13.00,"2023-04": 13.00,
    "2023-05": 13.25,"2023-06": 13.25,"2023-07": 13.25,"2023-08": 13.25,
    "2023-09": 13.25,"2023-10": 13.25,"2023-11": 13.25,"2023-12": 13.00,
    "2024-01": 12.75,"2024-02": 12.75,"2024-03": 12.25,"2024-04": 12.00,
    "2024-05": 11.75,"2024-06": 11.50,"2024-07": 11.25,"2024-08": 11.00,
    "2024-09": 10.75,"2024-10": 10.50,"2024-11": 10.25,"2024-12": 10.00,
    "2025-01": 9.75, "2025-02": 9.50, "2025-03": 9.25, "2025-04": 9.00,
    "2025-05": 9.00, "2025-06": 9.00, "2025-07": 9.00, "2025-08": 9.00,
    "2025-09": 9.25, "2025-10": 9.75, "2025-11": 10.25,"2025-12": 10.75,
    "2026-01": 11.25,"2026-02": 11.25,"2026-03": 11.25,"2026-04": 11.25,
    "2026-05": 11.25,
}


# ─────────────────────────────────────────────
# 4. FUNCIÓN PRINCIPAL DE CARGA DE DATOS
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def descargar_datos(fecha_inicio: str, fecha_fin: str):
    """
    Carga los datos con sistema de 3 capas:

    CAPA 1 — CSV local (datos/datos_bogota.csv):
        Datos reales 2018-2026 que descargaste y subiste a GitHub.
        Es la fuente más confiable. Si existe, la usa siempre.

    CAPA 2 — Yahoo Finance en vivo:
        Solo se intenta si el CSV no existe. Puede fallar en
        Streamlit Cloud por restricciones de red.

    CAPA 3 — Modo demostración:
        Datos simulados con patrones reales como último recurso.
        Solo para que la app nunca se quede en blanco.

    Retorna: (DataFrame, fuente) donde fuente es "real", "yahoo" o "demo"
    """

    # ── CAPA 1: CSV LOCAL ──────────────────────────────────────────
    rutas_csv = [
        "datos/datos_bogota.csv",   # Streamlit Cloud (carpeta datos/)
        "datos_bogota.csv",         # Raíz del proyecto (alternativa)
    ]

    for ruta in rutas_csv:
        try:
            df = pd.read_csv(ruta, index_col=0, parse_dates=True)

            # Verificar que tiene las columnas esperadas
            if "precio_bogota" not in df.columns or "colcap" not in df.columns:
                continue  # Probar la siguiente ruta

            # Filtrar por el rango de fechas del selector
            df = df[
                (df.index >= pd.to_datetime(fecha_inicio)) &
                (df.index <= pd.to_datetime(fecha_fin))
            ].copy()

            if len(df) < 30:
                continue  # Muy pocos datos para ese rango, probar siguiente

            # Agregar tasa BanRep (siempre viene del diccionario, no del CSV)
            df["tasa_banrep"] = df.index.to_period("M").astype(str).map(TASAS_BANREP)
            df["tasa_banrep"] = df["tasa_banrep"].ffill().bfill()
            df = df.dropna()

            return df, "real"  # ✅ Éxito — datos reales

        except FileNotFoundError:
            continue  # El archivo no existe en esta ruta, probar la siguiente
        except Exception:
            continue  # Cualquier otro error, probar la siguiente

    # ── CAPA 2: YAHOO FINANCE EN VIVO ─────────────────────────────
    try:
        import yfinance as yf

        bogota_raw = yf.download("BOGOTA.CL", start=fecha_inicio, end=fecha_fin,
                                  progress=False, auto_adjust=True)
        colcap_raw = yf.download("^COLCAP",   start=fecha_inicio, end=fecha_fin,
                                  progress=False, auto_adjust=True)

        # Extraer Close con manejo de MultiIndex
        def _extraer_close(df_raw, ticker):
            if isinstance(df_raw.columns, pd.MultiIndex):
                if "Close" in df_raw.columns.get_level_values(0):
                    s = df_raw["Close"]
                    if isinstance(s, pd.DataFrame):
                        s = s.iloc[:, 0]
                    return s.squeeze()
            if "Close" in df_raw.columns:
                return df_raw["Close"].squeeze()
            return pd.Series(dtype=float)

        bogota = _extraer_close(bogota_raw, "BOGOTA.CL")
        colcap = _extraer_close(colcap_raw, "^COLCAP")

        if not bogota.empty and not colcap.empty:
            df = pd.DataFrame({
                "precio_bogota": bogota,
                "colcap": colcap
            }).dropna()

            df["tasa_banrep"] = df.index.to_period("M").astype(str).map(TASAS_BANREP)
            df["tasa_banrep"] = df["tasa_banrep"].ffill().bfill()
            df = df.dropna()

            if len(df) >= 30:
                return df, "yahoo"  # ✅ Éxito — Yahoo Finance

    except Exception:
        pass  # Yahoo falló, ir a modo demo

    # ── CAPA 3: MODO DEMOSTRACIÓN ──────────────────────────────────
    np.random.seed(2024)
    n_demo = 350
    fechas_demo = pd.date_range("2021-06-01", periods=n_demo, freq="B")
    tasa_demo = np.concatenate([
        np.linspace(1.75, 13.25, 120),
        np.linspace(13.25, 11.25, 130),
        np.linspace(11.25, 11.25, 100),
    ])
    colcap_demo = np.concatenate([
        np.linspace(1100, 1600, 120) + np.random.randn(120) * 40,
        np.linspace(1600, 2400, 130) + np.random.randn(130) * 60,
        np.linspace(2400, 2100, 100) + np.random.randn(100) * 50,
    ])
    precio_demo = (
        42000
        - 850  * tasa_demo
        + 11.5 * colcap_demo
        + np.random.randn(n_demo) * 1800
    )
    df_demo = pd.DataFrame({
        "precio_bogota": precio_demo,
        "colcap": colcap_demo,
        "tasa_banrep": tasa_demo,
    }, index=fechas_demo)

    return df_demo, "demo"  # ⚠️ Datos simulados


# ─────────────────────────────────────────────
# 5. FUNCIONES ESTADÍSTICAS
# ─────────────────────────────────────────────

def correr_regresion(df: pd.DataFrame):
    Y = df["precio_bogota"]
    X = sm.add_constant(df[["tasa_banrep", "colcap"]])
    return sm.OLS(Y, X).fit()

def prueba_normalidad(residuos):
    return stats.jarque_bera(residuos)

def prueba_autocorrelacion(residuos):
    from statsmodels.stats.stattools import durbin_watson
    return durbin_watson(residuos)

def prueba_heterocedasticidad(modelo):
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_lm, bp_p, _, _ = het_breuschpagan(modelo.resid, modelo.model.exog)
    return bp_lm, bp_p

def calcular_vif(df: pd.DataFrame):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = sm.add_constant(df[["tasa_banrep", "colcap"]])
    vif_data = pd.DataFrame({
        "Variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data[vif_data["Variable"] != "const"]

def semaforo(condicion: bool, texto_ok: str, texto_mal: str):
    if condicion:
        return f'<span class="diag-ok">✅ {texto_ok}</span>'
    return f'<span class="diag-bad">❌ {texto_mal}</span>'


# ─────────────────────────────────────────────
# 6. SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Parámetros del Análisis")
    st.markdown("---")

    fecha_inicio = st.date_input(
        "📅 Fecha de inicio",
        value=pd.to_datetime("2019-01-01"),
        min_value=pd.to_datetime("2018-01-01"),
        max_value=pd.to_datetime("2024-01-01"),
    )

    fecha_fin = st.date_input(
        "📅 Fecha de fin",
        value=pd.to_datetime("2026-05-19"),
        min_value=pd.to_datetime("2018-06-01"),
        max_value=pd.to_datetime("2026-05-19"),
    )

    nivel_significancia = st.selectbox(
        "📊 Nivel de significancia (α)",
        options=[0.01, 0.05, 0.10],
        index=1,
        format_func=lambda x: f"{x*100:.0f}% (α = {x})"
    )

    st.markdown("---")
    st.markdown("**📋 Variables del modelo:**")
    st.markdown("""
    - 🎯 **Y**: Precio acción BOGOTA
    - 📉 **X₁**: Tasa BanRep (%)
    - 📈 **X₂**: Índice COLCAP
    """)

    st.markdown("---")
    st.markdown("**📡 Fuentes de datos:**")
    st.markdown("""
    - CSV local (datos reales 2018-2026)
    - Yahoo Finance (respaldo)
    - Banco de la República (tasas)
    """)

    ejecutar = st.button("🚀 Ejecutar Análisis", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# 7. CABECERA
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🏦 Banco de Bogotá — Regresión Lineal Múltiple</h1>
    <p>Modelo: Precio Acción = β₀ + β₁·(Tasa BanRep) + β₂·(COLCAP) + ε</p>
    <span class="badge">BVC: BOGOTA · Grupo Aval · Fundado 1870</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 8. EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if ejecutar or True:

    # ── 8.1 Cargar datos ──
    with st.spinner("📡 Cargando datos..."):
        df, fuente = descargar_datos(str(fecha_inicio), str(fecha_fin))

    # Banner de fuente — el usuario siempre sabe con qué datos está trabajando
    if fuente == "real":
        st.success(
            f"✅ **Datos reales cargados desde CSV** — {len(df):,} observaciones | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
    elif fuente == "yahoo":
        st.info(
            f"📡 **Datos en vivo desde Yahoo Finance** — {len(df):,} observaciones | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
    else:
        st.warning("""
        ⚠️ **Modo demostración** — No se encontró el archivo CSV ni Yahoo Finance respondió.
        
        Para usar datos reales: sube el archivo `datos/datos_bogota.csv` a tu repositorio de GitHub.
        Los resultados actuales muestran el comportamiento esperado del modelo con datos simulados.
        """)

    # ── 8.2 Regresión ──
    modelo      = correr_regresion(df)
    r2          = modelo.rsquared
    r2_ajustado = modelo.rsquared_adj
    f_stat      = modelo.fvalue
    f_pval      = modelo.f_pvalue
    coefs       = modelo.params
    pvals       = modelo.pvalues
    ic          = modelo.conf_int(alpha=nivel_significancia)
    residuos    = modelo.resid
    n           = len(df)

    df["predicho"] = modelo.fittedvalues
    df["error"]    = residuos

    jb_stat, jb_p = prueba_normalidad(residuos)
    dw_stat        = prueba_autocorrelacion(residuos)
    bp_stat, bp_p  = prueba_heterocedasticidad(modelo)
    vif_df         = calcular_vif(df)
    rmse           = np.sqrt(mean_squared_error(df["precio_bogota"], df["predicho"]))
    mae            = mean_absolute_error(df["precio_bogota"], df["predicho"])
    mape           = np.mean(np.abs((df["precio_bogota"] - df["predicho"]) / df["precio_bogota"])) * 100
    α              = nivel_significancia

    # ══════════════════════════════════════════
    # SECCIÓN A: DATOS
    # ══════════════════════════════════════════
    st.markdown('<div class="section-title">📊 A. DATOS CARGADOS</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{n:,}</div>
            <div class="label">Observaciones</div>
            <div class="sublabel">días de trading</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="value">${df['precio_bogota'].mean():,.0f}</div>
            <div class="label">Precio prom. BOGOTA</div>
            <div class="sublabel">COP por acción</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{df['colcap'].mean():,.0f}</div>
            <div class="label">COLCAP promedio</div>
            <div class="sublabel">puntos del índice</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="value">{df['tasa_banrep'].mean():.1f}%</div>
            <div class="label">Tasa BanRep prom.</div>
            <div class="sublabel">% anual</div>
        </div>""", unsafe_allow_html=True)

    with st.expander("📋 Ver tabla de datos completa"):
        df_mostrar = df[["precio_bogota", "colcap", "tasa_banrep", "predicho", "error"]].copy()
        df_mostrar.index = df_mostrar.index.strftime("%Y-%m-%d")
        df_mostrar.columns = ["Precio BOGOTA (COP)", "COLCAP (puntos)", "Tasa BanRep (%)", "Predicho", "Error"]
        st.dataframe(df_mostrar.round(2), use_container_width=True)

    # ══════════════════════════════════════════
    # SECCIÓN B: REGRESIÓN
    # ══════════════════════════════════════════
    st.markdown('<div class="section-title">📐 B. RESULTADOS DE LA REGRESIÓN</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    <strong>¿Qué estamos mirando aquí?</strong> La regresión encontró los valores exactos de
    β₀ (intercepto), β₁ (efecto de la tasa BanRep) y β₂ (efecto del COLCAP) que mejor explican
    el precio de la acción. La columna p-valor nos dice si ese efecto es real o pudo ocurrir por azar.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Ecuación estimada del modelo:**")
        signo1 = "+" if coefs["tasa_banrep"] >= 0 else "-"
        signo2 = "+" if coefs["colcap"]      >= 0 else "-"
        st.markdown(f"""
        <div style="background:#1a1a2e; color:white; padding:1.2rem; border-radius:8px;
                    font-family:'IBM Plex Mono',monospace; font-size:0.9rem; line-height:2;">
        <strong>Precio</strong> = {coefs['const']:,.1f}<br>
        &nbsp;&nbsp;&nbsp; {signo1} {abs(coefs['tasa_banrep']):,.1f} × (Tasa BanRep)<br>
        &nbsp;&nbsp;&nbsp; {signo2} {abs(coefs['colcap']):.4f} × (COLCAP)<br>
        &nbsp;&nbsp;&nbsp; + ε (error)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Tabla de coeficientes:**")
        tabla_coefs = pd.DataFrame({
            "Variable":      ["Constante (β₀)",  "Tasa BanRep (β₁)", "COLCAP (β₂)"],
            "Coeficiente":   [f"{coefs['const']:,.2f}", f"{coefs['tasa_banrep']:,.2f}", f"{coefs['colcap']:.4f}"],
            "Std Error":     [f"{modelo.bse['const']:,.2f}", f"{modelo.bse['tasa_banrep']:,.2f}", f"{modelo.bse['colcap']:.4f}"],
            "t-estadístico": [f"{modelo.tvalues['const']:.3f}", f"{modelo.tvalues['tasa_banrep']:.3f}", f"{modelo.tvalues['colcap']:.3f}"],
            "p-valor":       [f"{pvals['const']:.4f}", f"{pvals['tasa_banrep']:.4f}", f"{pvals['colcap']:.4f}"],
            "Significativo": [
                "✅ Sí" if pvals['const']       < α else "❌ No",
                "✅ Sí" if pvals['tasa_banrep'] < α else "❌ No",
                "✅ Sí" if pvals['colcap']      < α else "❌ No",
            ]
        })
        st.dataframe(tabla_coefs, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Métricas de ajuste del modelo:**")
        metricas = [
            ("R² (R cuadrado)",    f"{r2:.4f}",          f"{r2*100:.1f}% de la variación explicada"),
            ("R² Ajustado",        f"{r2_ajustado:.4f}", "Penaliza por número de variables"),
            ("F-estadístico",      f"{f_stat:.2f}",       "Significancia global del modelo"),
            ("p-valor del F",      f"{f_pval:.6f}",       f"{'Modelo significativo ✅' if f_pval < α else 'Modelo NO significativo ❌'}"),
            ("N observaciones",    f"{n}",                "Días de trading analizados"),
            ("RMSE",               f"{rmse:,.1f} COP",    "Error cuadrático medio"),
            ("MAE",                f"{mae:,.1f} COP",     "Error absoluto medio"),
            ("MAPE",               f"{mape:.2f}%",        "Error porcentual medio"),
            ("Log-verosimilitud",  f"{modelo.llf:.1f}",   "Bondad de ajuste"),
            ("AIC",                f"{modelo.aic:.1f}",   "Criterio Akaike (menor = mejor)"),
            ("BIC",                f"{modelo.bic:.1f}",   "Criterio Bayesiano (menor = mejor)"),
        ]
        for nombre, valor, descripcion in metricas:
            ca, cb = st.columns([1.2, 1.8])
            with ca: st.markdown(f"**{nombre}**")
            with cb: st.markdown(f"`{valor}` — *{descripcion}*")

    # ══════════════════════════════════════════
    # SECCIÓN C: DIAGNÓSTICO
    # ══════════════════════════════════════════
    st.markdown('<div class="section-title">🔬 C. PRUEBAS DE DIAGNÓSTICO</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    <strong>¿Por qué hacemos estas pruebas?</strong> Un modelo de regresión tiene supuestos
    que deben cumplirse para que los resultados sean válidos. Si no se cumplen, los coeficientes
    y p-valores podrían ser engañosos.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 1️⃣ Normalidad de Residuos — Test Jarque-Bera")
        st.markdown("""
        **¿Qué mide?** Los errores deben distribuirse como campana de Gauss (distribución normal).
        Si el p-valor > α, los errores son normales y el supuesto se cumple.
        """)
        normalidad_ok = jb_p > α
        st.markdown(f"""
        | Estadístico JB | p-valor | Resultado |
        |---|---|---|
        | {jb_stat:.4f} | {jb_p:.4f} | {semaforo(normalidad_ok, f"Normalidad aceptada (p={jb_p:.4f} > α={α})", f"Se rechaza normalidad (p={jb_p:.4f} < α={α})")} |
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 2️⃣ Autocorrelación — Durbin-Watson")
        st.markdown("""
        **¿Qué mide?** Los errores de un día no deben predecir los del día siguiente.
        Valor ideal = **2.0**. Rango aceptable: **1.5 a 2.5**.
        """)
        dw_ok = 1.5 <= dw_stat <= 2.5
        st.markdown(f"""
        | Estadístico DW | Rango aceptable | Resultado |
        |---|---|---|
        | {dw_stat:.4f} | 1.5 – 2.5 | {semaforo(dw_ok, "Sin autocorrelación problemática", "Posible autocorrelación")} |
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("##### 3️⃣ Homocedasticidad — Breusch-Pagan")
        st.markdown("""
        **¿Qué mide?** La variabilidad de los errores debe ser constante en toda la muestra.
        Si el p-valor > α, la varianza es constante y el supuesto se cumple.
        """)
        homo_ok = bp_p > α
        st.markdown(f"""
        | Estadístico BP | p-valor | Resultado |
        |---|---|---|
        | {bp_stat:.4f} | {bp_p:.4f} | {semaforo(homo_ok, f"Homocedasticidad aceptada (p={bp_p:.4f} > α={α})", f"Heterocedasticidad detectada (p={bp_p:.4f} < α={α})")} |
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### 4️⃣ Multicolinealidad — VIF")
        st.markdown("""
        **¿Qué mide?** ¿Están X₁ y X₂ diciendo lo mismo?
        **Regla:** VIF < 5 → sin problema ✅ | VIF 5-10 → moderado | VIF > 10 → grave ❌
        """)
        for _, row in vif_df.iterrows():
            vif_val = row["VIF"]
            vif_ok  = vif_val < 5
            nombre  = "Tasa BanRep" if "tasa" in row["Variable"] else "COLCAP"
            st.markdown(
                f"**{nombre}**: VIF = `{vif_val:.3f}` — "
                f"{semaforo(vif_ok, 'Sin multicolinealidad', 'Posible multicolinealidad')}",
                unsafe_allow_html=True
            )

    # Resumen semáforo
    st.markdown("##### 🚦 Resumen del Diagnóstico")
    checks = {
        "Normalidad de residuos (Jarque-Bera)":     normalidad_ok,
        "Sin autocorrelación (Durbin-Watson)":       dw_ok,
        "Homocedasticidad (Breusch-Pagan)":          homo_ok,
        "Sin multicolinealidad (VIF < 5)":           all(vif_df["VIF"] < 5),
        "Modelo globalmente significativo (F-test)": f_pval < α,
        "R² aceptable (> 0.50)":                     r2 > 0.50,
    }
    cols = st.columns(3)
    for i, (nombre, ok) in enumerate(checks.items()):
        with cols[i % 3]:
            icono  = "✅" if ok else "❌"
            color  = "#e6f4ea" if ok else "#fce8e6"
            border = "#34a853" if ok else "#ea4335"
            st.markdown(f"""
            <div style="background:{color}; border-left:4px solid {border};
                        padding:0.6rem 0.8rem; border-radius:4px; margin:0.3rem 0; font-size:0.85rem;">
                {icono} {nombre}
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # SECCIÓN D: GRÁFICAS
    # ══════════════════════════════════════════
    st.markdown('<div class="section-title">📈 D. VISUALIZACIONES</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Serie de Tiempo", "🎯 Real vs Predicho", "📊 Residuos", "🔗 Correlaciones"
    ])

    with tab1:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Precio Acción BOGOTA (COP)", "Índice COLCAP (puntos)", "Tasa BanRep (%)"),
            vertical_spacing=0.08
        )
        fig.add_trace(go.Scatter(x=df.index, y=df["precio_bogota"],
                                  name="Precio BOGOTA", line=dict(color="#e94560", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["predicho"],
                                  name="Predicho", line=dict(color="#f6ad55", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["colcap"],
                                  name="COLCAP", line=dict(color="#4299e1", width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["tasa_banrep"],
                                  name="Tasa BanRep", line=dict(color="#48bb78", width=2)), row=3, col=1)
        fig.update_layout(height=550, template="plotly_white", showlegend=True,
                          title_text="Variables del modelo a lo largo del tiempo")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig2 = px.scatter(df, x="precio_bogota", y="predicho",
                          title="Valores Reales vs. Valores Predichos por el modelo",
                          labels={"precio_bogota": "Precio Real (COP)", "predicho": "Precio Predicho (COP)"},
                          opacity=0.6, color_discrete_sequence=["#e94560"])
        rango = [df["precio_bogota"].min(), df["precio_bogota"].max()]
        fig2.add_trace(go.Scatter(x=rango, y=rango, mode="lines",
                                   name="Predicción perfecta",
                                   line=dict(color="gray", dash="dash")))
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""
        <div class="explain-box">
        <strong>¿Cómo leer esta gráfica?</strong> Cada punto es una observación. Si el modelo
        fuera perfecto, todos los puntos estarían sobre la línea gris diagonal. Nuestro R² =
        <strong>{r2:.4f}</strong> indica que el modelo explica el <strong>{r2*100:.1f}%</strong>
        de la variación del precio.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig3 = px.histogram(residuos, nbins=40, title="Distribución de los Residuos",
                                labels={"value": "Error (COP)", "count": "Frecuencia"},
                                color_discrete_sequence=["#4299e1"])
            x_norm = np.linspace(residuos.min(), residuos.max(), 100)
            y_norm = stats.norm.pdf(x_norm, residuos.mean(), residuos.std()) * len(residuos) * (residuos.max() - residuos.min()) / 40
            fig3.add_trace(go.Scatter(x=x_norm, y=y_norm, mode="lines", name="Normal teórica",
                                       line=dict(color="#e94560", width=2)))
            fig3.update_layout(template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            (osm, osr), (slope, intercept, _) = stats.probplot(residuos, dist="norm")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Cuantiles residuos",
                                       marker=dict(color="#4299e1", size=4, opacity=0.6)))
            fig4.add_trace(go.Scatter(x=osm, y=[slope * x + intercept for x in osm],
                                       mode="lines", name="Línea teórica normal",
                                       line=dict(color="#e94560", dash="dash")))
            fig4.update_layout(title="Q-Q Plot (Normalidad)", template="plotly_white",
                                xaxis_title="Cuantiles teóricos", yaxis_title="Cuantiles observados")
            st.plotly_chart(fig4, use_container_width=True)

        fig5 = px.scatter(x=df.index, y=residuos,
                          title="Residuos en el Tiempo (deben estar centrados en cero, sin patrón)",
                          labels={"x": "Fecha", "y": "Error (COP)"},
                          opacity=0.5, color_discrete_sequence=["#e94560"])
        fig5.add_hline(y=0,                   line_dash="dash", line_color="black")
        fig5.add_hline(y= 2*residuos.std(),   line_dash="dot",  line_color="orange", annotation_text="+2σ")
        fig5.add_hline(y=-2*residuos.std(),   line_dash="dot",  line_color="orange", annotation_text="-2σ")
        fig5.update_layout(template="plotly_white")
        st.plotly_chart(fig5, use_container_width=True)

    with tab4:
        corr_matrix = df[["precio_bogota", "tasa_banrep", "colcap"]].corr()
        fig6 = px.imshow(corr_matrix, text_auto=".3f", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, title="Matriz de Correlaciones",
                          labels=dict(x="Variable", y="Variable", color="Correlación"))
        fig6.update_layout(template="plotly_white")
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("""
        <div class="explain-box">
        <strong>¿Cómo leer la matriz?</strong> Valores cerca de <strong>+1</strong> = relación directa.
        Valores cerca de <strong>-1</strong> = relación inversa. Valores cerca de <strong>0</strong> =
        sin relación lineal. Las dos variables X entre sí deben tener correlación baja para evitar multicolinealidad.
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # SECCIÓN E: INTERPRETACIÓN ECONÓMICA
    # ══════════════════════════════════════════
    st.markdown('<div class="section-title">💡 E. INTERPRETACIÓN ECONÓMICA</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        signo_tasa = "sube" if coefs["tasa_banrep"] > 0 else "baja"
        dir_tasa   = "positiva" if coefs["tasa_banrep"] > 0 else "negativa (inversa)"
        st.markdown(f"""
        **β₁ — Efecto de la Tasa BanRep:** `{coefs['tasa_banrep']:,.2f} COP`

        Cuando la tasa BanRep sube **1 punto porcentual**, el precio de la acción
        **{signo_tasa} {abs(coefs['tasa_banrep']):,.0f} COP** en promedio
        *(relación {dir_tasa})*.

        📌 Cuando el BanRep sube tasas, los bancos pagan más por sus depósitos pero
        cobran más lento por sus créditos. El margen se comprime, las utilidades bajan
        y el precio de la acción cae.
        """)
    with col2:
        signo_colcap = "sube" if coefs["colcap"] > 0 else "baja"
        st.markdown(f"""
        **β₂ — Efecto del COLCAP:** `{coefs['colcap']:.4f} COP por punto`

        Cuando el COLCAP sube **100 puntos**, el precio de la acción
        **{signo_colcap} {abs(coefs['colcap'])*100:,.1f} COP** en promedio.

        📌 El COLCAP captura el estado de ánimo general del mercado colombiano.
        Cuando los inversionistas confían en Colombia, compran todas las acciones
        del índice, incluida BOGOTA.
        """)

    st.info(f"""
    **Conclusión del modelo:** Con un R² = **{r2:.4f}** ({r2*100:.1f}% de variación explicada)
    y un F-test con p-valor = **{f_pval:.6f}**, el modelo {'ES' if f_pval < α else 'NO ES'}
    estadísticamente significativo con α = {α}.
    El error promedio de predicción es **{mape:.1f}%** del precio real.
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#a0aec0; font-size:0.8rem;">
    Trabajo académico · Regresión Lineal Múltiple · Banco de Bogotá S.A. (BVC: BOGOTA)<br>
    Datos: CSV local (2018-2026) · Banco de la República de Colombia · statsmodels · Python
    </div>
    """, unsafe_allow_html=True)