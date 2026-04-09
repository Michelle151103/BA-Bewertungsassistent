# =============================================================================
# BEWERTUNGSASSISTENT – Region Stuttgart
# Streamlit Web-App | Bachelorarbeit
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='Bewertungsassistent Stuttgart',
    page_icon='🏠',
    layout='centered'
)

C_HAUPT  = '#534AB7'
C_AKZENT = '#BA7517'
C_ROT    = '#A32D2D'

def fmt_eur(wert):
    """Formatiert einen Eurobetrag mit Punkt als Tausendertrennzeichen."""
    return f"{wert:,.0f} €".replace(",", ".")

def fmt_kurz(wert):
    """Formatiert einen Eurobetrag in Tausend mit Punkt."""
    return f"{wert/1000:,.0f}k €".replace(",", ".")

@st.cache_resource
def modell_trainieren():
    df = pd.read_excel('Daten GAU.xlsx')

    def klassenmitte(s):
        teile = str(s).split('-')
        try:
            return (float(teile[0]) + float(teile[-1])) / 2
        except Exception:
            return np.nan

    df['flaeche_m2'] = df['Wohnflaeche_Klasse_m2'].apply(klassenmitte)
    df['baujahr']    = df['Baujahr_Klasse'].apply(klassenmitte)
    df['kaufpreis']  = df['Kaufpreis_EUR']
    df['bezirk']     = df['Stadtbezirk'].str.strip()
    df['alter']      = 2024 - df['baujahr']
    df['preis_pro_m2'] = df['kaufpreis'] / df['flaeche_m2']

    Q1, Q3 = df['kaufpreis'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[
        (df['kaufpreis'] >= Q1 - 1.5*IQR) &
        (df['kaufpreis'] <= Q3 + 1.5*IQR)
    ].dropna(subset=['flaeche_m2','baujahr','kaufpreis']).reset_index(drop=True)

    df['baujahr_klasse'] = pd.cut(
        df['baujahr'],
        bins=[0, 1944, 1964, 1984, 2004, 9999],
        labels=['vor 1945','1945–1964','1965–1984','1985–2004','ab 2005']
    )

    ref_bezirk = df['bezirk'].value_counts().idxmax()
    dummies = pd.get_dummies(df['bezirk'], drop_first=False, dtype=int)
    if ref_bezirk in dummies.columns:
        dummies = dummies.drop(columns=[ref_bezirk])
    dummies.columns = [
        'D_' + c.replace(' ','_').replace('-','_').replace('\u2011','_')
        for c in dummies.columns
    ]

    X = pd.concat([df[['flaeche_m2','alter']], dummies], axis=1)
    y = df['kaufpreis']
    modell = sm.OLS(y, sm.add_constant(X)).fit()

    df['schaetzwert'] = modell.fittedvalues.values
    df['residuum']    = df['kaufpreis'] - df['schaetzwert']
    df['ape']         = (df['residuum'].abs() / df['kaufpreis']) * 100
    mape_bezirk = df.groupby('bezirk')['ape'].mean().round(1)

    return modell, sorted(df['bezirk'].unique().tolist()), ref_bezirk, mape_bezirk, df

modell, bezirke, ref_bezirk, mape_bezirk, df = modell_trainieren()

st.title('🏠 Bewertungsassistent Stuttgart')
st.markdown('**Automatisierte Markteinschätzung auf Basis hedonischer Regressionsanalyse**')
st.markdown(f'*Trainingsdaten: Gutachterausschuss Stuttgart · n = {len(df)} Transaktionen*')
st.divider()

st.subheader('Objektparameter eingeben')

col1, col2, col3 = st.columns(3)

with col1:
    bezirk_input = st.selectbox('Stadtbezirk', bezirke)

with col2:
    flaeche_input = st.number_input(
        'Wohnfläche (m²)',
        min_value=20, max_value=300,
        value=75, step=5
    )

with col3:
    baujahr_input = st.number_input(
        'Baujahr',
        min_value=1850, max_value=2025,
        value=1985, step=1
    )

alter_input = 2024 - baujahr_input

def schaetzung_berechnen(bezirk, flaeche, alter):
    eingabe = pd.DataFrame({'const': [1.0],
                             'flaeche_m2': [flaeche],
                             'alter': [alter]})
    for col in modell.params.index:
        if col.startswith('D_') and col not in eingabe.columns:
            eingabe[col] = 0

    dummy_name = 'D_' + bezirk.replace(' ','_').replace('-','_').replace('\u2011','_')
    if dummy_name in eingabe.columns:
        eingabe[dummy_name] = 1

    eingabe = eingabe.reindex(columns=modell.params.index, fill_value=0)
    pred    = modell.get_prediction(eingabe)
    mw      = pred.predicted_mean[0]
    ci      = pred.conf_int(alpha=0.20)
    return mw, ci[0][0], ci[0][1]

schaetzwert, ci_low, ci_high = schaetzung_berechnen(
    bezirk_input, flaeche_input, alter_input
)
mape_bez = mape_bezirk.get(bezirk_input, None)

st.divider()
st.subheader('Bewertungsergebnis')

m1, m2, m3 = st.columns(3)
m1.metric('Marktpreisschätzung', fmt_eur(schaetzwert))
m2.metric('Preis pro m²', fmt_eur(schaetzwert / flaeche_input))
m3.metric('Gebäudealter', f'{alter_input:.0f} Jahre')

st.info(f'**Bandbreite (80 %):** {fmt_eur(ci_low)} — {fmt_eur(ci_high)}')

if mape_bez is not None:
    if mape_bez < 15:
        st.success(f'**Modellgenauigkeit in {bezirk_input}:** MAPE = {mape_bez:.1f}% — Gute Schätzqualität')
    elif mape_bez < 25:
        st.warning(f'**Modellgenauigkeit in {bezirk_input}:** MAPE = {mape_bez:.1f}% — Mittlere Schätzqualität, Bandbreite beachten')
    else:
        st.error(f'**Modellgenauigkeit in {bezirk_input}:** MAPE = {mape_bez:.1f}% — Eingeschränkte Schätzqualität, atypischer Teilmarkt')

st.divider()
st.subheader('Einordnung im Bezirksvergleich')

median_preise = df.groupby('bezirk')['kaufpreis'].median().sort_values()
farben = [C_HAUPT if b != bezirk_input else C_AKZENT for b in median_preise.index]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(median_preise.index, median_preise.values/1000, color=farben, alpha=0.85)
ax.axvline(schaetzwert/1000, color=C_ROT, lw=1.5, linestyle='--',
           label=f'Ihr Schätzwert: {fmt_kurz(schaetzwert)}')
ax.set_xlabel('Medianer Kaufpreis (Tsd. €)')
ax.set_title('Medianer Kaufpreis je Stadtbezirk', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()
st.caption(
    '**Methodischer Hinweis:** Der Marktpreisschätzung basiert auf einem '
    'hedonischen OLS-Regressionsmodell, trainiert auf Transaktionsdaten des Gutachterausschusses Stuttgart. '
    'Eingabevariablen: Wohnfläche, Gebäudealter, Stadtbezirk. '
    'Die Bandbreite entspricht dem 80%-Konfidenzintervall des Modells. '
    'Der Assistent ersetzt keine normierte Einzelbewertung nach ImmoWertV.'
)
