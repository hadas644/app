import streamlit as st
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import plotly.express as px
from functools import partial
import plotly.graph_objects as go

st.title('Fitting without IRF')

uploaded_file = st.file_uploader("Įkelkite savo duomenų failą:")

if uploaded_file is None:
  st.stop()

filename = uploaded_file.name

column_number = st.number_input("Kiek faile yra stulpelių?", min_value = 1, value = 10)
skiped_rows = st.number_input("Kiek faile yra nereikalingų eilučių pradžioje?", min_value = 0, value = 0)

if not column_number:
  st.stop()

df_read = pd.read_csv(uploaded_file, delimiter=r"\s+|;|,", names=[str(i+1) for i in range(column_number)], skiprows = int(skiped_rows), header= None, engine="python")
st.write(df_read.head())

df = df_read
df = df.apply(pd.to_numeric, errors='coerce')
ilgis = len(df['1'])

c1 = st.text_input('Kuris stulpelis atitinka X ašį? (jei tokio stulpelio nėra, įrašyti 0): ', '0')
if not c1:
  st.stop()
if c1 == '0':
  step = st.number_input("Koks yra X ašies žingsnio dydis (*step size*)? (ns)", min_value = 0.00001, value = 0.08, format="%.4f")
  if not step:
    st.stop()
  df['0'] = np.arange(0, (ilgis+10)*step, step)[:ilgis]
c2 = st.text_input('Kuris stulpelis atitinka Y ašį? ', '1')
if not c2:
  st.stop()

fig = go.Figure()

i = np.argmax(df[c2])
start = df[c1][i]-5
end = df[c1][i]+20

fig.update_layout({
'font_size':16, 
})

fig = px.line(df[(df[c1]>start) & (df[c1]<end)], x=c1, y=c2, labels={c1: "t, ns", c2: "Intensity, a.u."}, color_discrete_sequence = ["#f6f8de"]) 
fig.update_xaxes(showgrid=True, gridcolor = "#464755")
fig.update_yaxes(showgrid=True, gridcolor = "#464755")
st.plotly_chart(fig)

fwhm = st.number_input('Koks naudojamo lazerio FWHM? (ns): ', min_value = 0.00001,  step=0.00001, value = 0.1, format="%.5f")
if not fwhm:
  st.stop()


left, right = st.columns(2)

if "step" not in st.session_state:
  st.session_state.step = "idle"

if left.button("Monoexponential fitting", width="stretch"):
  st.session_state.step = 'run_exp'

if right.button("Biexponential fitting", width="stretch"):
   st.session_state.step = 'run_bi'

if st.session_state.step == 'run_exp':
  st.subheader ('Monoexponential fitting')
  def exp_gauss(x, tau, H, x0, sigma):
    gauss = np.exp(-(x - x0)**2 / (2 * sigma**2))
    y = np.exp(-x/tau)

    final = np.convolve(gauss, y)[:len(x)]
    final = (final-min(final))/(max(final)-min(final))
    final = final*(1-H)+H

    return final

  df_fit = df[(df[c1]>start) & (df[c1]<end)]
  x = np.array(df_fit[c1]-min(df_fit[c1]))
  y = np.array(df_fit[c2]/max(df_fit[c2]))

  s = fwhm/(2*np.sqrt(2*np.log(2)))
  funkc = partial(exp_gauss, sigma = s)

  j = np.argmax(y)
  tau = 1 
  delta0 = tau / (1 + s/tau)
  h = np.argmax(np.array(y) != 0)
  bg = np.average(np.array(y)[h:h+10])

  sakinys = 'Duomenų fono lygis y ašyje lygus ' + str(round(bg, 4))+ ' ir smailės padėtis x ašyje yra ' +  str(round(x[j], 3))
  st.write('Duomenų **fono lygis** y ašyje lygus', round(bg, 4), ' ir **smailės padėtis** x ašyje yra', round(x[j], 3))
  st.write('**Nustatykite parametrų ribas fit funkcijai:**')

  data = {
    'parametras': ['lifetime (ns)', 'fono lygis y ašyje', 'smailės padėtis x ašyje'],
    'min': [0.1, bg-bg/10, x[j]-delta0],
    'max': [9.5, bg+bg/10, x[j]+2*delta0],
    'spėjimas': [1, bg, x[j]]
  }
  ribos = pd.DataFrame(data)

  eribos = st.data_editor(ribos)

  popt, pcov = curve_fit(funkc, x, y, bounds = (eribos['min'], eribos['max']), p0 = eribos['spėjimas'])
  

  paklaida = np.sqrt(np.diag(pcov))
  residuals = y - funkc(x, *popt)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((y-np.mean(y))**2)
  r_squared = 1 - (ss_res / ss_tot)

  st.write('**Rezultatai iš fit funkcijos:**')

  res = pd.DataFrame({
    'parametras': ['lifetime (ns)', 'fono lygis y ašyje', 'smailės padėtis x ašyje'],
    'nustatyta vertė': popt,
    'paklaida': paklaida
  })

  st.write(res)

  st.write('Fit funkcijos tikslumas R² = ', round(r_squared, 3))

  toggl = st.toggle("log scale")

  fig = go.Figure()

  fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=3, color="#f6f8de"), name="data"))
  fig.add_trace(go.Scatter(x=x, y=funkc(x, *popt), mode="lines", line=dict(color="#A1727C", width=2), name="fit"))

  fig.update_layout(xaxis_title="t, ns", yaxis_title="intensity, a.u.", yaxis_type="log" if toggl else "linear", template="plotly_dark")
  fig.update_xaxes(showgrid=True, gridcolor = "#464755")
  fig.update_yaxes(showgrid=True, gridcolor = "#464755")

  st.plotly_chart(fig, use_container_width=True)

if st.session_state.step == 'run_bi':
  def exp_gauss(x, tau1, a, tau2, H, x0, sigma):
    gauss = np.exp(-(x - x0)**2 / (2 * sigma**2))
    y = a*np.exp(-x/tau1) + (1-a)*np.exp(-x/tau2)

    final = np.convolve(gauss, y)[:len(x)]
    final = (final-min(final))/(max(final)-min(final))
    final = final*(1-H)+H

    return final

  df_fit = df[(df[c1]>start) & (df[c1]<end)]
  x = np.array(df_fit[c1]-min(df_fit[c1]))
  y = np.array(df_fit[c2]/max(df_fit[c2]))

  s = fwhm/(2*np.sqrt(2*np.log(2)))
  funkc = partial(exp_gauss, sigma = s)

  j = np.argmax(y)
  tau = 1 
  delta0 = tau / (1 + s/tau)
  h = np.argmax(np.array(y) != 0)
  bg = np.average(np.array(y)[h:h+10])

  st.subheader ('Biexponential fitting')
  st.write('Duomenų **fono lygis** y ašyje lygus', round(bg, 4), ' ir **smailės padėtis** x ašyje yra', round(x[j], 3))
  st.write('**Nustatykite parametrų ribas fit funkcijai:**')


  data = {
    'parametras': ['trumpesnis lifetime (ns)', 'trumpesnio A', 'ilgesnis lifetime (ns)',  'fono lygis y ašyje', 'smailės padėtis x ašyje'],
    'min': [0.1, 0, 1, bg-bg/10, x[j]-delta0],
    'max': [1.5, 1, 9.5, bg+bg/10, x[j]+2*delta0],
    'spėjimas': [0.15, 0.8, 2, bg, x[j]]
  }
  ribos = pd.DataFrame(data)
  eribos = st.data_editor(ribos)

  popt, pcov = curve_fit(funkc, x, y, bounds = (eribos['min'], eribos['max']), p0 = eribos['spėjimas'])
  st.write('**Rezultatai iš fit funkcijos:**')


  paklaida = np.sqrt(np.diag(pcov))
  residuals = y - funkc(x, *popt)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((y-np.mean(y))**2)
  r_squared = 1 - (ss_res / ss_tot)

  res = pd.DataFrame({
    'parametras': ['trumpesnis lifetime', 'trumpesnio A', 'ilgesnis lifetime',  'fono lygis y ašyje', 'smailės padėtis x ašyje'],
    'nustatyta vertė': popt,
    'paklaida': paklaida
  })

  st.write(res)
  st.write('Fit funkcijos tikslumas R² = ', round(r_squared, 3))


  toggl = st.toggle("log scale")

  fig = go.Figure()

  fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=3, color="#f6f8de"), name="data"))
  fig.add_trace(go.Scatter(x=x, y=funkc(x, *popt), mode="lines", line=dict(color="#A1727C", width=2), name="fit"))

  fig.update_layout(xaxis_title="t, ns", yaxis_title="intensity, a.u.", yaxis_type="log" if toggl else "linear", template="plotly_dark")
  fig.update_xaxes(showgrid=True, gridcolor = "#464755")
  fig.update_yaxes(showgrid=True, gridcolor = "#464755")

  st.plotly_chart(fig, use_container_width=True)



