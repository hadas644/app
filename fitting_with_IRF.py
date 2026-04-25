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
from scipy.interpolate import interp1d

st.title('Fitting with IRF')

uploaded_file = st.file_uploader("Įkelkite savo IRF failą:")

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
start = float(df[c1][i])-10
end = float(df[c1][i])+30

fig.update_layout({
'font_size':16, 
})

fig = px.line(df[(df[c1]>start) & (df[c1]<end)], x=c1, y=c2, labels={c1: "t, ns", c2: "Intensity, a.u."}, color_discrete_sequence = ["#f6f8de"]) 
fig.update_xaxes(showgrid=True, gridcolor = "#464755")
fig.update_yaxes(showgrid=True, gridcolor = "#464755")
st.plotly_chart(fig)

uploaded_file_data = st.file_uploader("Įkelkite savo duomenų failą:")

if uploaded_file_data is None:
  st.stop()

filename_data = uploaded_file_data.name

column_number_data = st.number_input("Kiek duomenų faile yra stulpelių?", min_value = 1, value = 10)
skiped_rows_data = st.number_input("Kiek duomenų faile yra nereikalingų eilučių pradžioje?", min_value = 0, value = 0)

if not column_number_data:
  st.stop()

df_read_data = pd.read_csv(uploaded_file_data, delimiter=r"\s+|;|,", names=[str(i+1) for i in range(column_number_data)], skiprows = int(skiped_rows_data), header= None, engine="python")
st.write(df_read_data.head())

df_data = df_read_data
df_data = df_data.apply(pd.to_numeric, errors='coerce')
ilgis = len(df_data['1'])

c1_data = st.text_input('Kuris duomenų stulpelis atitinka X ašį? (jei tokio stulpelio nėra, įrašyti 0): ', '0')
if not c1_data:
  st.stop()
if c1_data == '0':
  step_data = st.number_input("Koks yra duomenų X ašies žingsnio dydis (*step size*)? (ns)", min_value = 0.00001, value = 0.08, format="%.4f")
  if not step_data:
    st.stop()
  df_data['0'] = np.arange(0, (ilgis+10)*step_data, step_data)[:ilgis]
c2_data = st.text_input('Kuris duomenų stulpelis atitinka Y ašį? ', '1')
if not c2_data:
  st.stop()

fig = go.Figure()

i_d = np.argmax(df_data[c2_data])
start_d = df_data[c1_data][i_d]-10
end_d = df_data[c1_data][i_d]+30

fig.update_layout({
'font_size':16, 
})

fig = px.line(df_data[(df_data[c1_data]>start_d) & (df_data[c1_data]<end_d)], x=c1_data, y=c2_data, labels={c1_data: "t, ns", c2_data: "Intensity, a.u."}, color_discrete_sequence = ["#f6f8de"]) 
fig.update_xaxes(showgrid=True, gridcolor = "#464755")
fig.update_yaxes(showgrid=True, gridcolor = "#464755")
st.plotly_chart(fig)


crop_data = df_data[(df_data[c1_data]>start_d) & (df_data[c1_data]<end_d)]
data_x = np.array(crop_data[c1_data]-min(crop_data[c1_data]))
data_y = np.array(crop_data[c2_data]/max(crop_data[c2_data]))


crop = df[(df[c1]>start) & (df[c1]<end)]
irf_x = np.array(crop[c1]-min(crop[c1]))
irf_y = np.array(crop[c2]/max(crop[c2]))

irf_interp = interp1d(irf_x, irf_y)
irf_yn = irf_interp(data_x)
irf_yn = irf_yn/max(irf_yn)


left, right = st.columns(2)

if "step" not in st.session_state:
  st.session_state.step = "idle"

if left.button("Monoexponential fitting", width="stretch"):
  st.session_state.step = 'run_exp'

if right.button("Biexponential fitting", width="stretch"):
   st.session_state.step = 'run_bi'

if st.session_state.step == 'run_exp':
  st.subheader ('Monoexponential fitting')
  def exp(x, tau, H, x0, irf):    
    y = np.exp(-x/tau)
    m = int(round(x0/(x[1]-x[0])))

    final = np.convolve(irf, y)
    d = np.argmax(final)-m
    final = final[d:d+len(x)]

    if len(final) == 0:
        st.warning('data smailės padėtis yra per daug dešinėje - pabandyk praleisti daugiau duomenų failo eilučių (data smailė turėtų būti kairiau nei IRF smailė)')

    final = (final-min(final))/(max(final)-min(final))
    final = final*(1-H)+H

    return final

  funkc = partial(exp, irf = irf_yn)

  j = np.argmax(data_y)
  delta0 = data_x[1]-data_x[0]
  h = np.argmax(np.array(data_y) != 0)
  bg = np.average(np.array(data_y)[h:h+10])

  st.write('Duomenų **fono lygis** y ašyje lygus', round(bg, 4), ' ir **smailės padėtis** x ašyje yra', round(data_x[j], 3))
  st.write('**Nustatykite parametrų ribas fit funkcijai:**')

  data = {
    'parametras': ['lifetime (ns)', 'fono lygis y ašyje', 'smailės padėtis x ašyje'],
    'min': [float(0.1), bg-bg/10, data_x[j]-delta0],
    'max': [float(9.5), bg+bg/10, data_x[j]+2*delta0],
    'spėjimas': [float(1.0), bg, data_x[j]]
  }
  ribos = pd.DataFrame(data)

  eribos = st.data_editor(ribos)

  popt, pcov = curve_fit(funkc, data_x, data_y, bounds = (eribos['min'].to_numpy(), eribos['max'].to_numpy()), p0 = eribos['spėjimas'].to_numpy())
  

  paklaida = np.sqrt(np.diag(pcov))
  residuals = data_y - funkc(data_x, *popt)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((data_y-np.mean(data_y))**2)
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


  fig.add_trace(go.Scatter(x=data_x, y=data_y, mode="markers", marker=dict(size=3, color="#f6f8de"), name="data"))
  fig.add_trace(go.Scatter(x=data_x, y=funkc(data_x, *popt), mode="lines", line=dict(color="#A1727C", width=2), name="fit"))

  fig.update_layout(xaxis_title="t, ns", yaxis_title="intensity, a.u.", yaxis_type="log" if toggl else "linear", template="plotly_dark")
  fig.update_xaxes(showgrid=True, gridcolor = "#464755")
  fig.update_yaxes(showgrid=True, gridcolor = "#464755")

  st.plotly_chart(fig, use_container_width=True)

if st.session_state.step == 'run_bi':
  def biexp(x, tau1, a, tau2, H, x0, irf):
    y = a*np.exp(-x/tau1) + (1-a)*np.exp(-x/tau2)
   
    m = int(round(x0/(x[1]-x[0])))

    final = np.convolve(irf_yn, y)
    d = np.argmax(final)-m
    final = final[d:d+len(x)]

    if len(final) == 0:
        st.warning('data smailės padėtis yra per daug dešinėje - pabandyk praleisti daugiau duomenų failo eilučių (data smailė turėtų būti kairiau nei IRF smailė)')

    final = (final-min(final))/(max(final)-min(final))
    final = final*(1-H)+H

    return final

  funkc = partial(biexp, irf = irf_yn)

  j = np.argmax(data_y)
  delta0 = data_x[1]-data_x[0]
  h = np.argmax(np.array(data_y) != 0)
  bg = np.average(np.array(data_y)[h:h+10])

  st.subheader ('Biexponential fitting')
  st.write('Duomenų **fono lygis** y ašyje lygus', round(bg, 4), ' ir **smailės padėtis** x ašyje yra', round(data_x[j], 3))
  st.write('**Nustatykite parametrų ribas fit funkcijai:**')


  data = {
    'parametras': ['trumpesnis lifetime (ns)', 'trumpesnio A', 'ilgesnis lifetime (ns)',  'fono lygis y ašyje', 'smailės padėtis x ašyje'],
    'min': [0.1, 0, 1, bg-bg/10, data_x[j]-delta0],
    'max': [1.1, 1, 10.1, bg+bg/10, data_x[j]+2*delta0],
    'spėjimas': [0.15, 0.8, 2, bg, data_x[j]]
  }
  ribos = pd.DataFrame(data)
  eribos = st.data_editor(ribos)

  popt, pcov = curve_fit(funkc, data_x, data_y, bounds = (eribos['min'], eribos['max']), p0 = eribos['spėjimas'])
  st.write('**Rezultatai iš fit funkcijos:**')


  paklaida = np.sqrt(np.diag(pcov))
  residuals = data_y - funkc(data_x, *popt)
  ss_res = np.sum(residuals**2)
  ss_tot = np.sum((data_y-np.mean(data_y))**2)
  r_squared = 1 - (ss_res / ss_tot)

  res = pd.DataFrame({
    'parametras': ['trumpesnis lifetime (ns)', 'trumpesnio A', 'ilgesnis lifetime (ns)',  'fono lygis y ašyje', 'smailės padėtis x ašyje'],
    'nustatyta vertė': popt,
    'paklaida': paklaida
  })

  st.write(res)
  st.write('Fit funkcijos tikslumas R² = ', round(r_squared, 3))


  toggl = st.toggle("log scale")

  fig = go.Figure()

  fig.add_trace(go.Scatter(x=data_x, y=data_y, mode="markers", marker=dict(size=3, color="#f6f8de"), name="data"))
  fig.add_trace(go.Scatter(x=data_x, y=funkc(data_x, *popt), mode="lines", line=dict(color="#A1727C", width=2), name="fit"))

  fig.update_layout(xaxis_title="t, ns", yaxis_title="intensity, a.u.", yaxis_type="log" if toggl else "linear", template="plotly_dark")
  fig.update_xaxes(showgrid=True, gridcolor = "#464755")
  fig.update_yaxes(showgrid=True, gridcolor = "#464755")

  st.plotly_chart(fig, use_container_width=True)
