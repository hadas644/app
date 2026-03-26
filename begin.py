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


def page2():
  st.title('')

pg = st.navigation([st.Page("testforapp.py", title = 'Fitting without IRF'), st.Page(page2, title = 'Fitting with IRF')])
pg.run()