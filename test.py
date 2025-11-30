import time
import numpy as np
import pandas as pd
import streamlit as st


st.title('testing')
map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [22.6, 120.4],
    columns=['lat', 'lon'])
st.map(map_data)