from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import time

data_url = "https://www.dropbox.com/s/0gac54ifqvnqlrl/full_2020%20%281%29.csv?dl=1"
@st.cache(max_entries=10, ttl=3600)
def load_data(url):
    df = pd.read_csv(url)[:100]
    return df
df = load_data(data_url)
st.write(df.head())

