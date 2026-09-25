import streamlit as st

from utils.charts import models_per_year
from utils.data_loader import load_data

st.set_page_config(page_title="AI Model Evolution Explorer", layout="wide")
st.title("AI Model Evolution Explorer")

df = load_data()
st.caption(f"{len(df):,} notable models, {int(df['Year'].min())}–{int(df['Year'].max())}")
st.plotly_chart(models_per_year(df), use_container_width=True)

st.caption("Data: Epoch AI (CC-BY 4.0)")
