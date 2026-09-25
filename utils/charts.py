"""All Plotly chart builders for the app live here."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def models_per_year(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("Year").size().reset_index(name="Models")
    return px.bar(counts, x="Year", y="Models", title="Notable AI models published per year")
