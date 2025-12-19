import pandas as pd
import streamlit as st

def title_block(title: str, subtitle: str | None = None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)

def df_download_button(df: pd.DataFrame, filename: str, label: str = "Download CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")
