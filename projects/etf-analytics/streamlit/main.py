import os
import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")

# os.environ['KAGGLE_USERNAME'] = st.secrets["kaggle_username"]
# os.environ['KAGGLE_KEY'] = st.secrets["kaggle_key"]

# from kaggle import api # import the already authenticated API client

# api.dataset_download_files(f'{st.secrets["kaggle_username"]}/etfdb-overview', path='data', unzip=True)

# df = pd.read_csv("data/etfdb.csv", index_col=0)

st.title("ETF Analytics")

from pathlib import Path

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

intro_markdown = read_markdown_file("projects/etf-analytics/README.md")

st.markdown(intro_markdown, unsafe_allow_html=True)
