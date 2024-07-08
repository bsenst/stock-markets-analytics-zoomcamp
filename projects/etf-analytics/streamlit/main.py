import os
import streamlit as st
import pandas as pd
from streamlit_timeline import timeline

st.set_page_config(layout="wide")

os.environ['KAGGLE_USERNAME'] = st.secrets["kaggle_username"]
os.environ['KAGGLE_KEY'] = st.secrets["kaggle_key"]

from kaggle import api # import the already authenticated API client

api.dataset_download_files(f'{st.secrets["kaggle_username"]}/etfdb-overview', path='data', unzip=True)
df = pd.read_csv("data/etfdb.csv", index_col=0)

# Convert 'Inception' column to datetime
df['Inception'] = pd.to_datetime(df['Inception'])

# Sort by 'Inception' date
df = df.sort_values(by='Inception')

# Prepare timeline data
events = []
for i, row in df.iterrows():
    events.append({
        'start_date': {'year': row['Inception'].year, 'month': row['Inception'].month, 'day': row['Inception'].day},
        'text': {'headline': row['Symbol'], 'text': f"Inception Date: {row['Inception'].strftime('%b %d, %Y')}"}
    })

timeline_data = {
    'title': {
        'text': {'headline': 'ETF Inception Dates'}
    },
    'events': events
}

# Create Streamlit app
# st.title('ETF Inception Dates Timeline')

# Display the timeline
timeline(timeline_data)
