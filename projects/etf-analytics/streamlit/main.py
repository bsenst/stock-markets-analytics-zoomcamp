import streamlit as st
import pandas as pd
from streamlit_timeline import timeline

# Sample data
data = {
    'Symbol': ['SPY', 'IVV', 'VOO', 'VTI', 'QQQ', 'BITW', 'MAGX', 'ULTY', 'USDX', 'MAGQ'],
    'Inception': [
        'Jan 22, 1993', 'May 15, 2000', 'Sep 07, 2010', 'May 24, 2001', 
        'Mar 10, 1999', 'Nov 22, 2017', 'Feb 29, 2024', 'Feb 29, 2024', 
        'Feb 29, 2024', 'Feb 29, 2024'
    ]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

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
st.title('ETF Inception Dates Timeline')

# Display the timeline
timeline(timeline_data)
