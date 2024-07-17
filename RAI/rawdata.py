import streamlit as st
import pandas as pd

file_path = "C:\\Users\\sanka\\Downloads\\rawdata.xlsx"
raw_data = pd.read_excel(file_path)

raw_data['date'] = raw_data['date'].astype(str)
raw_data['datetime'] = pd.to_datetime(raw_data['date'] + ' ' + raw_data['time'].astype(str))

raw_data['position'] = raw_data['position'].str.lower()


raw_data['next_datetime'] = raw_data['datetime'].shift(-1)
raw_data['duration'] = (raw_data['next_datetime'] - raw_data['datetime']).dt.total_seconds()

duration_summary = raw_data.groupby(['date', 'position'])['duration'].sum().unstack(fill_value=0)

activity_count = raw_data.groupby(['date', 'activity']).size().unstack(fill_value=0)

st.title('Activity Duration and Count Summary')

st.header('Datewise Total Duration for Inside and Outside in seconds')
st.dataframe(duration_summary)

st.header('Datewise Number of Picking and Placing Activities in seconds')
st.dataframe(activity_count)
