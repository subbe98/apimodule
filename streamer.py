import time  # to simulate a real time data, time loop
import requests  # to receive data from the API
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import json  # to read the API response

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

# read csv from a github repo
api_url = "http://localhost:8000/predict"
response = requests.get(api_url)
predictions = json.loads(response.text)['Predictions']

# read csv from a URL
@st.experimental_memo
def get_data() -> pd.DataFrame:
    dataset = pd.read_csv("dataset_cleaned_V3.csv", index_col='DATETIME')
    return dataset

df = get_data()
df_2 = df.copy()
print(len(df))
print(len(predictions))

# dashboard title
st.title("Real-Time / Live Data Science Dashboard")

# creating a single-element container
placeholder = st.empty()

# near real-time / live feed simulation
for seconds in range(0, len(df), 60):
    df_2 = df.iloc[:seconds]

    with placeholder.container():
        
        st.markdown("### First Chart")
        fig = px.line(
            data_frame=df_2, y="PWEA-T"
        )
        st.write(fig)

        st.markdown("### Detailed Data View")
        st.dataframe(df_2)
        time.sleep(1)