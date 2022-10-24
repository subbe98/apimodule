import time  # to simulate a real time data, time loop
import requests  # to receive data from the API
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import json  # to read the API response
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="📈",
    layout="wide",
)

# read csv from a github repo
api_url = "https://apimodule.herokuapp.com/predict"
response = requests.get(api_url)
data = json.loads(response.text)

# read csv from a URL
@st.experimental_memo
def get_data() -> pd.DataFrame:
    dataset = pd.read_csv("dataset_cleaned_V3.csv", index_col='DATETIME')
    return dataset

df = get_data()
df['PWEA-T predictions'] = data['predictions']
df_2 = df.copy()

# dashboard title
st.title("Real-Time / Live Data Science Dashboard")

selected = st.selectbox('Choose model', np.array(['XGB', 'Linear', 'FFNN', 'LSTM']))

# creating a single-element container
placeholder = st.empty()

old_mse = 0
old_rmse = 0
old_r2 = 0

# near real-time / live feed simulation
for seconds in range(1, len(df), 60):
    df_2 = df.iloc[:seconds]

    new_mse = mean_squared_error(df_2['PWEA-T'], df_2['PWEA-T predictions'])
    new_rmse = mean_squared_error(df_2['PWEA-T'], df_2['PWEA-T predictions'], squared=False)
    new_r2 = r2_score(df_2['PWEA-T'], df_2['PWEA-T predictions'])

    with placeholder.container():
        
        st.header(f"Visualization using {selected} model")

        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)
        

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="MSE",
            value=new_mse,
            delta=new_mse-old_mse,
        )
        
        kpi2.metric(
            label="RMSE",
            value=new_rmse,
            delta=new_rmse-old_rmse,
        )
        
        kpi3.metric(
            label="R^2",
            value=new_r2,
            delta=-new_r2-old_r2,
        )

        old_mse = new_mse
        old_rmse = new_rmse
        old_r2 = new_r2

        fig = px.line(
            data_frame=df_2, y=["PWEA-T", "PWEA-T predictions"]
        )
        st.write(fig)
        

        st.markdown("### Detailed Data Viewer")
        st.dataframe(df_2)
        time.sleep(1)