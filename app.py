import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("preprocessed_dataset.csv")

#set style
sns.set(style="whitegrid")

# configure streamlit page
st.set_page_config(page_title="Srilanka External Debt Dashboard", layout="wide")
st.title("Sri Lanka External Debt Analysis Dashboard")

# About section
with st.sidebar.markdown("ℹ About this Dashboard"):
    st.sidebar.info("""
    This interactive dashboard visualizes **Sri Lanka’s external debt indicators** using World Bank data.
    
    It is designed for analysts, economists, and the public to explore trends in external debt and understand the country’s financial obligations to international creditors.

    *Possible Indicators Covered:*
    - Total external debt stocks (current US$)
    - External debt to GDP (%)
    - Short-term and long-term debt compositions
    - External debt by type of creditor

    👇 Use the options below to select chart types and indicators.
    """)

KEY_INDICATORS = [
    "External debt stocks, total (DOD, current US$)",
    "External debt stocks (% of GNI)",
    "External debt stocks, long-term (DOD, current US$)",
    "External debt stocks, short-term (DOD, current US$)",
    "External debt stocks, public and publicly guaranteed (PPG) (DOD, current US$)",
    "External debt stocks, private nonguaranteed (PNG) (DOD, current US$)",
    "Use of IMF credit (DOD, current US$)",
    "Short-term debt (% of total external debt)",
    "Short-term debt (% of exports of goods, services and primary income)",
    "Short-term debt (% of total reserves)"
]

