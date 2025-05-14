import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px 

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


# Load dataset
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path) 
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df['Indicator Name'] = df['Indicator Name'].astype(str)
        df.dropna(subset=['Year', 'Value'], inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()
    
# Function to convert DataFrame to CSV
@st.cache_data
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

#Load and filter data
df= load_data("preprocessed_dataset.csv")

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

# Filter dataset
df = df[df['Indicator Name'].isin(KEY_INDICATORS)].copy()

if df.empty:
    st.error("No data available after filtering for external debt indicators.")
    st.stop()

# Sidebar options
st.sidebar.title("Analysis Options")
chart_types = ["Line Chart", "Bar Chart", "Scatter Plot", "Box Plot", "Histogram", "Area Chart", "Statistics"]
analysis_choice = st.sidebar.radio("Select Chart Type:", chart_types)

# Indicator selection
selected_indicator = st.selectbox("Select an Indicator to Analyze", sorted(df['Indicator Name'].unique()))

# Filter data for selected indicator
indicator_df = df[df['Indicator Name'] == selected_indicator].sort_values('Year')

st.subheader(f"{analysis_choice} for: {selected_indicator}")

# KPI Metrics
if not indicator_df.empty:
    col1, col2, col3 = st.columns(3)
    col1.metric("🔺 Max Value", f"{indicator_df['Value'].max():,.2f}")
    col2.metric("🔻 Min Value", f"{indicator_df['Value'].min():,.2f}")
    col3.metric("📊 Average", f"{indicator_df['Value'].mean():,.2f}")

# Chart rendering
fig = None

if indicator_df.empty:
    st.warning("No data for the selected indicator.")
else:
    if analysis_choice == "Line Chart":
        fig = px.line(indicator_df, x='Year', y='Value', title='Trend Over Time',
                      markers=True, labels={'Value': 'Value', 'Year': 'Year'})
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_choice == "Bar Chart":
        fig = px.bar(indicator_df, x='Year', y='Value', title='Value Each Year',
                     labels={'Value': 'Value', 'Year': 'Year'}, color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig, use_container_width=True)


    elif analysis_choice == "Scatter Plot":
        fig = px.scatter(indicator_df, x='Year', y='Value', title='Scatter Plot',
                         labels={'Value': 'Value', 'Year': 'Year'})
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_choice == "Box Plot":
        fig = px.box(indicator_df, y='Value', title='Value Distribution',
                     labels={'Value': 'Value'})
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_choice == "Histogram":
        fig = px.histogram(indicator_df, x='Value', nbins=10, title='Value Frequency Distribution',
                           labels={'Value': 'Value'})
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_choice == "Area Chart":
        fig = px.area(indicator_df, x='Year', y='Value', title='Trend Over Time (Area)',
                      labels={'Value': 'Value', 'Year': 'Year'})
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_choice == "Statistics":
        st.write("Basic Statistics:")
        st.dataframe(indicator_df['Value'].describe().to_frame())
        st.write("All Data Points:")
        st.dataframe(indicator_df[['Year', 'Value', 'Indicator Code']].reset_index(drop=True).style.format({'Value': '{:,.2f}'}))

 # CSV Download
    st.markdown("---")
    csv_data = convert_df_to_csv(indicator_df[['Year', 'Indicator Name', 'Indicator Code', 'Value']])
    safe_name = "".join([c if c.isalnum() else "_" for c in selected_indicator])[:50]
    file_name = f"{safe_name}_data.csv"
    st.download_button("📥 Download Data as CSV", data=csv_data, file_name=file_name, mime='text/csv')

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("📉 Sri Lanka External Debt Dashboard")