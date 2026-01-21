import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Page Configuration 
st.set_page_config(page_title="Global CO2 Analysis & Prediction Dashboard", layout="wide")

# Data Loading 
@st.cache_data
def load_all_data():
    try:
        df_clean = pd.read_csv("df_clean.csv")
        df_world = pd.read_csv("df_world.csv")
        if 'country' in df_world.columns:
            df_world['country'] = df_world['country'].astype(str).str.strip()
        df_ml = pd.read_csv("model_results.csv")
        df_ml['Residuals'] = df_ml['Actual'] - df_ml['Predicted']
        return df_clean, df_world, df_ml
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df_clean, df_world, df_ml = load_all_data()

st.title("Global CO₂ Emissions: Historical Trends & Machine Learning Evaluation")
st.markdown("---")

# --- Part 1: Historical Emissions Analysis ---
st.header("Part 1: Historical Emissions Analysis")
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    st.subheader("Comparative Time-Series Analysis of National Emissions")
    
    # Using a container to fix the height of the widget area
    with st.container(height=230, border=False):
        st.caption("<br>", unsafe_allow_html=True)
        # st.caption("How have national emission levels evolved over the decades?") # Restored if needed
        
        all_countries = sorted(df_world['country'].unique())
        default_countries = ["World"] if "World" in all_countries else [all_countries[0]]
        selected_countries = st.multiselect("Select Countries/Regions for Comparison", all_countries, default=default_countries)
        
        y_min, y_max = int(df_world['year'].min()), int(df_world['year'].max())
        year_range = st.slider("Select Analysis Period", y_min, y_max, (y_min, y_max))

    mask = (df_world['country'].isin(selected_countries)) & (df_world['year'].between(year_range[0], year_range[1]))
    df_line = df_world[mask]

    if not df_line.empty:
        fig_line = px.line(
            df_line, x="year", y="co2_per_capita", color="country", 
            line_dash="country", 
            markers=True,
            color_discrete_sequence=px.colors.qualitative.D3, 
            template="plotly_white", 
            title=f"CO₂ per Capita Trend: {year_range[0]} – {year_range[1]}",
            labels={
                "year": "Year of Record",
                "co2_per_capita": "Tonnes per Person",
                "country": "Nation/Region"
            }
        )
        fig_line.update_layout(
            hovermode="x unified", 
            legend=dict(orientation="h", y=-0.2),
            yaxis_title="CO₂ (Metric Tonnes per Capita)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

with row1_col2:
    st.subheader("Global Carbon Footprint Map")
    
    # Matching the container height of the left column
    with st.container(height=230, border=False):
        st.caption("Hover over countries to see precise emission metrics.")
        available_years = sorted(df_clean['year'].unique(), reverse=True)
        selected_year = st.selectbox("View Data for Year:", available_years, key="map_year")
    
    df_map_filtered = df_clean[df_clean['year'] == selected_year]

    fig_map = make_subplots(
        rows=2, cols=1, 
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        subplot_titles=(f"Total National Emissions (Million Tonnes, {selected_year})", 
                        f"Emissions per Person (Tonnes/Capita, {selected_year})"),
        vertical_spacing=0.1
    )

    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], 
        z=df_map_filtered["co2"],
        locationmode="ISO-3", 
        colorscale="Viridis",
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar=dict(title="m/Tonnes", x=1.02, y=0.78, len=0.4, thickness=15)
    ), row=1, col=1)

    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], 
        z=df_map_filtered["co2_per_capita"],
        locationmode="ISO-3", 
        colorscale="Viridis",
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar=dict(title="t/Capita", x=1.02, y=0.22, len=0.4, thickness=15)
    ), row=2, col=1)

    fig_map.update_layout(
        height=750, 
        margin=dict(l=0, r=0, t=60, b=0),
        geo=dict(projection_type="natural earth", showframe=False),
        geo2=dict(projection_type="natural earth", showframe=False)
    )
    st.plotly