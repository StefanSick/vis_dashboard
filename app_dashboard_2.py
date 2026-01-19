import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Page Configuration ---
st.set_page_config(page_title="Global CO2 Analysis & Prediction Dashboard", layout="wide")

# --- 2. Data Loading ---
@st.cache_data
def load_all_data():
    try:
        # Ensure these match your GitHub filenames exactly
        df_clean = pd.read_csv("df_clean.csv")
        df_ml = pd.read_csv("model_results.csv")
        df_ml['Residuals'] = df_ml['Actual'] - df_ml['Predicted']
        return df_clean, df_ml
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df_clean, df_ml = load_all_data()

st.title("🌍 Global CO₂ Emissions: Historical Trends & Machine Learning Evaluation")
st.markdown("---")

# ==========================================
# LINE 1: HISTORICAL EMISSIONS ANALYSIS
# ==========================================
st.header("📊 Part 1: Historical Emissions Analysis")
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    st.subheader("Comparative Time-Series Analysis")
    all_countries = sorted(df_clean['country'].unique())
    # Safe default check
    default_countries = ["World"] if "World" in all_countries else [all_countries[0]]
    selected_countries = st.multiselect("Select Countries/Regions", all_countries, default=default_countries)
    
    y_min, y_max = int(df_clean['year'].min()), int(df_clean['year'].max())
    year_range = st.slider("Select Temporal Range", y_min, y_max, (y_min, y_max))

    mask = (df_clean['country'].isin(selected_countries)) & (df_clean['year'].between(year_range[0], year_range[1]))
    df_line = df_clean[mask]

    if not df_line.empty:
        # Using 'Safe' qualitative palette and line dashes for colorblind accessibility
        fig_line = px.line(
            df_line, x="year", y="co2", color="country", 
            line_dash="country", markers=True,
            color_discrete_sequence=px.colors.qualitative.Safe,
            template="plotly_white", 
            title=f"Annual CO₂ Emissions Trend ({year_range[0]} - {year_range[1]})"
        )
        fig_line.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_line, use_container_width=True)

with row1_col2:
    st.subheader("Geospatial Distribution")
    available_years = sorted(df_clean['year'].unique(), reverse=True)
    selected_year = st.selectbox("Select Year for Map View", available_years, key="map_year")
    
    df_map_filtered = df_clean[df_clean['year'] == selected_year]

    # Stacking maps vertically within this column
    fig_map = make_subplots(
        rows=2, cols=1, 
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        subplot_titles=(f"Total CO₂ Emissions ({selected_year})", 
                        f"CO₂ Per Capita (Tonnes, {selected_year})"),
        vertical_spacing=0.12
    )

    # Using 'Viridis' for colorblind-safe sequential scales
    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], z=df_map_filtered["co2"],
        locationmode="ISO-3", colorscale="Viridis",
        colorbar=dict(title="Million Tonnes", x=1.0)
    ), row=1, col=1)

    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], z=df_map_filtered["co2_per_capita"],
        locationmode="ISO-3", colorscale="Viridis",
        colorbar=dict(title="Tonnes/Capita", x=1.0)
    ), row=2, col=1)

    fig_map.update_layout(height=700, margin=dict(l=0, r=0, t=50, b=0),
                          geo=dict(projection_type="natural earth"),
                          geo2=dict(projection_type="natural earth"))
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# ==========================================
# LINE 2: MACHINE LEARNING MODEL EVALUATIONS
# ==========================================
st.header("🤖 Part 2: Machine Learning Model Performance")

# Global Filter for ML section
all_conts = sorted(df_ml['Continent'].unique())
sel_conts = st.multiselect("Filter Analysis by Continent", all_conts, default=all_conts)
df_ml_filtered = df_ml[df_ml['Continent'].isin(sel_conts)]

if not df_ml_filtered.empty:
    # Top-level metrics for Row 2
    mae = mean_absolute_error(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    r2 = r2_score(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
    m_col2.metric("R² Score (Coefficient of Determination)", f"{r2:.4f}")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Regression Accuracy Analysis")
        plt.clf()
        # Using palette='colorblind' for accessibility
        g = sns.JointGrid(data=df_ml_filtered, x='Actual', y='Predicted', hue='Continent', palette="colorblind", height=7)
        g.plot_joint(sns.scatterplot, alpha=0.5, s=60, edgecolor='w')
        g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
        
        lims = [min(df_ml_filtered['Actual'].min(), df_ml_filtered['Predicted'].min()),
                max(df_ml_filtered['Actual'].max(), df_ml_filtered['Predicted'].max())]
        g.ax_joint.plot(lims, lims, 'k--', alpha=0.7, label='Perfect Prediction')
        g.ax_joint.set_title("Observed vs. Predicted CO₂ Values", pad=20)
        st.pyplot(g.fig)

    with row2_col2:
        st.subheader("Residual Error Distribution")
        fig_res, ax_res = plt.subplots(figsize=(10, 7))
        sns.scatterplot(data=df_ml_filtered, x='Predicted', y='Residuals', hue='Continent', palette="colorblind", alpha=0.5, ax=ax_res)
        ax_res.axhline(0, color='black', linestyle='--', linewidth=2)
        ax_res.set_title("Residual Plot: Detecting Systematic Bias in Predictions")
        ax_res.set_xlabel("Predicted CO₂ Emissions")
        ax_res.set_ylabel("Residual Error (Actual - Predicted)")
        st.pyplot(fig_res)
else:
    st.warning("Please select at least one continent to view results.")