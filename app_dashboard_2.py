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
        # Note: Ensure these files exist in your working directory
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

# --- PART 1: HISTORICAL ANALYSIS ---
st.header("Part 1: Historical Emissions Analysis")
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    st.subheader("Comparative Time-Series Analysis")
    
    # Using a container with a fixed height to align with the right column
    with st.container(height=170, border=False):
        all_countries = sorted(df_world['country'].unique())
        default_countries = ["World"] if "World" in all_countries else [all_countries[0]]
        selected_countries = st.multiselect("Select Countries/Regions", all_countries, default=default_countries)
        
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
            labels={"year": "Year", "co2_per_capita": "Tonnes/Person", "country": "Nation"}
        )
        fig_line.update_layout(
            hovermode="x unified", 
            legend=dict(orientation="h", y=-0.2),
            yaxis_title="CO₂ (Metric Tonnes per Capita)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

with row1_col2:
    st.subheader("Global Carbon Footprint Map")
    
    # Aligning height with the left column's container
    with st.container(height=170, border=False):
        available_years = sorted(df_clean['year'].unique(), reverse=True)
        selected_year = st.selectbox("View Data for Year:", available_years, key="map_year")
        st.caption("Hover over countries to see precise emission metrics.")
    
    df_map_filtered = df_clean[df_clean['year'] == selected_year]

    fig_map = make_subplots(
        rows=2, cols=1, 
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        subplot_titles=(f"Total National Emissions (m/Tonnes, {selected_year})", 
                        f"Emissions per Person (t/Capita, {selected_year})"),
        vertical_spacing=0.1
    )

    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], 
        z=df_map_filtered["co2"],
        locationmode="ISO-3", 
        colorscale="Viridis",
        marker_line_color='white', marker_line_width=0.5,
        colorbar=dict(title="m/Tonnes", x=1.02, y=0.78, len=0.4, thickness=15)
    ), row=1, col=1)

    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], 
        z=df_map_filtered["co2_per_capita"],
        locationmode="ISO-3", 
        colorscale="Viridis",
        marker_line_color='white', marker_line_width=0.5,
        colorbar=dict(title="t/Capita", x=1.02, y=0.22, len=0.4, thickness=15)
    ), row=2, col=1)

    fig_map.update_layout(
        height=750, 
        margin=dict(l=0, r=0, t=60, b=0),
        geo=dict(projection_type="natural earth", showframe=False),
        geo2=dict(projection_type="natural earth", showframe=False)
    )
    st.plotly_chart(fig_map, use_container_width=True)


# --- PART 2: MACHINE LEARNING ---
st.header("Part 2: Machine Learning Model Performance")

all_conts = sorted(df_ml['Continent'].unique())
sel_conts = st.multiselect("Filter Analysis by Continent", all_conts, default=all_conts)
df_ml_filtered = df_ml[df_ml['Continent'].isin(sel_conts)]

if not df_ml_filtered.empty:
    # Performance Summary
    m_col1, m_col2 = st.columns(2)
    mae = mean_absolute_error(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    r2 = r2_score(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    
    m_col1.metric("Average Prediction Error (MAE)", f"{mae:.3f} t")
    m_col2.metric("Model Reliability ($R^2$ Score)", f"{r2:.2%}")

    # Description moved to full width to keep the plots below aligned
    st.markdown("""
    **Analysis Overview:** The scatter plot (left) shows actual vs. predicted values with density marginals. 
    The residual plot (right) identifies if the model consistently over or under-predicts across different scales.
    """)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Model Accuracy & Distribution")
        
        g = sns.JointGrid(data=df_ml_filtered, x='Actual', y='Predicted', hue='Continent', palette="viridis", height=7)
        g.plot_joint(sns.scatterplot, alpha=0.6, s=70, edgecolor='white') 
        g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
        
        lims = [df_ml_filtered[['Actual', 'Predicted']].min().min(), df_ml_filtered[['Actual', 'Predicted']].max().max()]
        g.ax_joint.plot(lims, lims, color='red', linestyle='--', alpha=0.7, label='Perfect Prediction')
        g.ax_joint.set_title("Actual vs. Predicted", pad=25, fontweight='bold')
        g.ax_joint.legend(loc='lower right', title="Continent", fontsize='small')
        
        st.pyplot(g.fig, use_container_width=True)

    with row2_col2:
        st.subheader("Error (Residual) Analysis")
        
        fig_res, ax_res = plt.subplots(figsize=(7, 7))
        sns.scatterplot(data=df_ml_filtered, x='Predicted', y='Residuals', hue='Continent', palette="viridis", alpha=0.6, s=70, ax=ax_res)
        ax_res.axhline(0, color='red', linestyle='--', linewidth=2)
        ax_res.set_title("Residual Plot: Prediction Errors", pad=20, fontweight='bold')
        sns.move_legend(ax_res, "upper left", bbox_to_anchor=(1, 1))
        
        fig_res.tight_layout()
        st.pyplot(fig_res, use_container_width=True)

st.info("**Note:** Points closer to the red dashed line (left) indicate higher accuracy. Residuals (right) centered around 0 indicate unbiased predictions.")