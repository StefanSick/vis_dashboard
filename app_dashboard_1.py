import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from streamlit_plotly_events import plotly_events

# Page Configuration 
st.set_page_config(page_title="Global CO2 Analysis & Prediction Dashboard", layout="wide")

#  Data Loading 
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


st.header("Part 1: Historical Emissions Analysis")
row1_col1, row1_col2 = st.columns([1, 1])



with row1_col1:
    st.subheader("Comparative Time-Series Analysis of National Emissions")
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
        # Using 'D3' or 'Bold' which provides more distinct color separation than 'Safe'
        fig_line = px.line(
            df_line, x="year", y="co2_per_capita", color="country", 
            line_dash="country", # Keeps lines distinct even if colors look similar
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

    # Note: 'Viridis' is already the best-practice choice for colorblind accessibility.
    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], 
        z=df_map_filtered["co2"],
        locationmode="ISO-3", 
        colorscale="Viridis",
        marker_line_color='white', # Adds borders to countries for better visibility
        marker_line_width=0.5,
        colorbar=dict(
            title="m/Tonnes", 
            x=1.02, y=0.78, len=0.4, thickness=15
        )
    ), row=1, col=1)

    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], 
        z=df_map_filtered["co2_per_capita"],
        locationmode="ISO-3", 
        colorscale="Viridis",
        marker_line_color='white',
        marker_line_width=0.5,
        colorbar=dict(
            title="t/Capita", 
            x=1.02, y=0.22, len=0.4, thickness=15
        )
    ), row=2, col=1)

    fig_map.update_layout(
        height=750, 
        margin=dict(l=0, r=0, t=60, b=0),
        geo=dict(projection_type="natural earth", showframe=False),
        geo2=dict(projection_type="natural earth", showframe=False)
    )
    st.plotly_chart(fig_map, use_container_width=True)

st.header("Part 2: Machine Learning Model Performance")

# --- Metric Definitions ---
# $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
# $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$

all_conts = sorted(df_ml['Continent'].unique())
sel_conts = st.multiselect("Filter Analysis by Continent", all_conts, default=all_conts)
df_ml_filtered = df_ml[df_ml['Continent'].isin(sel_conts)]

if not df_ml_filtered.empty:
    # 1. Top-level Performance Summary
    m_col1, m_col2 = st.columns(2)
    mae = mean_absolute_error(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    r2 = r2_score(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    
    m_col1.metric("Average Prediction Error (MAE)", f"{mae:.3f} t", 
                  help="Mean Absolute Error: On average, how many tonnes the prediction is off.")
    m_col2.metric("Model Reliability ($R^2$ Score)", f"{r2:.2%}", 
                  help="How much of the CO2 variation is captured by the model (100% is perfect).")

    # 2. Visual Analysis Layout
    row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.subheader("Model Accuracy & Data Distribution")
    
    # Short description explaining the marginal plots
    st.markdown("""
    The **density plots on the top and right** (marginal distributions) show where the majority of the data points are concentrated.
    """)
    
    # Initialize the JointGrid
    g = sns.JointGrid(
        data=df_ml_filtered, 
        x='Actual', 
        y='Predicted', 
        hue='Continent', 
        palette="viridis", 
        height=7
    )
    
    # 1. Main Scatter Plot (The "Joint" part)
    g.plot_joint(sns.scatterplot, alpha=0.6, s=70, edgecolor='white') 
    
    # 2. Marginal Plots (The "Distribution" part)
    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
    
    # 3. Identity Line (Ideal Model)
    lims = [df_ml_filtered[['Actual', 'Predicted']].min().min(),
            df_ml_filtered[['Actual', 'Predicted']].max().max()]
    g.ax_joint.plot(lims, lims, color='red', linestyle='--', alpha=0.7, label='Perfect Prediction ($y=x$)')
    
    # Labels & Title
    g.ax_joint.set_title("Actual vs. Predicted CO₂ Emissions", pad=25, fontweight='bold')
    g.ax_joint.set_xlabel("Measured Actual (Tonnes/Capita)")
    g.ax_joint.set_ylabel("RF Model Predicted (Tonnes/Capita)")

    # 4. Legend Position (Bottom Right)
    # We target the joint axis to place the legend inside the plot area
    g.ax_joint.legend(loc='lower right', title="Continent", fontsize='small', frameon=True)
    
    st.pyplot(g.fig, use_container_width=True)

with row2_col2:
    st.subheader("Error (Residual) Analysis")
    st.write("Are there specific continents where the model struggles?")
    #st.markdown("<br><br>", unsafe_allow_html=True)
    fig_res, ax_res = plt.subplots(figsize=(7, 7))
    
    sns.scatterplot(
        data=df_ml_filtered, 
        x='Predicted', 
        y='Residuals', 
        hue='Continent', 
        palette="viridis", 
        alpha=0.6, 
        s=70,
        marker='o',
        ax=ax_res
    )
    
    # Zero Error Line
    ax_res.axhline(0, color='red', linestyle='--', linewidth=2)
    
    # Descriptive Labels
    ax_res.set_title("Residual Plot: Prediction Errors by Scale", pad=20, fontweight='bold')
    ax_res.set_xlabel("Predicted CO₂ (Tonnes per Capita)")
    ax_res.set_ylabel("Residual Error (Actual - Predicted)")
    
    # Move legend to be more readable
    sns.move_legend(ax_res, "upper left", bbox_to_anchor=(1, 1), title="Continent")
    
    fig_res.tight_layout()
    st.pyplot(fig_res, use_container_width=True)

# Descriptive Footnote
st.info("""
**Understanding the Charts:**
* **The Diagonal Line:** Points closer to the red dashed line indicate higher accuracy. 
* **The Residual Cloud:** Points above 0 represent under-predictions; points below 0 represent over-predictions. 
* **Color Contrast:** We use the 'Viridis' palette, which is designed to be readable in grayscale and for all common types of colorblindness.
""")
