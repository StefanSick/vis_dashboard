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

# Data Loading 
@st.cache_data
def load_all_data():
    try:
        df_clean = pd.read_csv("df_clean.csv")
        df_ml = pd.read_csv("model_results.csv")
        df_ml['Residuals'] = df_ml['Actual'] - df_ml['Predicted']
        return df_clean, df_ml
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df_clean, df_ml = load_all_data()

# Initialize session state for country selections
if "selected_countries" not in st.session_state:
    st.session_state.selected_countries = ["Austria"]

st.title("Global CO₂ Emissions: Historical Trends & Machine Learning Evaluation")
st.markdown("---")

st.header("Part 1: Historical Emissions Analysis")
row1_col1, row1_col2 = st.columns([1, 1])

# --- LEFT: GEOSPATIAL SELECTOR (THE BRUSH) ---
with row1_col1:
# 1. Year Selection
    available_years = sorted(df_clean['year'].unique(), reverse=True)
    selected_year = st.selectbox("Select Year", available_years)
    df_map_filtered = df_clean[df_clean['year'] == selected_year]

    # 2. Build the figure using the "Working Style" (Graph Objects)
    fig_map = go.Figure(go.Choropleth(
        locations=df_map_filtered["iso_code"], 
        z=df_map_filtered["co2_per_capita"],
        locationmode="ISO-3", 
        colorscale="Viridis",
        text=df_map_filtered["country"], # For hover
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title="Tonnes/Capita",
    ))

    fig_map.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        )
    )

    # 3. The Interactive Wrapper
    selected_point = plotly_events(
        fig_map, 
        click_event=True, 
        key=f"map_{selected_year}", 
        override_height=600
    )

    # 4. Handle the Click
    if selected_point:
        # IMPORTANT: In go.Choropleth, the key is often 'location'
        clicked_iso = selected_point[0].get('location')
        
        if clicked_iso:
            match = df_clean[df_clean['iso_code'] == clicked_iso]['country'].unique()
            if len(match) > 0:
                clicked_country = match[0]
                if clicked_country not in st.session_state.selected_countries:
                    st.session_state.selected_countries.append(clicked_country)
                    st.rerun()
# --- RIGHT: LINKED TIME-SERIES ---
with row1_col2:
    st.subheader("Comparative Time-Series Analysis")
    
    if st.button("🗑️ Clear All Selections"):
        st.session_state.selected_countries = ["World"]
        st.rerun()
    
    all_countries = sorted(df_clean['country'].unique())
    
    # Linked multiselect
    chosen = st.multiselect(
        "Currently Selected Countries:", 
        options=all_countries, 
        default=st.session_state.selected_countries,
        key="country_selector"
    )
    st.session_state.selected_countries = chosen

    y_min, y_max = int(df_clean['year'].min()), int(df_clean['year'].max())
    year_range = st.slider("Select Temporal Range", y_min, y_max, (y_min, y_max))

    mask = (df_clean['country'].isin(st.session_state.selected_countries)) & \
           (df_clean['year'].between(year_range[0], year_range[1]))
    df_line = df_clean[mask]

    if not df_line.empty:
        fig_line = px.line(
            df_line, x="year", y="co2_per_capita", color="country", 
            markers=True,
            template="plotly_white", 
            title="Trend: Annual CO₂ per Capita Over Time",
            labels={'co2_per_capita': 'CO₂ per Capita', 'year': 'Year'}
        )
        fig_line.update_layout(height=500, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Click a country on the map to add it to this comparison.")
st.markdown("---")

st.header("Part 2: Machine Learning Model Performance")

all_conts = sorted(df_ml['Continent'].unique())
sel_conts = st.multiselect("Filter Analysis by Continent", all_conts, default=all_conts)
df_ml_filtered = df_ml[df_ml['Continent'].isin(sel_conts)]

if not df_ml_filtered.empty:
    # Top-level Metrics
    m_col1, m_col2 = st.columns(2)
    mae = mean_absolute_error(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    r2 = r2_score(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    m_col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
    m_col2.metric("Coefficient of Determination (R² Score)", f"{r2:.4f}")

    # The Grid Layout for Plots
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Model Accuracy Analysis")
        plt.clf()
        
        
        g = sns.JointGrid(
            data=df_ml_filtered, 
            x='Actual', 
            y='Predicted', 
            hue='Continent', 
            palette="colorblind", 
            height=7
        )
        
        g.plot_joint(sns.scatterplot, alpha=0.5, s=60, edgecolor='w') 
        g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
        
        # Perfect prediction identity line
        lims = [min(df_ml_filtered['Actual'].min(), df_ml_filtered['Predicted'].min()),
                max(df_ml_filtered['Actual'].max(), df_ml_filtered['Predicted'].max())]
        g.ax_joint.plot(lims, lims, 'k--', alpha=0.6, label='Perfect Prediction Identity Line')
        
        # Combine Continents and Line into one legend
        handles, labels = g.ax_joint.get_legend_handles_labels()
        g.ax_joint.legend(
            handles=handles, 
            labels=labels, 
            loc='lower right', 
            title="Continent / Reference", 
            fontsize='small',
            framealpha=0.8
        )
        
        # Descriptive Titles and Axis Labels
        g.ax_joint.set_title("Actual Observed vs. Model-Predicted CO₂ Values", pad=25, fontweight='bold')
        g.ax_joint.set_xlabel("Actual Observed CO₂ Emissions (Tonnes per Capita)")
        g.ax_joint.set_ylabel("Model-Predicted CO₂ Emissions (Tonnes per Capita)")
        
        st.pyplot(g.fig)

    with row2_col2:
        st.subheader("Residual Error Distribution Analysis")
        # figsize=(10, 8.5) compensates for the lack of marginal plots to match the height
        fig_res, ax_res = plt.subplots(figsize=(10, 8.5))
        
        sns.scatterplot(
            data=df_ml_filtered, 
            x='Predicted', 
            y='Residuals', 
            hue='Continent', 
            palette="colorblind", 
            alpha=0.5, 
            ax=ax_res
        )
        
        ax_res.axhline(0, color='black', linestyle='--', linewidth=2)
        
        # Move legend to lower left
        sns.move_legend(ax_res, "lower left", title="Continent", framealpha=0.7)
        
        # Descriptive Titles and Axis Labels
        ax_res.set_title("Residual Plot: Detecting Systematic Bias in Predictions", pad=20, fontweight='bold')
        ax_res.set_xlabel("Model-Predicted CO₂ Emissions (Tonnes per Capita)")
        ax_res.set_ylabel("Residual Deviation / Error (Tonnes per Capita)")
        
        st.pyplot(fig_res)
else:
    st.warning("Please select a continent to populate the model performance data.")