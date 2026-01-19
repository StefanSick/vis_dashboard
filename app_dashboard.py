import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Page Configuration ---
st.set_page_config(page_title="Global CO2 Dashboard", layout="wide")

# --- 2. Data Loading ---
@st.cache_data
def load_all_data():
    # Historical Data
    df_clean = pd.read_csv("df_clean.csv")
    # ML Results (the CSV you created earlier)
    df_ml = pd.read_csv("model_results.csv")
    df_ml['Residuals'] = df_ml['Actual'] - df_ml['Predicted']
    return df_clean, df_ml

# Replace these names with your actual filenames
df_clean, df_ml = load_all_data()

st.title("Global CO2 Emissions Analysis & Prediction")

# --- 3. Sidebar Global Filters ---
st.sidebar.header("Dashboard Controls")

# Tab Selection
app_mode = st.sidebar.radio("Navigate Dashboard", ["Global Trends & Maps", "Model Evaluation"])

# --- TAB 1: HISTORICAL DATA & MAPS ---
if app_mode == "Global Trends & Maps":
    
    tab_map, tab_line = st.tabs(["Global Maps", "Country Comparisons"])

    with tab_map:
        st.header("Global CO2 Distribution")
        available_years = sorted(df_clean['year'].unique(), reverse=True)
        selected_year = st.selectbox("Select Map Year", available_years)
        
        df_map = df_clean[df_clean['year'] == selected_year]

        fig_map = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
            subplot_titles=("Total CO2 Emissions", "CO2 Per Capita")
        )

        fig_map.add_trace(go.Choropleth(
            locations=df_map["iso_code"], z=df_map["co2"],
            locationmode="ISO-3", colorscale="Plasma",
            colorbar=dict(title="Mt CO2", x=0.45)
        ), row=1, col=1)

        fig_map.add_trace(go.Choropleth(
            locations=df_map["iso_code"], z=df_map["co2_per_capita"],
            locationmode="ISO-3", colorscale="Viridis",
            colorbar=dict(title="Tonnes/Capita", x=1.0)
        ), row=1, col=2)

        fig_map.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10),
                              geo=dict(projection_type="natural earth"),
                              geo2=dict(projection_type="natural earth"))
        
        st.plotly_chart(fig_map, width="container")

    with tab_line:
        st.header("Country Comparison Over Time")
        all_countries = sorted(df_clean['country'].unique())
        selected_countries = st.multiselect("Add/Remove Countries", all_countries, default=["World"])
        
        y_min, y_max = int(df_clean['year'].min()), int(df_clean['year'].max())
        year_range = st.slider("Select Year Range", y_min, y_max, (y_min, y_max))

        mask = (df_clean['country'].isin(selected_countries)) & (df_clean['year'].between(year_range[0], year_range[1]))
        df_line = df_clean[mask]

        if not df_line.empty:
            fig_line = px.line(df_line, x="year", y="co2", color="country", markers=True,
                               template="plotly_white", title="CO2 Emissions Trend")
            fig_line.update_layout(hovermode="x unified")
            st.plotly_chart(fig_line, width="container")

# --- TAB 2: ML MODEL EVALUATION ---
else:
    st.header("🤖 Random Forest Model Performance")
    
    # Continent Filter
    all_conts = sorted(df_ml['Continent'].unique())
    sel_conts = st.multiselect("Filter by Continent", all_conts, default=all_conts)
    df_ml_filtered = df_ml[df_ml['Continent'].isin(sel_conts)]

    if not df_ml_filtered.empty:
        # Metrics Row
        m1, m2 = st.columns(2)
        mae = mean_absolute_error(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
        r2 = r2_score(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
        m1.metric("MAE", f"{mae:.3f}")
        m2.metric("R² Score", f"{r2:.2f}")

        # Actual vs Predicted Plot
        st.subheader("Actual vs. Predicted")
        plt.clf()
        g = sns.JointGrid(data=df_ml_filtered, x='Actual', y='Predicted', hue='Continent', palette="colorblind", height=7)
        g.plot_joint(sns.scatterplot, alpha=0.5)
        g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
        
        # Identity line
        lims = [min(df_ml_filtered['Actual'].min(), df_ml_filtered['Predicted'].min()),
                max(df_ml_filtered['Actual'].max(), df_ml_filtered['Predicted'].max())]
        g.ax_joint.plot(lims, lims, 'k--', alpha=0.7, label='Ideal')
        st.pyplot(g.fig)

        # Residual Plot
        st.subheader("Residual Analysis")
        fig_res, ax_res = plt.subplots(figsize=(10, 4))
        sns.scatterplot(data=df_ml_filtered, x='Predicted', y='Residuals', hue='Continent', alpha=0.5, ax=ax_res)
        ax_res.axhline(0, color='black', linestyle='--')
        st.pyplot(fig_res)
    else:

        st.warning("Please select a continent.")
