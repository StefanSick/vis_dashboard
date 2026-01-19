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
        df_clean = pd.read_csv("df_clean.csv")
        df_ml = pd.read_csv("model_results.csv")
        df_ml['Residuals'] = df_ml['Actual'] - df_ml['Predicted']
        return df_clean, df_ml
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df_clean, df_ml = load_all_data()

st.title("Global CO₂ Emissions: Historical Trends & Machine Learning Evaluation")
st.markdown("---")

# ==========================================
# ROW 1: HISTORICAL EMISSIONS ANALYSIS
# ==========================================
st.header("Part 1: Historical Emissions Analysis")
row1_col1, row1_col2 = st.columns([1, 1])

with row1_col1:
    st.subheader("Comparative Time-Series Analysis of National Emissions")
    all_countries = sorted(df_clean['country'].unique())
    default_countries = ["World"] if "World" in all_countries else [all_countries[0]]
    selected_countries = st.multiselect("Select Countries/Regions for Comparison", all_countries, default=default_countries)
    
    y_min, y_max = int(df_clean['year'].min()), int(df_clean['year'].max())
    year_range = st.slider("Select Temporal Range for Trend Analysis", y_min, y_max, (y_min, y_max))

    mask = (df_clean['country'].isin(selected_countries)) & (df_clean['year'].between(year_range[0], year_range[1]))
    df_line = df_clean[mask]

    if not df_line.empty:
        fig_line = px.line(
            df_line, x="year", y="co2", color="country", 
            line_dash="country", markers=True,
            color_discrete_sequence=px.colors.qualitative.Safe,
            template="plotly_white", 
            title=f"Evolution of Annual CO₂ Emissions ({year_range[0]} - {year_range[1]})"
        )
        fig_line.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_line, use_container_width=True)

with row1_col2:
    st.subheader("Geospatial Distribution of CO₂ Emissions")
    available_years = sorted(df_clean['year'].unique(), reverse=True)
    selected_year = st.selectbox("Select Map Year", available_years, key="map_year")
    
    df_map_filtered = df_clean[df_clean['year'] == selected_year]

    fig_map = make_subplots(
        rows=2, cols=1, 
        specs=[[{"type": "choropleth"}], [{"type": "choropleth"}]],
        subplot_titles=(f"Total Annual CO₂ Emissions by Country ({selected_year})", 
                        f"CO₂ Emissions Per Capita (Tonnes per Person, {selected_year})"),
        vertical_spacing=0.12
    )

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
# ROW 2: MACHINE LEARNING MODEL EVALUATIONS
# ==========================================
st.header("Part 2: Machine Learning Model Performance")

# all_conts = sorted(df_ml['Continent'].unique())
# sel_conts = st.multiselect("Filter Regression Analysis by Continent", all_conts, default=all_conts)
# df_ml_filtered = df_ml[df_ml['Continent'].isin(sel_conts)]

# if not df_ml_filtered.empty:
#     mae = mean_absolute_error(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
#     r2 = r2_score(df_ml_filtered['Actual'], df_ml_filtered['Predicted'])
    
#     m_col1, m_col2 = st.columns(2)
#     m_col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
#     m_col2.metric("Coefficient of Determination (R² Score)", f"{r2:.4f}")

#     row2_col1, row2_col2 = st.columns(2)

#     # with row2_col1:
#     #     st.subheader("Model Accuracy Analysis (Tonnes per Capita)")
#     #     plt.clf()
#     #     # JointGrid doesn't accept legend location directly in plot_joint, 
#     #     # so we disable the default legend and add it to the ax_joint
#     #     g = sns.JointGrid(data=df_ml_filtered, x='Actual Observation', y='Predicted Values', hue='Continent', palette="colorblind", height=7)
#     #     g.plot_joint(sns.scatterplot, alpha=0.5, s=60, edgecolor='w', legend=False) 
#     #     g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
        
#     #     # Perfect prediction identity line
#     #     lims = [min(df_ml_filtered['Actual'].min(), df_ml_filtered['Predicted'].min()),
#     #             max(df_ml_filtered['Actual'].max(), df_ml_filtered['Predicted'].max())]
#     #     g.ax_joint.plot(lims, lims, 'k--', alpha=0.6, label='Perfect Prediction Identity Line')
        
#     #     # MANUALLY ADD LEGEND TO LOWER RIGHT
#     #     g.ax_joint.legend(loc='lower right', title="Continent", fontsize='small')
        
#     #     g.ax_joint.set_title("Actual Observed vs. Model-Predicted CO₂ Values", pad=25)
#     #     st.pyplot(g.fig)

#     with row2_col1:
#     # Title focuses on the purpose of the plot
#         st.subheader("Model Accuracy Analysis")
#         plt.clf()
        
#         # 1. Initialize JointGrid 
#         # (Note: data= uses the column names, but we set descriptive labels below)
#         g = sns.JointGrid(
#             data=df_ml_filtered, 
#             x='Actual', 
#             y='Predicted', 
#             hue='Continent', 
#             palette="colorblind", 
#             height=7
#         )
        
#         # 2. Plot the scatter points (legend=False so we can build it manually)
#         g.plot_joint(sns.scatterplot, alpha=0.5, s=60, edgecolor='w', legend=False) 
#         g.plot_marginals(sns.kdeplot, fill=True, alpha=0.3)
        
#         # 3. Add the Identity Line
#         lims = [min(df_ml_filtered['Actual'].min(), df_ml_filtered['Predicted'].min()),
#                 max(df_ml_filtered['Actual'].max(), df_ml_filtered['Predicted'].max())]
#         line_label = 'Perfect Prediction Identity Line'
#         g.ax_joint.plot(lims, lims, 'k--', alpha=0.6, label=line_label)
        
#         # --- FIX: COMBINE CONTINENTS AND REFERENCE LINE IN LEGEND ---
#         # We collect all handles (icons) and labels (text) currently on the joint axis
#         handles, labels = g.ax_joint.get_legend_handles_labels()
        
#         # We place the combined legend in the lower right
#         g.ax_joint.legend(
#             handles=handles, 
#             labels=labels, 
#             loc='lower right', 
#             title="Continent / Reference", 
#             fontsize='small',
#             framealpha=0.7
#         )
        
#         # 4. SET DESCRIPTIVE AXIS LABELS WITH UNITS
#         g.ax_joint.set_title("Actual Observed vs. Model-Predicted CO₂ Values", pad=25)
#         g.ax_joint.set_xlabel("Actual Observed CO₂ Emissions (Tonnes per Capita)")
#         g.ax_joint.set_ylabel("Model-Predicted CO₂ Emissions (Tonnes per Capita)")
        
#         st.pyplot(g.fig)
            


#     with row2_col2:
#         st.subheader("Residual Error Distribution Analysis")
#         fig_res, ax_res = plt.subplots(figsize=(10, 7.5))
#         sns.scatterplot(data=df_ml_filtered, x='Predicted', y='Residuals', hue='Continent', palette="colorblind", alpha=0.5, ax=ax_res)
        
#         ax_res.axhline(0, color='black', linestyle='--', linewidth=2)
        
#         # MANUALLY MOVE LEGEND TO LOWER LEFT
#         sns.move_legend(ax_res, "lower left", title="Continent")
        
#         ax_res.set_title("Residual Plot: Detecting Systematic Bias and Variance in Predictions", pad=15)
#         ax_res.set_xlabel("Predicted CO₂ Emissions (Tonnes Per Capita)")
#         ax_res.set_ylabel("Residual Deviation (Actual - Predicted)")
#         st.pyplot(fig_res)
# else:
#     st.warning("No continents selected. Please update the filters above.")
st.header("🤖 Part 2: Machine Learning Model Performance")

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
        
        # height=7 is the specific size for JointGrid
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