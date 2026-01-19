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
    all_countries = sorted(df_world['country'].unique())
    default_countries = ["World"] if "World" in all_countries else [all_countries[0]]
    selected_countries = st.multiselect("Select Countries/Regions for Comparison", all_countries, default=default_countries)
    
    y_min, y_max = int(df_world['year'].min()), int(df_world['year'].max())
    year_range = st.slider("Select Temporal Range for Trend Analysis", y_min, y_max, (y_min, y_max))

    mask = (df_world['country'].isin(selected_countries)) & (df_world['year'].between(year_range[0], year_range[1]))
    df_line = df_world[mask]

    if not df_line.empty:
        fig_line = px.line(
            df_line, x="year", y="co2_per_capita", color="country", 
            line_dash="country", markers=True,
            color_discrete_sequence=px.colors.qualitative.Safe,
            template="plotly_white", 
            title=f"Evolution of Annual CO₂ Emissions per Capita({year_range[0]} - {year_range[1]})"
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
                        f"Annual CO₂ Emissions Per Capita ({selected_year})"),
        vertical_spacing=0.12
    )

    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], z=df_map_filtered["co2"],
        locationmode="ISO-3", colorscale="Viridis",
        colorbar=dict(
            title="Million Tonnes", 
            x=1.02,     
            y=0.78,     
            len=0.4,    
            thickness=15
        )
    ), row=1, col=1)

    fig_map.add_trace(go.Choropleth(
        locations=df_map_filtered["iso_code"], z=df_map_filtered["co2_per_capita"],
        locationmode="ISO-3", colorscale="Viridis",
        colorbar=dict(
            title="Tonnes/Capita", 
            x=1.02,     # Same horizontal position
            y=0.22,     # Centered on the bottom row
            len=0.4,    # Shorter length
            thickness=15
        )
    ), row=2, col=1)

    fig_map.update_layout(height=700, margin=dict(l=0, r=0, t=50, b=0),
                          geo=dict(projection_type="natural earth"),
                          geo2=dict(projection_type="natural earth"))
    st.plotly_chart(fig_map, use_container_width=True)

st.markdown("---")

# # --- LEFT: GEOSPATIAL SELECTOR (THE BRUSH) ---
# # --- LEFT: GEOSPATIAL SELECTOR (THE BRUSH) ---
# # --- LEFT: GEOSPATIAL SELECTOR (THE BRUSH) ---
# with row1_col1:
#     st.subheader("Geospatial Distribution Analysis")
#     available_years = sorted(df_clean['year'].unique(), reverse=True)
#     selected_year = st.selectbox("Select Year for Geospatial Mapping", available_years, key="map_year")
    
#     df_map = df_clean[df_clean['year'] == selected_year]

#     try:
#         # Using Plotly Express for the interactive map
#         fig_map = px.choropleth(
#             data_frame=df_map,
#             locations="iso_code",
#             z="co2_per_capita",
#             locationmode="ISO-3",
#             colorscale="Viridis",
#             labels={'co2_per_capita': 'Annual CO₂ per Capita (tonnes)'},
#             title=f"Global Annual CO₂ per Capita (tonnes) in {selected_year}",
#             hover_name="country"
#         )
        
#         fig_map.update_layout(height=550, margin=dict(l=0, r=0, t=50, b=0),
#                               geo=dict(projection_type="natural earth"))

#         # Captured click events act as the 'Brush' to add countries to the next plot
#         selected_point = plotly_events(fig_map, click_event=True, hover_event=False, key=f"map_events_{selected_year}")

#         if selected_point:
#             # Look up the country name based on the clicked index
#             clicked_index = selected_point[0]['pointNumber']
#             clicked_country = df_map.iloc[clicked_index]['country']
            
#             if clicked_country not in st.session_state.selected_countries:
#                 st.session_state.selected_countries.append(clicked_country)
#                 st.rerun()

#     except Exception as e:
#         st.error(f"Error building map: {e}")

# # --- RIGHT: LINKED TIME-SERIES ---
# with row1_col2:
#     st.subheader("Comparative Time-Series Analysis")
    
#     # CLEAR SELECTION BUTTON
#     if st.button("🗑️ Clear All Selections"):
#         st.session_state.selected_countries = ["World"]
#         st.rerun()
    
#     all_countries = sorted(df_clean['country'].unique())
    
#     # This multiselect is linked to map clicks via st.session_state
#     chosen = st.multiselect(
#         "Currently Selected for Comparison:", 
#         all_countries, 
#         default=st.session_state.selected_countries,
#         key="country_selector"
#     )
#     # Update memory if user adds/removes via the multiselect directly
#     st.session_state.selected_countries = chosen

#     y_min, y_max = int(df_clean['year'].min()), int(df_clean['year'].max())
#     year_range = st.slider("Select Temporal Range", y_min, y_max, (y_min, y_max))

#     mask = (df_clean['country'].isin(st.session_state.selected_countries)) & \
#            (df_clean['year'].between(year_range[0], year_range[1]))
#     df_line = df_clean[mask]

#     if not df_line.empty:
#         fig_line = px.line(
#             df_line, x="year", y="co2_per_capita", color="country", 
#             line_dash="country", markers=True,
#             color_discrete_sequence=px.colors.qualitative.Safe,
#             template="plotly_white", 
#             title=f"Trend: Annual CO₂ per Capita (tonnes) Over Time",
#             labels={'co2_per_capita': 'Annual CO₂ per Capita (tonnes)', 'year': 'Year'}
#         )
#         fig_line.update_layout(height=500, hovermode="x unified", legend=dict(orientation="h", y=-0.2))
#         st.plotly_chart(fig_line, use_container_width=True)
#     else:
#         st.info("💡 Click a country on the map to automatically add it to this comparison graph.")
# st.markdown("---")

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