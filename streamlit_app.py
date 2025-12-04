import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add 'Data product' to sys.path to import existing scripts if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "Data product"))

# Set page config
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for compact layout
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .element-container {
        margin-bottom: 0.5rem;
    }
    h1, h2, h3 {
        margin-top: 0;
        padding-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Bike Sharing Dashboard")

# Sidebar
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.header("Navigation")
    page = st.radio("Go to", ["Insight & Exploration", "Prediction & Decision"])

if page == "Insight & Exploration":
    st.header("Insight & Exploration")
    
    # Layout: Filters on top or left? app.R had them in a box.
    # Let's use columns.
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        weather_filter = st.multiselect(
            "Weather Condition:",
            options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Clear/Few Clouds",
                2: "Mist/Cloudy",
                3: "Light Rain/Snow",
                4: "Heavy Rain/Snow"
            }.get(x, str(x)),
            default=[1, 2, 3, 4]
        )
        
        user_type = st.radio(
            "User Type:",
            options=["cnt", "registered", "casual"],
            format_func=lambda x: {
                "cnt": "All Users",
                "registered": "Membership",
                "casual": "Non-Membership"
            }.get(x, x)
        )
        st.info("Note: Registered users are typically commuters, while Casual users are tourists.")

    with col2:
        # Placeholder for Heatmap/Scatter Plot
        st.subheader("Demand Visualization")
        # Toggle for Scatter Plot (Step 2)
        viz_mode = st.radio("View Mode", ["Heatmap", "Scatter Plot"], horizontal=True)
        
        # Load data (will implement caching later)
        @st.cache_data
        def load_data():
            df = pd.read_csv("Data product/bikehour.csv")
            return df
        
        df = load_data()
        
        # Filter data
        filtered_df = df[df['weathersit'].isin(weather_filter)]
        
        if viz_mode == "Heatmap":
            # Aggregate data
            weekday_map = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
            filtered_df['weekday_label'] = filtered_df['weekday'].map(weekday_map)
            
            agg_df = filtered_df.groupby(['weekday_label', 'hr'])[user_type].mean().reset_index()
            
            # Order weekdays
            weekday_order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            
            fig = px.density_heatmap(
                agg_df, 
                x="weekday_label", 
                y="hr", 
                z=user_type, 
                category_orders={"weekday_label": weekday_order},
                nbinsy=24,
                color_continuous_scale="Greens",
                title="Average Bike Rentals Heatmap"
            )
            fig.update_layout(yaxis=dict(autorange="reversed")) # 0 at top
            st.plotly_chart(fig, use_container_width=True)
            
        else: # Scatter Plot
            # Step 2: Scatter Plot with specific coloring
            # Rain points Blue (weathersit 3 or 4)
            # Wind/Sunny colors? 
            # Let's define a color map based on weathersit
            
            # Map weathersit to string for legend
            weather_map = {
                1: "Clear/Sunny",
                2: "Mist/Cloudy", 
                3: "Light Rain",
                4: "Heavy Rain"
            }
            filtered_df['Weather Desc'] = filtered_df['weathersit'].map(weather_map)
            
            # Define colors: Rain=Blue. Others?
            # 1 (Clear) -> Orange/Yellow?
            # 2 (Mist) -> Grey?
            color_discrete_map = {
                "Clear/Sunny": "orange",
                "Mist/Cloudy": "grey",
                "Light Rain": "blue",
                "Heavy Rain": "darkblue"
            }
            
            fig = px.scatter(
                filtered_df,
                x="hr",
                y=user_type,
                color="Weather Desc",
                color_discrete_map=color_discrete_map,
                title="Demand Scatter Plot",
                hover_data=["temp", "hum", "windspeed"]
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction & Decision":
    st.header("Prediction & Decision")
    
    # Layout: Main Content | Feature Importance (Step 3)
    main_col, right_col = st.columns([3, 1])
    
    with main_col:
        st.subheader("Scenario Simulation")
        
        col_inputs = st.columns(4)
        with col_inputs[0]:
            pred_date = st.date_input("Select Date", value=pd.to_datetime("2012-07-01"))
        with col_inputs[1]:
            pred_weather = st.selectbox(
                "Weather Forecast", 
                options=[1, 2, 3],
                format_func=lambda x: {1: "Clear/Few Clouds", 2: "Mist/Cloudy", 3: "Light Rain/Snow"}.get(x)
            )
        with col_inputs[2]:
            pred_temp = st.slider("Temperature (°C)", -10, 40, 25)
        with col_inputs[3]:
            run_pred = st.button("Run Prediction", type="primary")
            
        if run_pred:
            # Import prediction logic
            # For now, we'll need to adapt the existing script or inline the logic
            # I will assume I'll refactor linear-regression.py to be importable
            try:
                from linear_regression_model import get_predictions_for_streamlit
                
                results = get_predictions_for_streamlit(str(pred_date), pred_weather, pred_temp)
                
                # Plot
                fig = go.Figure()
                
                # CI
                fig.add_trace(go.Scatter(
                    x=results['Hour'].tolist() + results['Hour'].tolist()[::-1],
                    y=results['Upper_CI'].tolist() + results['Lower_CI'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(50, 205, 50, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='90% CI'
                ))
                
                # Main line
                fig.add_trace(go.Scatter(
                    x=results['Hour'],
                    y=results['Predicted_Demand'],
                    mode='lines+markers',
                    line=dict(color='#32CD32', width=3),
                    name='Predicted Demand'
                ))
                
                fig.update_layout(
                    title=f"Demand Prediction for {pred_date}",
                    xaxis_title="Hour (0-23)",
                    yaxis_title="Predicted Demand",
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Decision Card Logic
                max_demand = results['Predicted_Demand'].max()
                max_hour = results.loc[results['Predicted_Demand'].idxmax(), 'Hour']
                
                st.subheader("Manager's Decision Support")
                
                if max_demand > 900:
                    st.error(f"⚠️ **Capacity Breach Imminent!** Peak of {int(max_demand)} bikes at hour {max_hour}. Dispatch supply trucks at {max(0, max_hour-2)}:00.")
                elif max_demand > 700:
                    st.warning(f"⚡ **High Demand Expected.** Peak: {int(max_demand)} bikes. Enable Surge Pricing.")
                elif max_demand < 200:
                    st.info(f"🏷️ **Low Demand Expected.** Peak: {int(max_demand)} bikes. Push Coupons.")
                else:
                    st.success(f"✅ **Normal Operations.** Demand is within normal range.")
                    
                # Step 4: Text Prediction & Advice
                st.subheader("AI Assistant Advice")
                
                # Current Status (Mock logic: compare current hour prediction to threshold)
                # In a real app, "Current" would be now(). Here we use the simulation context? 
                # Or just summarize the whole day.
                # Let's assume "Current" means the peak for the summary, or we can pick a specific hour.
                # The prompt says: "Current Status: Explicitly state if it is currently 'Busy' or 'Free'."
                # I'll use the peak as the reference for "Busy" status for the day.
                
                status = "Busy" if max_demand > 500 else "Free"
                st.markdown(f"**Current Status**: The system is expected to be **{status}**.")
                
                # Actionable Advice: Next drop in demand
                # Find the first hour after peak where demand drops below a threshold (e.g., 70% of peak)
                threshold = max_demand * 0.7
                off_peak_hours = results[(results['Hour'] > max_hour) & (results['Predicted_Demand'] < threshold)]
                
                if not off_peak_hours.empty:
                    next_off_peak = off_peak_hours.iloc[0]['Hour']
                    st.markdown(f"**Actionable Advice**: Predicted peak ends around {max_hour}:00. Suggest traveling after **{next_off_peak}:00** for an 'Off-Peak' experience.")
                else:
                    st.markdown(f"**Actionable Advice**: Demand stays high after peak. Consider traveling early morning.")

            except ImportError:
                st.error("Backend model not found. Please ensure 'linear_regression_model.py' is set up.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    with right_col:
        # Step 3: Feature Importance
        st.subheader("Feature Importance")
        
        # Load feature importance data
        # Assuming we can read the Excel file or compute it
        try:
            fi_df = pd.read_excel("Feature importance.xlsx")
            # Ensure columns are correct. If not, fallback to mock or recompute
            if 'Feature' in fi_df.columns and 'Importance' in fi_df.columns:
                fig_fi = px.bar(
                    fi_df.sort_values('Importance', ascending=True),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top Influencing Factors"
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.warning("Feature importance data format incorrect.")
        except Exception:
            st.info("Loading feature importance from model...")
            # Fallback: Load from the python script if excel fails
            # For now, placeholder
            st.write("Feature importance data not available.")

