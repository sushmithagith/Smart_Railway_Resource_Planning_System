import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta

# Configuration
st.set_page_config(
    page_title="Smart Railway Resource Planning",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb;
    }
    .metric-label {
        font-size: 1rem;
        color: #6b7280;
    }
    .warning {
        color: #dc2626;
        font-weight: bold;
    }
    .success {
        color: #16a34a;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None
        
    # Standardize columns (basic cleaning)
    df['Date and Time'] = pd.to_datetime(df['Date and Time'], errors='coerce')
    
    # Calculate Capacity & Occupancy if not fully present but base metrics exist
    if 'Total Capacity' not in df.columns and 'Number of Coaches' in df.columns:
        df['Total Capacity'] = df['Number of Coaches'] * 60
    
    if 'Seat Occupancy (%)' not in df.columns and 'Passenger Count' in df.columns and 'Total Capacity' in df.columns:
        df['Seat Occupancy (%)'] = (df['Passenger Count'] / df['Total Capacity']) * 100
    
    return df

def main():
    st.title("🚆 Smart Railway Resource Planning System")
    st.markdown("A data-driven dashboard for train resource allocation and demand forecasting.")
    
    # Sidebar
    st.sidebar.header("Data Input & Settings")
    
    uploaded_file = st.sidebar.file_uploader("Upload Railway Data (CSV/Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is None:
        st.info("Please upload a dataset to proceed. You can use the `generate_mock_data.py` script to generate a sample `railway_data.csv`.")
        
        # Display sample format expectations
        st.markdown("""
        ### Expected Dataset Fields:
        - `Train ID`
        - `Route` (Source - Destination)
        - `Date and Time`
        - `Passenger Count`
        - `Number of Coaches`
        - `Platform Number`
        - `Delay Records (mins)` (optional)
        - `Holiday or Weekend Indicator` ('Yes' or 'No')
        """)
        return

    # Load data
    df = load_data(uploaded_file)
    
    if df is None or df.empty:
        st.warning("Uploaded file is empty or invalid.")
        return
        
    st.sidebar.success("Data successfully loaded!")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview & Summary", 
        "📈 Visualizations", 
        "🔮 Demand Prediction",
        "⚙️ Resource Recommendations"
    ])
    
    with tab1:
        st.header("Overview Dashboard")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Records</div></div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df["Route"].nunique()}</div><div class="metric-label">Unique Routes</div></div>', unsafe_allow_html=True)
            
        with col3:
            avg_occ = df["Seat Occupancy (%)"].mean()
            color = "#dc2626" if avg_occ > 80 else "#16a34a"
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color: {color};">{avg_occ:.1f}%</div><div class="metric-label">Avg Occupancy</div></div>', unsafe_allow_html=True)
            
        with col4:
            total_passengers = df['Passenger Count'].sum()
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_passengers:,}</div><div class="metric-label">Total Passengers</div></div>', unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Overcrowding Detection
        st.subheader("⚠️ Overcrowding Detection")
        overcrowded = df[df['Seat Occupancy (%)'] > 100].copy()
        if not overcrowded.empty:
            num_overcrowded = len(overcrowded)
            st.error(f"High Risk: Detected {num_overcrowded} train trips with over 100% capacity!")
            
            # Show summary table of overcrowded trains
            display_cols = ['Date and Time', 'Train ID', 'Route', 'Passenger Count', 'Total Capacity', 'Seat Occupancy (%)']
            st.dataframe(
                overcrowded[display_cols].sort_values(by='Seat Occupancy (%)', ascending=False).head(10),
                use_container_width=True
            )
        else:
            st.success("No critical overcrowding detected in the dataset (Occupancy > 100%).")
            
        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)


    with tab2:
        st.header("Interactive Data Visualizations")
        
        # 1. Passenger Demand By Route
        st.subheader("Passenger Demand by Route")
        route_demand = df.groupby('Route')['Passenger Count'].sum().reset_index()
        route_demand = route_demand.sort_values(by='Passenger Count', ascending=False)
        fig_route = px.bar(
            route_demand, 
            x='Passenger Count', 
            y='Route', 
            orientation='h',
            title='Total Passengers Handled per Route',
            color='Passenger Count',
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_route, use_container_width=True)
        
        # 2. Passenger Demand Over Time
        st.subheader("Passenger Demand Over Time")
        # Ensure data is sorted
        df_sorted = df.sort_values(by='Date and Time')
        # Group by day to make a cleaner line chart
        df_sorted['Date'] = df_sorted['Date and Time'].dt.date
        time_demand = df_sorted.groupby('Date')['Passenger Count'].sum().reset_index()
        fig_time = px.line(
            time_demand, 
            x='Date', 
            y='Passenger Count',
            title='Daily Passenger Count Aggregated',
            markers=True
        )
        # add a trendline
        fig_time.update_traces(line_color='#2563eb')
        st.plotly_chart(fig_time, use_container_width=True)
        
        # 3. Platform Usage Statistics
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("Platform Usage")
            platform_usage = df['Platform Number'].value_counts().reset_index()
            platform_usage.columns = ['Platform Number', 'Count']
            platform_usage['Platform Number'] = platform_usage['Platform Number'].astype(str)
            fig_plat = px.pie(
                platform_usage, 
                names='Platform Number', 
                values='Count',
                hole=0.4,
                title='Frequency of Platform Assignments'
            )
            st.plotly_chart(fig_plat, use_container_width=True)
            
        with col_chart2:
            st.subheader("Occupancy Heatmap (Hour vs Day)")
            # Extract hour and day of week
            df['Hour'] = df['Date and Time'].dt.hour
            df['DayOfWeek'] = df['Date and Time'].dt.day_name()
            # Order days appropriately
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            heatmap_data = pd.pivot_table(
                df, 
                values='Seat Occupancy (%)', 
                index='DayOfWeek', 
                columns='Hour', 
                aggfunc='mean'
            ).reindex(days_order)
            
            fig_heat = px.imshow(
                heatmap_data, 
                labels=dict(x="Hour of Day", y="Day of Week", color="Avg Occupancy (%)"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale="YlOrRd",
                aspect="auto",
                title="Average Seat Occupancy Patterns"
            )
            st.plotly_chart(fig_heat, use_container_width=True)


    with tab3:
        st.header("Machine Learning Demand Prediction")
        st.markdown("We use a **Random Forest Regressor** to learn from historical features and predict future passenger levels.")
        
        # ML Data Prep
        with st.spinner("Training predictive model..."):
            ml_df = df.copy()
            
            # Feature Engineering
            ml_df['Hour'] = ml_df['Date and Time'].dt.hour
            ml_df['DayOfWeek_Num'] = ml_df['Date and Time'].dt.dayofweek
            ml_df['IsWeekend'] = ml_df['DayOfWeek_Num'].apply(lambda x: 1 if x >= 5 else 0)
            ml_df['IsHoliday'] = ml_df['Holiday or Weekend Indicator'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
            
            # Encode Route
            ml_df['RouteCode'] = ml_df['Route'].astype('category').cat.codes
            route_mapping = dict(enumerate(ml_df['Route'].astype('category').cat.categories))
            
            # Features and Target
            features = ['RouteCode', 'Hour', 'DayOfWeek_Num', 'IsWeekend', 'IsHoliday']
            target = 'Passenger Count'
            
            X = ml_df[features]
            y = ml_df[target]
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Evaluate
            predictions = rf_model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            
        st.success(f"Model trained successfully! (R² Score: {r2:.2f})")
        
        # Interactive Prediction Form
        st.subheader("Predict Future Demand")
        
        col_form1, col_form2 = st.columns(2)
        with col_form1:
            pred_route = st.selectbox("Select Route", options=list(route_mapping.values()))
            pred_date = st.date_input("Select Date", value=datetime.now() + timedelta(days=1))
            
        with col_form2:
            pred_time = st.time_input("Select Time", value=datetime.strptime("08:00", "%H:%M").time())
            pred_holiday = st.checkbox("Is it a holiday/festival?", value=False)
            
        # Transform inputs
        if st.button("Predict Passenger Demand"):
            pred_route_code = list(route_mapping.keys())[list(route_mapping.values()).index(pred_route)]
            pred_hour = pred_time.hour
            pred_dayofweek = pred_date.weekday()
            pred_isweekend = 1 if pred_dayofweek >= 5 else 0
            pred_isholiday = 1 if pred_holiday else 0
            
            input_df = pd.DataFrame([{
                'RouteCode': pred_route_code,
                'Hour': pred_hour,
                'DayOfWeek_Num': pred_dayofweek,
                'IsWeekend': pred_isweekend,
                'IsHoliday': pred_isholiday
            }])
            
            # Make prediction
            forecast = rf_model.predict(input_df)[0]
            
            # Save the latest prediction & inputs to session state to be used in Tab 4
            st.session_state['latest_prediction'] = {
                'route': pred_route,
                'date': pred_date,
                'time': pred_time,
                'forecast': forecast,
                'is_holiday': pred_isholiday
            }
            
            st.markdown(f"### Predicted Passengers: <span style='color:#2563eb;'>{int(forecast)}</span>", unsafe_allow_html=True)
            
            # Simple capacity calculation logic for the prediction
            standard_coach_capacity = 60
            needed_coaches = int(np.ceil(forecast / standard_coach_capacity))
            
            st.info(f"💡 At ~{standard_coach_capacity} passengers per coach, you will need approximately **{needed_coaches} coaches** to prevent overcrowding.")
            
            # Dynamic Suggestion Based on Output
            if forecast > 350:
                st.warning(f"🚨 **High Volume Alert**: This prediction implies massive crowding (>{forecast:.0f} passengers). Consider deploying a dedicated high-capacity relief train or immediately maximizing carriage length.")
            elif pred_hour in [7, 8, 9, 17, 18, 19]:
                st.warning(f"⏳ **Peak Commuter Hour**: {int(forecast)} passengers is standard for {pred_time.strftime('%H:%M')}. Ensure quick platform turnover to avoid concourse bottlenecks.")
            elif pred_isholiday == 1:
                 st.info(f"🎉 **Holiday Service**: This {int(forecast)} volume represents adjusted holiday traffic. Adjust staff allocation accordingly.")
            else:
                st.success(f"✅ **Standard Traffic**: {int(forecast)} passengers represents manageable base-load traffic. Standard dispatch protocols apply.")


    with tab4:
        st.header("Active Resource Allocation Recommendations")
        
        # Check if a prediction has been made
        if 'latest_prediction' in st.session_state:
            pred_data = st.session_state['latest_prediction']
            route = pred_data['route']
            forecast = pred_data['forecast']
            time_val = pred_data['time'].strftime('%H:%M')
            date_val = pred_data['date']
            
            st.markdown(f"### Recommendations for Forecasted Trip: {route}")
            st.markdown(f"**Date:** {date_val} | **Time:** {time_val} | **Predicted Load:** {int(forecast)} Passengers")
            
            st.markdown("---")
            recs = []
            
            # Recommendation 1: Coach Scaling
            base_capacity = 60
            baseline_train_coaches = 10 
            baseline_capacity = base_capacity * baseline_train_coaches
            
            if forecast > baseline_capacity:
                shortage = int(forecast - baseline_capacity)
                extra_coaches = int(np.ceil(shortage / base_capacity))
                recs.append({
                    "title": f"🚨 Critical: Add {extra_coaches} Extra Coaches",
                    "desc": f"The forecasted demand ({int(forecast)}) exceeds standard train capacity ({baseline_capacity} pax). You must securely attach at least **{extra_coaches}** extra carriage(s) to the {time_val} run to prevent extreme standing-room crowding."
                })
            else:
                 recs.append({
                    "title": "✅ Standard Coach Configuration is Sufficient",
                    "desc": f"A standard {baseline_train_coaches}-coach train can comfortably seat {baseline_capacity} passengers. The exact forecast of {int(forecast)} allows for a standard operational layout with no alterations needed."
                })
                 
            # Recommendation 2: Frequency & Platforming
            if forecast > 400:
                recs.append({
                    "title": "🚉 High-Density: Deploy Relief Train",
                    "desc": f"Ridership approaches dangerous platform saturation points. Recommend scheduling an unscheduled \"Relief Train\" along the {route} line 15 minutes prior to {time_val}."
                })
            elif pred_data['time'].hour in [7,8,9,17,18,19]:
                recs.append({
                    "title": "⏳ Peak Hour Platform Management",
                    "desc": f"Because {time_val} falls during peak commuter hours, passenger boarding times will increase. Assign this train to your longest, most accessible platform (e.g. Platform 1 or 2) to ease concourse throughput."
                })
            
            # Recommendation 3: Staffing
            if pred_data['is_holiday'] == 1:
                 recs.append({
                    "title": "🎉 Holiday Staffing Required",
                    "desc": "Because this prediction falls on an observed holiday and/or weekend, passenger baggage and boarding confusion will be higher. Double the platform attendant staff for this specific departure."
                })
                 
            for i, rec in enumerate(recs):
                icon = rec['title'].split()[0] if len(rec['title']) > 0 and not rec['title'][0].isalpha() else "👉"
                color = "#dc2626" if "Critical" in rec['title'] else ("#16a34a" if "Standard" in rec['title'] else "#2563eb")
                bg_color = "#fef2f2" if "Critical" in rec['title'] else ("#f0fdf4" if "Standard" in rec['title'] else "#eff6ff")
                
                st.markdown(f"""
                <div style="padding: 15px; border-left: 5px solid {color}; background-color: {bg_color}; margin-bottom: 15px; border-radius: 4px;">
                    <h4 style="margin-top: 0; color: {color};">{rec['title']}</h4>
                    <p style="margin-bottom: 0;">{rec['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
             st.info("👈 **Please generate a prediction in the 'Demand Prediction' tab to see active resource recommendations for that specific forecast.**")
             
        
        st.markdown("---")
        st.subheader("Historical System-Wide Insights")
        
        # Keep the top historic recommendations as generic fallback analysis
        # (The original historical logic is simplified and placed below the active prediction)
        
        avg_occ_per_route = df.groupby('Route')['Seat Occupancy (%)'].mean().sort_values(ascending=False)
        high_demand_routes = avg_occ_per_route[avg_occ_per_route > 85].index.tolist()
        
        if high_demand_routes:
             top_route = high_demand_routes[0]
             st.markdown(f"**Long-Term Trend:** The **{top_route}** consistently requires maximum operational capacity week-over-week.")
             
        hr_occ = df.groupby('Hour')['Passenger Count'].sum()
        peak_hour = hr_occ.idxmax()
        st.markdown(f"**System Peak Hour:** Historically, system-wide crowding occurs strictly around **{peak_hour}:00** daily.")


if __name__ == "__main__":
    main()
