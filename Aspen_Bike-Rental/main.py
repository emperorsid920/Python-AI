import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Import our custom modules
from pricing_model import DynamicPricingModel, calculate_revenue_impact
from data_utils import (
    load_and_analyze_data, get_seasonal_summary, get_weather_impact,
    get_best_worst_days, create_monthly_trend, get_demand_patterns,
    simulate_scenario, format_currency, format_percentage, get_capacity_recommendations,
    CSV_FILE_PATH
)

# Page config
st.set_page_config(
    page_title="Mountain Peak Bike Rentals - Dynamic Pricing Dashboard",
    page_icon="🚴‍♂️",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = DynamicPricingModel()
    st.session_state.data_loaded = False
    st.session_state.model_trained = False

# Main title
st.title("🚴‍♂️ Mountain Peak Bike Rentals")
st.subheader("Dynamic Pricing Intelligence Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", [
    "📊 Revenue Analysis",
    "🤖 ML Model Training",
    "💰 Price Predictor",
    "📈 Market Insights",
    "🎯 Scenario Planner"
])

# Data loading
df = load_and_analyze_data(CSV_FILE_PATH)

if df is not None:
    st.session_state.data_loaded = True

    if page == "📊 Revenue Analysis":
        st.header("Revenue Impact Analysis")

        # Calculate revenue impact
        revenue_stats = calculate_revenue_impact(CSV_FILE_PATH)

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Revenue (Fixed)",
                format_currency(revenue_stats['total_fixed_revenue']),
                help="Revenue with current fixed pricing"
            )

        with col2:
            st.metric(
                "Total Revenue (Dynamic)",
                format_currency(revenue_stats['total_dynamic_revenue']),
                help="Revenue with ML-optimized dynamic pricing"
            )

        with col3:
            st.metric(
                "Additional Revenue",
                format_currency(revenue_stats['additional_revenue']),
                format_percentage(revenue_stats['increase_percentage'])
            )

        with col4:
            st.metric(
                "Avg Daily Increase",
                format_currency(revenue_stats['average_daily_increase']),
                help="Average additional revenue per day"
            )

        # Monthly trend chart
        st.subheader("Monthly Revenue Comparison")
        monthly_data = create_monthly_trend(df)

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly_data['month_name'],
            y=monthly_data['total_fixed_revenue'],
            mode='lines+markers',
            name='Fixed Pricing',
            line=dict(color='red')
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly_data['month_name'],
            y=monthly_data['total_dynamic_revenue'],
            mode='lines+markers',
            name='Dynamic Pricing',
            line=dict(color='green')
        ))
        fig_monthly.update_layout(
            title="Monthly Revenue: Fixed vs Dynamic Pricing",
            xaxis_title="Month",
            yaxis_title="Revenue ($)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Seasonal breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Seasonal Performance")
            seasonal_data = get_seasonal_summary(df)
            st.dataframe(seasonal_data, use_container_width=True)

        with col2:
            st.subheader("Best Performing Days")
            best_days, _ = get_best_worst_days(df, 5)
            st.dataframe(best_days, use_container_width=True)

    elif page == "🤖 ML Model Training":
        st.header("Machine Learning Model Training")

        if st.button("Train ML Model", type="primary"):
            with st.spinner("Training Random Forest models..."):
                metrics = st.session_state.model.train(CSV_FILE_PATH)
                st.session_state.model_trained = True
                st.session_state.model.save_model()

                st.success("✅ Models trained successfully!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mountain Bike MAE", f"${metrics['mountain_bike_mae']:.2f}")
                with col2:
                    st.metric("E-Bike MAE", f"${metrics['ebike_mae']:.2f}")

        # Try to load existing model
        elif st.session_state.model.load_model():
            st.session_state.model_trained = True
            st.success("✅ Pre-trained model loaded successfully!")

        if st.session_state.model_trained:
            st.subheader("Feature Importance")
            importance_data = st.session_state.model.get_feature_importance()

            if importance_data:
                col1, col2 = st.columns(2)

                with col1:
                    fig_mb = px.bar(
                        x=importance_data['mountain_bike_importance'],
                        y=importance_data['features'],
                        orientation='h',
                        title="Mountain Bike Pricing - Feature Importance"
                    )
                    st.plotly_chart(fig_mb, use_container_width=True)

                with col2:
                    fig_eb = px.bar(
                        x=importance_data['ebike_importance'],
                        y=importance_data['features'],
                        orientation='h',
                        title="E-Bike Pricing - Feature Importance"
                    )
                    st.plotly_chart(fig_eb, use_container_width=True)

    elif page == "💰 Price Predictor":
        st.header("Real-Time Price Prediction")

        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the ML model first in the 'ML Model Training' section.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Current Conditions")
                temperature = st.slider("Temperature (°F)", 20, 85, 65)
                weather = st.selectbox("Weather", ["sunny", "partly_cloudy", "cloudy", "overcast", "rainy", "snowy"])
                season = st.selectbox("Season", ["peak_season", "shoulder_season", "off_season"])

            with col2:
                st.subheader("Business Factors")
                date_input = st.date_input("Date", datetime.now())
                is_weekend = date_input.weekday() >= 5
                st.write(f"Weekend: {'Yes' if is_weekend else 'No'}")

                mb_capacity = st.slider("Mountain Bike Capacity Used", 0.0, 1.0, 0.5, 0.1)
                eb_capacity = st.slider("E-Bike Capacity Used", 0.0, 1.0, 0.5, 0.1)

            # Make prediction
            scenario_params = {
                'temperature': temperature,
                'month': date_input.month,
                'day_of_year': date_input.timetuple().tm_yday,
                'is_weekend': 1 if is_weekend else 0,
                'season': season,
                'weather': weather,
                'mb_capacity': mb_capacity,
                'eb_capacity': eb_capacity
            }

            prediction = simulate_scenario(st.session_state.model, scenario_params)

            if prediction:
                st.subheader("🎯 Recommended Pricing")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Mountain Bike",
                        f"${prediction['mountain_bike_price']:.0f}",
                        f"vs $45 fixed (+{prediction['mountain_bike_price'] - 45:.0f})"
                    )
                    st.write(get_capacity_recommendations(prediction['estimated_mb_bookings'], 30))

                with col2:
                    st.metric(
                        "E-Bike",
                        f"${prediction['ebike_price']:.0f}",
                        f"vs $65 fixed (+{prediction['ebike_price'] - 65:.0f})"
                    )
                    st.write(get_capacity_recommendations(prediction['estimated_eb_bookings'], 20))

                with col3:
                    st.metric(
                        "Revenue Impact",
                        format_currency(prediction['revenue_difference']),
                        format_percentage(prediction['revenue_increase_percent'])
                    )

    elif page == "📈 Market Insights":
        st.header("Market Insights & Analytics")

        # Weather impact analysis
        st.subheader("Weather Impact on Performance")
        weather_stats = get_weather_impact(df)

        col1, col2 = st.columns(2)

        with col1:
            fig_weather = px.bar(
                weather_stats,
                x=weather_stats.index,
                y='revenue_increase_percent',
                title="Revenue Increase % by Weather",
                color='revenue_increase_percent',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_weather, use_container_width=True)

        with col2:
            fig_bookings = px.bar(
                weather_stats,
                x=weather_stats.index,
                y=['mountain_bike_bookings', 'ebike_bookings'],
                title="Average Bookings by Weather",
                barmode='group'
            )
            st.plotly_chart(fig_bookings, use_container_width=True)

        # Day of week patterns
        st.subheader("Weekly Demand Patterns")
        dow_stats = get_demand_patterns(df)

        fig_dow = px.line(
            dow_stats,
            x=dow_stats.index,
            y=['mountain_bike_bookings', 'ebike_bookings'],
            title="Average Daily Bookings by Day of Week",
            markers=True
        )
        st.plotly_chart(fig_dow, use_container_width=True)

        # Detailed stats table
        st.subheader("Detailed Weather Statistics")
        st.dataframe(weather_stats, use_container_width=True)

    elif page == "🎯 Scenario Planner":
        st.header("Scenario Planning & What-If Analysis")

        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the ML model first.")
        else:
            st.subheader("Compare Multiple Scenarios")

            scenarios = []
            scenario_names = []

            # Create 3 scenarios
            for i in range(3):
                with st.expander(f"Scenario {i + 1}", expanded=(i == 0)):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        name = st.text_input(f"Scenario Name", f"Scenario {i + 1}", key=f"name_{i}")
                        temp = st.slider("Temperature", 20, 85, 50 + i * 15, key=f"temp_{i}")
                        weather = st.selectbox("Weather", ["sunny", "cloudy", "rainy"], key=f"weather_{i}")

                    with col2:
                        season = st.selectbox("Season", ["peak_season", "shoulder_season", "off_season"],
                                              key=f"season_{i}")
                        weekend = st.checkbox("Weekend", key=f"weekend_{i}")
                        month = st.slider("Month", 1, 12, 6 + i * 2, key=f"month_{i}")

                    with col3:
                        mb_cap = st.slider("MB Capacity", 0.0, 1.0, 0.3 + i * 0.3, key=f"mb_cap_{i}")
                        eb_cap = st.slider("EB Capacity", 0.0, 1.0, 0.3 + i * 0.3, key=f"eb_cap_{i}")

                    # Calculate prediction for this scenario
                    scenario_params = {
                        'temperature': temp,
                        'month': month,
                        'day_of_year': month * 30,  # Approximation
                        'is_weekend': 1 if weekend else 0,
                        'season': season,
                        'weather': weather,
                        'mb_capacity': mb_cap,
                        'eb_capacity': eb_cap
                    }

                    prediction = simulate_scenario(st.session_state.model, scenario_params)
                    if prediction:
                        scenarios.append(prediction)
                        scenario_names.append(name)

            # Display comparison
            if scenarios:
                st.subheader("Scenario Comparison")

                comparison_data = pd.DataFrame({
                    'Scenario': scenario_names,
                    'Mountain Bike Price': [s['mountain_bike_price'] for s in scenarios],
                    'E-Bike Price': [s['ebike_price'] for s in scenarios],
                    'Total Revenue': [s['dynamic_revenue'] for s in scenarios],
                    'Revenue vs Fixed': [s['revenue_difference'] for s in scenarios],
                    'Increase %': [s['revenue_increase_percent'] for s in scenarios]
                })

                st.dataframe(comparison_data, use_container_width=True)

                # Visualization
                fig_comparison = px.bar(
                    comparison_data,
                    x='Scenario',
                    y=['Mountain Bike Price', 'E-Bike Price'],
                    title="Price Comparison Across Scenarios",
                    barmode='group'
                )
                st.plotly_chart(fig_comparison, use_container_width=True)

else:
    st.error("Could not load data. Please ensure 'mountain_peak_bike_rentals_2023_2024.csv' is in the same directory.")
    st.info("Run the data generation script first to create the CSV file.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚴‍♂️ Mountain Peak Bike Rentals")
st.sidebar.markdown("Dynamic Pricing Dashboard")
st.sidebar.markdown("*Powered by Machine Learning*")