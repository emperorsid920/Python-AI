import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

# File path configuration
CSV_FILE_PATH = "mountain_peak_bike_rentals_2023_2024.csv"


def load_and_analyze_data(csv_file_path):
    """Load CSV data and perform basic analysis"""
    try:
        df = pd.read_csv(csv_file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error(f"CSV file '{csv_file_path}' not found. Please make sure the file exists.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def get_seasonal_summary(df):
    """Generate seasonal performance summary"""
    seasonal_stats = df.groupby('season').agg({
        'total_fixed_revenue': 'sum',
        'total_dynamic_revenue': 'sum',
        'revenue_increase_percent': 'mean',
        'mountain_bike_bookings': 'mean',
        'ebike_bookings': 'mean',
        'temperature_f': 'mean'
    }).round(2)

    seasonal_stats['revenue_difference'] = (
            seasonal_stats['total_dynamic_revenue'] - seasonal_stats['total_fixed_revenue']
    )

    return seasonal_stats


def get_weather_impact(df):
    """Analyze impact of weather on pricing and bookings"""
    weather_stats = df.groupby('weather_condition').agg({
        'mountain_bike_dynamic_price': 'mean',
        'ebike_dynamic_price': 'mean',
        'mountain_bike_bookings': 'mean',
        'ebike_bookings': 'mean',
        'revenue_increase_percent': 'mean',
        'temperature_f': 'mean'
    }).round(2)

    return weather_stats.sort_values('revenue_increase_percent', ascending=False)


def get_best_worst_days(df, n=5):
    """Find best and worst performing days"""
    best_days = df.nlargest(n, 'revenue_difference')[
        ['date', 'day_of_week', 'weather_condition', 'temperature_f',
         'revenue_difference', 'revenue_increase_percent']
    ]

    worst_days = df.nsmallest(n, 'revenue_difference')[
        ['date', 'day_of_week', 'weather_condition', 'temperature_f',
         'revenue_difference', 'revenue_increase_percent']
    ]

    return best_days, worst_days


def create_monthly_trend(df):
    """Create monthly revenue trend data"""
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['year_month'] = df['date'].dt.to_period('M')

    monthly_data = df.groupby(['year', 'month']).agg({
        'total_fixed_revenue': 'sum',
        'total_dynamic_revenue': 'sum',
        'mountain_bike_bookings': 'sum',
        'ebike_bookings': 'sum'
    }).reset_index()

    monthly_data['revenue_difference'] = (
            monthly_data['total_dynamic_revenue'] - monthly_data['total_fixed_revenue']
    )
    monthly_data['month_name'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1)).dt.strftime('%Y-%m')

    return monthly_data


def get_demand_patterns(df):
    """Analyze demand patterns by day of week"""
    dow_stats = df.groupby('day_of_week').agg({
        'mountain_bike_bookings': 'mean',
        'ebike_bookings': 'mean',
        'mountain_bike_dynamic_price': 'mean',
        'ebike_dynamic_price': 'mean',
        'revenue_increase_percent': 'mean'
    }).round(2)

    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = dow_stats.reindex(day_order)

    return dow_stats


def simulate_scenario(model, scenario_params):
    """Simulate pricing for a given scenario"""
    try:
        prediction = model.predict_price(
            temperature=scenario_params['temperature'],
            month=scenario_params['month'],
            day_of_year=scenario_params['day_of_year'],
            is_weekend=scenario_params['is_weekend'],
            season=scenario_params['season'],
            weather_condition=scenario_params['weather'],
            mb_capacity=scenario_params['mb_capacity'],
            eb_capacity=scenario_params['eb_capacity']
        )

        # Calculate potential revenue with estimated bookings
        estimated_mb_bookings = min(30, max(1, int(30 * scenario_params['mb_capacity'])))
        estimated_eb_bookings = min(20, max(1, int(20 * scenario_params['eb_capacity'])))

        fixed_revenue = (estimated_mb_bookings * 45) + (estimated_eb_bookings * 65)
        dynamic_revenue = (estimated_mb_bookings * prediction['mountain_bike_price']) + (
                    estimated_eb_bookings * prediction['ebike_price'])

        return {
            'mountain_bike_price': prediction['mountain_bike_price'],
            'ebike_price': prediction['ebike_price'],
            'estimated_mb_bookings': estimated_mb_bookings,
            'estimated_eb_bookings': estimated_eb_bookings,
            'fixed_revenue': fixed_revenue,
            'dynamic_revenue': dynamic_revenue,
            'revenue_difference': dynamic_revenue - fixed_revenue,
            'revenue_increase_percent': ((
                                                     dynamic_revenue - fixed_revenue) / fixed_revenue) * 100 if fixed_revenue > 0 else 0
        }
    except Exception as e:
        st.error(f"Error in scenario simulation: {str(e)}")
        return None


def format_currency(amount):
    """Format currency for display"""
    return f"${amount:,.2f}"


def format_percentage(percent):
    """Format percentage for display"""
    return f"{percent:+.1f}%"


def get_capacity_recommendations(current_bookings, max_capacity):
    """Get capacity utilization recommendations"""
    utilization = current_bookings / max_capacity

    if utilization >= 0.9:
        return "🔴 High Demand - Premium Pricing Opportunity"
    elif utilization >= 0.7:
        return "🟡 Good Demand - Standard Pricing"
    elif utilization >= 0.4:
        return "🟢 Moderate Demand - Consider Incentives"
    else:
        return "🔵 Low Demand - Discount Pricing Recommended"