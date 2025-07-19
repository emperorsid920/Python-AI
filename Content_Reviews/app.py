# app.py
"""
Main Streamlit application for Electronic Repair Shop Review Moderator
Provides web interface for uploading, processing, and analyzing reviews
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import io

# Import our custom modules
from config import PAGE_TITLE, PAGE_ICON, LAYOUT, REQUIRED_COLUMNS, REVIEWS_PER_PAGE
from database import DatabaseManager
from content_moderator import ContentModerator
from utils import CSVProcessor, format_confidence, format_date, truncate_text

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if 'moderator' not in st.session_state:
    # The ContentModerator class now handles its own AI availability check
    st.session_state.moderator = ContentModerator()
if 'csv_processor' not in st.session_state:
    st.session_state.csv_processor = CSVProcessor()


def main():
    """Main application function"""
    st.title("🔧 Electronic Repair Shop Review Moderator")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Dashboard", "📁 Upload Reviews", "🔍 Analyze Reviews", "🚩 Flagged Reviews", "📈 Statistics"]
    )

    # Display AI (Gemini) status in sidebar
    if st.session_state.moderator.is_ai_available:
        st.sidebar.success("✅ AI Analysis (Gemini) Active")
    else:
        st.sidebar.warning("⚠️ AI Analysis (Gemini) Not Available - Using Rule-Based Fallback")

    if page == "📊 Dashboard":
        st.header("📊 Dashboard")
        st.markdown("---")
        dashboard_stats = st.session_state.db_manager.get_dashboard_stats()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", dashboard_stats.get('total_reviews', 0))
        with col2:
            st.metric("Processed Reviews", dashboard_stats.get('processed_reviews', 0))
        with col3:
            st.metric("Flagged Reviews", dashboard_stats.get('flagged_reviews', 0))
        with col4:
            st.metric("Unprocessed Reviews", dashboard_stats.get('unprocessed_reviews', 0))

        st.markdown("### Analysis Overview")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Spam Detected", dashboard_stats.get('spam_count', 0))
        with col_s2:
            st.metric("Inappropriate Detected", dashboard_stats.get('inappropriate_count', 0))
        with col_s3:
            sentiment_dist = dashboard_stats.get('sentiment_distribution', {})
            st.metric("Positive Sentiment", sentiment_dist.get('positive', 0))
        with col_s4:
            st.metric("Negative Sentiment", sentiment_dist.get('negative', 0))

        st.markdown("### Quick Actions")
        if st.button("Clear All Data (Reviews & Analysis)"):
            st.session_state.db_manager.clear_all_data()
            st.success("All data cleared successfully!")
            # Changed from st.experimental_rerun()
            st.rerun()


    elif page == "📁 Upload Reviews":
        st.header("📁 Upload Reviews")
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            df = st.session_state.csv_processor.load_csv(uploaded_file)

            if df is not None:
                is_valid, errors = st.session_state.csv_processor.validate_csv_structure(df)

                if is_valid:
                    st.success("CSV file uploaded and validated successfully!")
                    st.dataframe(df.head(), use_container_width=True) # Show first few rows

                    st.markdown("### Save Reviews to Database")
                    if st.button("Save Reviews"):
                        start_time = time.time()
                        rows_added = st.session_state.db_manager.add_reviews_from_dataframe(df)
                        end_time = time.time()
                        st.success(f"{rows_added} new reviews added to the database in {end_time - start_time:.2f} seconds.")
                        st.info("Navigate to 'Analyze Reviews' to process them.")
                else:
                    st.error("CSV validation failed:")
                    for error in errors:
                        st.write(f"- {error}")


    elif page == "🔍 Analyze Reviews":
        st.header("🔍 Analyze Reviews")
        st.markdown("---")
        st.info("Click 'Analyze New Reviews' to process reviews that haven't been analyzed yet.")

        if st.button("Analyze New Reviews"):
            reviews_to_analyze = st.session_state.db_manager.get_unprocessed_reviews()
            if not reviews_to_analyze:
                st.info("No new reviews to analyze.")
            else:
                st.write(f"Found {len(reviews_to_analyze)} new reviews to analyze. This may take a while...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for i, review_data in enumerate(reviews_to_analyze):
                    review_id = review_data['review_id']
                    review_text = review_data['review_text']
                    rating = review_data['rating']

                    # Perform analysis
                    analysis = st.session_state.moderator.analyze_review(review_text, rating, review_id)
                    results.append(analysis)

                    # Save analysis result to database
                    st.session_state.db_manager.add_analysis_result(analysis)

                    progress = (i + 1) / len(reviews_to_analyze)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {i + 1}/{len(reviews_to_analyze)} reviews.")
                    time.sleep(0.05) # Small delay for UI update and potential API rate limiting

                st.success("Analysis complete!")
                st.dataframe(pd.DataFrame(results).head(), use_container_width=True) # Show first few results


    elif page == "🚩 Flagged Reviews":
        st.header("🚩 Flagged Reviews")
        st.markdown("---")
        st.info("Reviews flagged as spam or inappropriate.")

        flagged_reviews = st.session_state.db_manager.get_flagged_reviews()

        if flagged_reviews.empty:
            st.write("No flagged reviews found yet.")
        else:
            # Add filters
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                filter_spam = st.checkbox("Show Spam", value=True)
            with col_filter2:
                filter_inappropriate = st.checkbox("Show Inappropriate", value=True)

            filtered_df = flagged_reviews.copy()
            if not filter_spam:
                filtered_df = filtered_df[filtered_df['is_spam'] == 0]
            if not filter_inappropriate:
                filtered_df = filtered_df[filtered_df['is_inappropriate'] == 0]

            if filtered_df.empty:
                st.write("No reviews match the selected filters.")
            else:
                st.dataframe(filtered_df.apply(lambda x: pd.Series({
                    'Review ID': x['review_id'],
                    'Review Text': truncate_text(x['review_text'], MAX_DISPLAY_LENGTH),
                    'Rating': x['rating'],
                    'Sentiment': x['sentiment'],
                    'Spam': 'Yes' if x['is_spam'] else 'No',
                    'Inappropriate': 'Yes' if x['is_inappropriate'] else 'No',
                    'Confidence': format_confidence(x['confidence']),
                    'Reasoning': x['reasoning'],
                    'Date': format_date(x['date']),
                    'Reviewer': x['reviewer_name']
                })), use_container_width=True)

    elif page == "📈 Statistics":
        st.header("📈 Statistics")
        st.markdown("---")

        all_reviews = st.session_state.db_manager.get_all_reviews_with_analysis()

        if all_reviews.empty:
            st.write("No analyzed data available for statistics.")
            return

        stats = st.session_state.moderator.get_analysis_summary(all_reviews.to_dict('records'))

        st.markdown("### Overall Analysis Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyzed", stats.get('total_analyzed', 0))
        with col2:
            st.metric("Total Flagged", stats.get('total_flagged', 0))
        with col3:
            st.metric("Flagged Percentage", f"{stats.get('flagged_percentage', 0.0)}%")

        st.markdown("### Distribution Charts")
        col1, col2 = st.columns(2)
        with col1:
            # Rating distribution
            fig = px.histogram(all_reviews, x='rating', title='Rating Distribution')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Sentiment distribution
            if stats['sentiment_distribution']:
                fig = px.pie(
                    values=list(stats['sentiment_distribution'].values()),
                    names=list(stats['sentiment_distribution'].keys()),
                    title='Sentiment Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Time series analysis
        if 'date' in all_reviews.columns:
            st.markdown("### 📅 Time Series Analysis")

            # Convert date column to datetime
            all_reviews['date'] = pd.to_datetime(all_reviews['date'])

            # Reviews over time
            daily_reviews = all_reviews.groupby(all_reviews['date'].dt.date).size().reset_index()
            daily_reviews.columns = ['date', 'count']

            fig = px.line(daily_reviews, x='date', y='count', title='Reviews Over Time')
            st.plotly_chart(fig, use_container_width=True)

        # Reviewer analysis
        st.markdown("### 👥 Reviewer Analysis")
        reviewer_stats = st.session_state.db_manager.get_reviewer_patterns()

        if len(reviewer_stats) > 0:
            st.markdown("**Multiple Review Patterns (Potential Spam Indicators):**")
            st.dataframe(reviewer_stats, use_container_width=True)


if __name__ == "__main__":
    main()