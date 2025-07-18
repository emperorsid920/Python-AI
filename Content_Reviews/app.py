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

    # Display Ollama status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Status")
    if st.session_state.moderator.is_ollama_available:
        st.sidebar.success("✅ Ollama AI Available")
        st.sidebar.info(f"Model: {st.session_state.moderator.model}")
    else:
        st.sidebar.warning("⚠️ Ollama Offline - Using Rule-based Analysis")

    # Route to appropriate page
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "📁 Upload Reviews":
        show_upload_page()
    elif page == "🔍 Analyze Reviews":
        show_analyze_page()
    elif page == "🚩 Flagged Reviews":
        show_flagged_page()
    elif page == "📈 Statistics":
        show_statistics_page()


def show_dashboard():
    """Display main dashboard with overview stats"""
    st.header("📊 Dashboard Overview")

    # Get statistics
    stats = st.session_state.db_manager.get_statistics()

    if stats['total_reviews'] == 0:
        st.info("👋 Welcome! Upload your reviews.csv file to get started.")
        st.markdown("### 📋 Getting Started")
        st.markdown("""
        1. **Upload Reviews**: Go to 'Upload Reviews' page and upload your CSV file
        2. **Analyze Content**: Process reviews through AI moderation
        3. **Review Results**: Check flagged reviews and statistics
        4. **Monitor Trends**: Track sentiment and patterns over time
        """)
        return

    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📝 Total Reviews", stats['total_reviews'])

    with col2:
        st.metric("✅ Processed", stats['processed_reviews'])

    with col3:
        st.metric("🚩 Flagged", stats['flagged_reviews'])

    with col4:
        st.metric("📊 Processing Rate", f"{stats['processing_rate']}%")

    # Show progress bar
    if stats['total_reviews'] > 0:
        progress = stats['processed_reviews'] / stats['total_reviews']
        st.progress(progress)
        st.caption(f"Progress: {stats['processed_reviews']}/{stats['total_reviews']} reviews processed")

    # Quick action buttons
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Analyze Unprocessed Reviews", use_container_width=True):
            st.switch_page("🔍 Analyze Reviews")

    with col2:
        if st.button("🚩 View Flagged Reviews", use_container_width=True):
            st.switch_page("🚩 Flagged Reviews")

    with col3:
        if st.button("📈 View Statistics", use_container_width=True):
            st.switch_page("📈 Statistics")

    # Recent activity
    if stats['processed_reviews'] > 0:
        st.markdown("---")
        st.markdown("### 📋 Recent Reviews")

        # Get recent reviews
        recent_reviews = st.session_state.db_manager.get_all_reviews().head(5)

        for _, review in recent_reviews.iterrows():
            with st.expander(f"Review {review['review_id']} - {review['rating']}⭐"):
                st.write(f"**Reviewer:** {review['reviewer_name']}")
                st.write(f"**Date:** {format_date(review['date'])}")
                st.write(f"**Review:** {truncate_text(review['review_text'], 150)}")

                if pd.notna(review['sentiment']):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        sentiment_color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
                        st.write(
                            f"**Sentiment:** {sentiment_color.get(review['sentiment'], '⚪')} {review['sentiment']}")
                    with col2:
                        st.write(f"**Confidence:** {format_confidence(review['confidence'])}")
                    with col3:
                        if review['flagged_for_review']:
                            st.error("🚩 Flagged")
                        else:
                            st.success("✅ Clean")


def show_upload_page():
    """Display file upload page"""
    st.header("📁 Upload Reviews")

    st.markdown("""
    Upload your CSV file containing electronic repair shop reviews. 
    The file should contain the following columns:
    - `review_id`: Unique identifier for each review
    - `review_text`: The actual review content
    - `rating`: Star rating (1-5)
    - `date`: Review date (YYYY-MM-DD format)
    - `reviewer_name`: Name of the reviewer
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with review data"
    )

    if uploaded_file is not None:
        try:
            # Process the uploaded file
            df = st.session_state.csv_processor.load_csv(uploaded_file)

            st.success(f"✅ Successfully loaded {len(df)} reviews!")

            # Show preview
            st.markdown("### 👀 Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Show data quality info
            st.markdown("### 📊 Data Quality Check")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("📝 Total Reviews", len(df))
                st.metric("📅 Date Range", f"{df['date'].min()} to {df['date'].max()}")

            with col2:
                st.metric("⭐ Avg Rating", f"{df['rating'].mean():.1f}")
                st.metric("👥 Unique Reviewers", df['reviewer_name'].nunique())

            # Rating distribution
            fig = px.histogram(df, x='rating', title='Rating Distribution')
            st.plotly_chart(fig, use_container_width=True)

            # Import button
            if st.button("📥 Import Reviews to Database", type="primary", use_container_width=True):
                with st.spinner("Importing reviews..."):
                    inserted_count = st.session_state.db_manager.insert_reviews_from_csv(df)

                if inserted_count > 0:
                    st.success(f"✅ Successfully imported {inserted_count} new reviews!")
                    st.info("💡 You can now go to the 'Analyze Reviews' page to process them.")
                else:
                    st.warning("⚠️ No new reviews were imported. They may already exist in the database.")

                # Refresh the page to show updated stats
                time.sleep(2)
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check that your CSV file has the correct format and column names.")


def show_analyze_page():
    """Display analysis page"""
    st.header("🔍 Analyze Reviews")

    # Get unprocessed reviews
    unprocessed_df = st.session_state.db_manager.get_unprocessed_reviews()

    if len(unprocessed_df) == 0:
        st.info("🎉 All reviews have been processed!")

        # Option to reprocess all reviews
        if st.button("🔄 Reprocess All Reviews"):
            st.warning("This will reanalyze all reviews. This may take some time.")
            if st.button("✅ Confirm Reprocessing"):
                all_reviews = st.session_state.db_manager.get_all_reviews()
                process_reviews(all_reviews[['review_id', 'review_text', 'rating', 'reviewer_name']])
        return

    st.info(f"📝 Found {len(unprocessed_df)} unprocessed reviews")

    # Show preview of unprocessed reviews
    with st.expander("👀 Preview Unprocessed Reviews"):
        st.dataframe(unprocessed_df.head(10), use_container_width=True)

    # Analysis options
    st.markdown("### ⚙️ Analysis Options")

    col1, col2 = st.columns(2)

    with col1:
        batch_size = st.slider("Batch Size", 1, min(100, len(unprocessed_df)),
                               min(20, len(unprocessed_df)),
                               help="Number of reviews to process at once")

    with col2:
        if st.session_state.moderator.is_ollama_available:
            st.success("🤖 AI Analysis Available")
        else:
            st.warning("⚠️ Using Rule-based Analysis")

    # Start analysis button
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        reviews_to_process = unprocessed_df.head(batch_size)
        process_reviews(reviews_to_process)


def process_reviews(reviews_df):
    """Process reviews with progress tracking"""
    total_reviews = len(reviews_df)

    # Create progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Process reviews with progress callback
    def update_progress(progress, current, total):
        progress_bar.progress(progress)
        status_text.text(f"Processing review {current}/{total}...")

    try:
        results = st.session_state.moderator.batch_analyze(reviews_df, update_progress)

        # Store results in database
        success_count = 0
        for result in results:
            if st.session_state.db_manager.insert_moderation_result(result['review_id'], result):
                success_count += 1

        # Show results
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis Complete!")

        st.success(f"🎉 Successfully analyzed {success_count} reviews!")

        # Show summary
        summary = st.session_state.moderator.get_analysis_summary(results)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🚩 Flagged Reviews", f"{summary['total_flagged']}")

        with col2:
            st.metric("📊 Flagged Rate", f"{summary['flagged_percentage']}%")

        with col3:
            st.metric("🎯 Avg Confidence", f"{summary['average_confidence']}")

        # Show flagged reviews
        if summary['total_flagged'] > 0:
            st.markdown("### 🚩 Newly Flagged Reviews")
            flagged_results = [r for r in results if r['is_spam'] or r['is_inappropriate']]

            for result in flagged_results:
                review_row = reviews_df[reviews_df['review_id'] == int(result['review_id'])].iloc[0]

                with st.expander(f"⚠️ Review {result['review_id']} - {review_row['rating']}⭐"):
                    st.write(f"**Reviewer:** {review_row['reviewer_name']}")
                    st.write(f"**Review:** {review_row['review_text']}")
                    st.write(f"**Reasoning:** {result['reasoning']}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if result['is_spam']:
                            st.error("🚫 Spam Detected")
                        if result['is_inappropriate']:
                            st.error("⚠️ Inappropriate Content")
                    with col2:
                        st.write(f"**Confidence:** {format_confidence(result['confidence'])}")

        time.sleep(2)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")


def show_flagged_page():
    """Display flagged reviews page"""
    st.header("🚩 Flagged Reviews")

    flagged_df = st.session_state.db_manager.get_flagged_reviews()

    if len(flagged_df) == 0:
        st.info("🎉 No flagged reviews found! Your reviews look clean.")
        return

    st.warning(f"⚠️ Found {len(flagged_df)} flagged reviews requiring attention")

    # Filter options
    st.markdown("### 🔍 Filter Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_spam = st.checkbox("Show Spam", value=True)

    with col2:
        filter_inappropriate = st.checkbox("Show Inappropriate", value=True)

    with col3:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5)

    # Apply filters
    filtered_df = flagged_df[
        (flagged_df['confidence'] >= min_confidence) &
        ((flagged_df['is_spam'] & filter_spam) |
         (flagged_df['is_inappropriate'] & filter_inappropriate))
        ]

    st.info(f"Showing {len(filtered_df)} flagged reviews")

    # Display flagged reviews
    for _, review in filtered_df.iterrows():
        with st.expander(f"Review {review['review_id']} - {review['rating']}⭐ - {format_date(review['date'])}"):

            # Review content
            st.markdown(f"**Reviewer:** {review['reviewer_name']}")
            st.markdown(f"**Review Text:**")
            st.write(review['review_text'])

            # Flags and analysis
            col1, col2 = st.columns(2)

            with col1:
                if review['is_spam']:
                    st.error("🚫 SPAM DETECTED")
                if review['is_inappropriate']:
                    st.error("⚠️ INAPPROPRIATE CONTENT")

                sentiment_colors = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
                st.write(f"**Sentiment:** {sentiment_colors.get(review['sentiment'], '⚪')} {review['sentiment']}")

            with col2:
                st.write(f"**Confidence:** {format_confidence(review['confidence'])}")
                st.write(f"**Analysis:** {review['reasoning']}")

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Mark as Safe", key=f"safe_{review['review_id']}"):
                    st.success("Review marked as safe (feature coming soon)")
            with col2:
                if st.button(f"❌ Confirm Flag", key=f"flag_{review['review_id']}"):
                    st.error("Review confirmed as problematic (feature coming soon)")


def show_statistics_page():
    """Display comprehensive statistics page"""
    st.header("📈 Statistics & Analytics")

    stats = st.session_state.db_manager.get_statistics()

    if stats['total_reviews'] == 0:
        st.info("No data available. Please upload and analyze some reviews first.")
        return

    # Overall metrics
    st.markdown("### 📊 Overall Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📝 Total Reviews", stats['total_reviews'])

    with col2:
        st.metric("🚩 Flagged Reviews", stats['flagged_reviews'])

    with col3:
        st.metric("📊 Flag Rate", f"{(stats['flagged_reviews'] / stats['total_reviews'] * 100):.1f}%")

    with col4:
        st.metric("🎯 Avg Confidence", stats['average_confidence'])

    # Get all reviews for detailed analysis
    all_reviews = st.session_state.db_manager.get_all_reviews()

    if len(all_reviews) > 0:
        # Charts
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