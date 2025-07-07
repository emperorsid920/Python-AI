import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from data_processor import DataProcessor
from model import ChurnPredictor, BusinessMetrics
import warnings



warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False


def load_data():
    """Load and cache the dataset"""
    try:
        data = pd.read_csv('Telco-Customer-Churn.csv')
        st.session_state.data = data
        return data
    except FileNotFoundError:
        st.error("❌ Dataset not found. Please ensure 'Telco-Customer-Churn.csv' is in the project directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("🎛️ Control Panel")

    # Navigation
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📈 Model Training", "🔮 Predictions", "💼 Business Insights", "📊 Data Analysis"]
    )

    # Load data
    if st.session_state.data is None:
        with st.spinner("Loading data..."):
            data = load_data()
            if data is None:
                st.stop()
    else:
        data = st.session_state.data

    # Page routing
    if page == "🏠 Overview":
        show_overview(data)
    elif page == "📈 Model Training":
        show_model_training(data)
    elif page == "🔮 Predictions":
        show_predictions(data)
    elif page == "💼 Business Insights":
        show_business_insights(data)
    elif page == "📊 Data Analysis":
        show_data_analysis(data)


def show_overview(data):
    """Overview page with key metrics and business context"""
    st.header("📋 Project Overview")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🎯 Business Objective
        Predict which customers are likely to churn and take proactive measures to retain them.

        ### 📊 Key Benefits
        - **Reduce churn by 15-25%** through targeted retention campaigns
        - **Identify at-risk customers** 2 months in advance
        - **Optimize retention spending** by focusing on high-value customers
        - **Increase customer lifetime value** through proactive engagement
        """)

    with col2:
        st.markdown("### 📈 Dataset Overview")
        st.metric("Total Customers", f"{len(data):,}")

        churn_rate = (data['Churn'] == 'Yes').mean()
        st.metric("Overall Churn Rate", f"{churn_rate:.1%}")

        avg_monthly_charges = data['MonthlyCharges'].mean()
        st.metric("Avg Monthly Charges", f"${avg_monthly_charges:.2f}")

    # Quick data preview
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(), use_container_width=True)

    # Churn distribution
    st.subheader("📊 Churn Distribution")
    fig = px.pie(
        data,
        names='Churn',
        title='Customer Churn Distribution',
        color_discrete_map={'Yes': '#ff7f7f', 'No': '#7fbf7f'}
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_training(data):
    """Model training interface"""
    st.header("🤖 Model Training")

    # Model selection
    col1, col2 = st.columns(2)

    with col1:
        model_choice = st.selectbox(
            "Choose ML Algorithm:",
            ["Random Forest", "Logistic Regression"],
            help="Random Forest: Higher accuracy, feature importance\nLogistic Regression: Faster, more interpretable"
        )

    with col2:
        train_button = st.button("🚀 Train Model", type="primary")

    # Algorithm comparison
    st.subheader("🔍 Algorithm Comparison")

    comparison_data = {
        'Metric': ['Accuracy', 'Training Speed', 'Interpretability', 'Feature Importance', 'Handling Mixed Data'],
        'Random Forest': ['⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
        'Logistic Regression': ['⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐']
    }

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Training process
    if train_button:
        with st.spinner("Training model... This may take a moment."):
            try:
                # Process data
                processor = st.session_state.data_processor
                X_train, X_test, y_train, y_test = processor.process_pipeline(data)

                # Initialize model
                model_type = 'random_forest' if model_choice == 'Random Forest' else 'logistic_regression'
                model = ChurnPredictor(model_type=model_type)

                # Train model
                results = model.train(X_train, y_train, X_test, y_test)

                # Store in session state
                st.session_state.model = model
                st.session_state.model_trained = True

                # Display results
                st.success("✅ Model trained successfully!")

                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.1%}")
                with col2:
                    st.metric("Precision", f"{results['precision']:.1%}")
                with col3:
                    st.metric("Recall", f"{results['recall']:.1%}")
                with col4:
                    st.metric("F1 Score", f"{results['f1']:.1%}")

                # Confusion Matrix
                st.subheader("📊 Model Performance")

                col1, col2 = st.columns(2)

                with col1:
                    # Confusion matrix heatmap
                    cm = results['confusion_matrix']
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title('Confusion Matrix')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)

                with col2:
                    # Feature importance (if available)
                    if model.feature_importance is not None:
                        feature_names = processor.get_feature_names()
                        importance_df = model.get_feature_importance(feature_names)

                        fig = px.bar(
                            importance_df.head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title='Top 10 Most Important Features'
                        )
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

                # Classification report
                st.subheader("📋 Detailed Classification Report")
                st.text(results['classification_report'])

            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")


def show_predictions(data):
    """Prediction interface for single customers and batch predictions"""
    st.header("🔮 Make Predictions")

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Model Training' section.")
        return

    model = st.session_state.model
    processor = st.session_state.data_processor

    # Prediction type selection
    prediction_type = st.selectbox(
        "Choose prediction type:",
        ["Single Customer", "Batch Predictions"]
    )

    if prediction_type == "Single Customer":
        show_single_prediction(data, model, processor)
    else:
        show_batch_predictions(data, model, processor)


def show_single_prediction(data, model, processor):
    """Interface for single customer prediction"""
    st.subheader("👤 Single Customer Prediction")

    # Create input form
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📊 Service Information**")
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

        st.markdown("**📞 Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        st.markdown("**💰 Billing Information**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method",
                                      ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                       "Credit card (automatic)"])

        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

        st.markdown("**🌐 Additional Services**")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary"):
        try:
            # Create customer data
            customer_data = {
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': 'No',  # Default values for simplicity
                'TechSupport': 'No',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }

            # Convert to DataFrame
            customer_df = pd.DataFrame([customer_data])

            # Process the data
            customer_clean = processor.clean_data(customer_df)

            # Prepare features (without target variable)
            categorical_columns = customer_clean.select_dtypes(include=['object']).columns
            customer_processed = customer_clean.copy()

            for column in categorical_columns:
                if column in processor.label_encoders:
                    # Use existing encoder
                    le = processor.label_encoders[column]
                    customer_processed[column] = le.transform(customer_processed[column])

            # Scale features
            customer_scaled = processor.scaler.transform(customer_processed)

            # Make prediction
            prediction = model.predict(customer_scaled)[0]
            probability = model.predict_proba(customer_scaled)[0][1]

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                if prediction == 1:
                    st.error(f"⚠️ HIGH RISK: {probability:.1%} chance of churn")
                else:
                    st.success(f"✅ LOW RISK: {probability:.1%} chance of churn")

            with col2:
                st.metric("Churn Probability", f"{probability:.1%}")

            # Risk level and recommendations
            if probability > 0.7:
                st.error("🚨 **Critical Risk Level** - Immediate action required!")
                st.markdown("""
                **Recommended Actions:**
                - Personal call from customer service
                - Offer special discount or upgrade
                - Schedule retention meeting
                """)
            elif probability > 0.4:
                st.warning("⚠️ **Medium Risk Level** - Monitor closely")
                st.markdown("""
                **Recommended Actions:**
                - Send targeted email campaign
                - Offer loyalty program enrollment
                - Survey for satisfaction feedback
                """)
            else:
                st.success("✅ **Low Risk Level** - Customer is stable")
                st.markdown("""
                **Recommended Actions:**
                - Continue standard service
                - Consider upselling opportunities
                - Maintain regular communication
                """)

        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")


def show_batch_predictions(data, model, processor):
    """Interface for batch predictions"""
    st.subheader("📊 Batch Predictions")

    # File upload
    uploaded_file = st.file_uploader("Upload CSV file for batch predictions", type="csv")

    if uploaded_file is not None:
        try:
            # Load data
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"📋 Loaded {len(batch_data)} customers")

            # Show preview
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head(), use_container_width=True)

            if st.button("🔮 Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    # Process data
                    batch_clean = processor.clean_data(batch_data)

                    # Prepare features
                    categorical_columns = batch_clean.select_dtypes(include=['object']).columns
                    batch_processed = batch_clean.copy()

                    for column in categorical_columns:
                        if column in processor.label_encoders:
                            le = processor.label_encoders[column]
                            batch_processed[column] = le.transform(batch_processed[column])

                    # Scale features
                    batch_scaled = processor.scaler.transform(batch_processed)

                    # Make predictions
                    predictions = model.predict(batch_scaled)
                    probabilities = model.predict_proba(batch_scaled)[:, 1]

                    # Add predictions to original data
                    results_df = batch_data.copy()
                    results_df['Churn_Prediction'] = predictions
                    results_df['Churn_Probability'] = probabilities
                    results_df['Risk_Level'] = pd.cut(probabilities,
                                                      bins=[0, 0.3, 0.7, 1.0],
                                                      labels=['Low', 'Medium', 'High'])

                    # Display results
                    st.success("✅ Predictions generated successfully!")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        high_risk = (probabilities > 0.7).sum()
                        st.metric("High Risk Customers", high_risk)

                    with col2:
                        medium_risk = ((probabilities > 0.3) & (probabilities <= 0.7)).sum()
                        st.metric("Medium Risk Customers", medium_risk)

                    with col3:
                        low_risk = (probabilities <= 0.3).sum()
                        st.metric("Low Risk Customers", low_risk)

                    # Results table
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"❌ Error processing batch predictions: {str(e)}")


def show_business_insights(data):
    """Business insights and metrics"""
    st.header("💼 Business Insights")

    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first to see business insights.")
        return

    model = st.session_state.model
    processor = st.session_state.data_processor

    # Calculate business metrics
    with st.spinner("Calculating business metrics..."):
        # Process data for predictions
        data_clean = processor.clean_data(data)
        X, y = processor.prepare_features(data_clean)
        X_scaled = processor.scaler.transform(X)

        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        # Business metrics
        business_metrics = BusinessMetrics(data)

        # Revenue at risk
        revenue_at_risk = business_metrics.calculate_revenue_at_risk(predictions, data['MonthlyCharges'].values)

        # Customer segments
        segments = business_metrics.segment_customers(probabilities, data['MonthlyCharges'].values)

        # ROI calculation
        roi_data = business_metrics.calculate_retention_roi(revenue_at_risk, 50)  # $50 retention cost per customer

    # Display metrics
    st.subheader("💰 Financial Impact")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}/month")

    with col2:
        st.metric("Annual Risk", f"${revenue_at_risk * 12:,.0f}")

    with col3:
        st.metric("Retention ROI", f"{roi_data['roi_percentage']:.1f}%")

    with col4:
        customers_at_risk = (predictions == 1).sum()
        st.metric("Customers at Risk", f"{customers_at_risk:,}")

    # Customer segments
    st.subheader("🎯 Customer Segmentation")

    segment_data = []
    for segment, info in segments.items():
        segment_data.append({
            'Segment': segment,
            'Customers': info['count'],
            'Revenue at Risk': f"${info['revenue_at_risk']:,.0f}",
            'Recommended Action': info['action']
        })

    st.dataframe(pd.DataFrame(segment_data), use_container_width=True)

    # Segment visualization
    segment_names = list(segments.keys())
    segment_counts = [segments[seg]['count'] for seg in segment_names]

    fig = px.pie(
        values=segment_counts,
        names=segment_names,
        title="Customer Risk Segmentation"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Retention campaign simulation
    st.subheader("📊 Retention Campaign Simulator")

    col1, col2 = st.columns(2)

    with col1:
        campaign_cost = st.slider("Campaign Cost per Customer ($)", 10, 200, 50)
        success_rate = st.slider("Expected Success Rate (%)", 10, 80, 30) / 100

    with col2:
        roi_sim = business_metrics.calculate_retention_roi(revenue_at_risk, campaign_cost, success_rate)
        st.metric("Projected ROI", f"{roi_sim['roi_percentage']:.1f}%")
        st.metric("Revenue Saved", f"${roi_sim['revenue_saved']:,.0f}")


def show_data_analysis(data):
    """Data analysis and visualization"""
    st.header("📊 Data Analysis")

    # Basic statistics
    st.subheader("📈 Dataset Statistics")
    st.write(data.describe())

    # Churn analysis
    st.subheader("🔍 Churn Analysis")

    # Churn by categorical variables
    categorical_cols = ['Contract', 'PaymentMethod', 'InternetService', 'SeniorCitizen']

    for col in categorical_cols:
        if col in data.columns:
            churn_by_cat = data.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean()).reset_index()
            churn_by_cat.columns = [col, 'Churn_Rate']

            fig = px.bar(
                churn_by_cat,
                x=col,
                y='Churn_Rate',
                title=f'Churn Rate by {col}',
                labels={'Churn_Rate': 'Churn Rate'}
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)

    # Correlation analysis
    st.subheader("🔗 Feature Correlations")

    # Prepare numeric data for correlation
    numeric_data = data.select_dtypes(include=[np.number])
    if 'Churn' in data.columns:
        numeric_data['Churn_Numeric'] = (data['Churn'] == 'Yes').astype(int)

    corr_matrix = numeric_data.corr()

    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()