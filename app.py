import streamlit as st
import pandas as pd
import pickle
import os

# Page Config
st.set_page_config(page_title='Seller Churn Prediction', layout='wide')

st.title('Prediksi Sellers Churn')
st.markdown("Enter the seller's metrics below to predict churn probability.")

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Default path for Colab
    path = '/content/ada_final_model_churnNewLable.pkl'
    
    # Check if file exists
    if not os.path.exists(path):
        st.error(f"Model not found at {path}. Please upload your .pkl file.")
        return None
        
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Input Form ---
if model:
    st.header("Seller Metrics")
    
    # Create 3 columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Sales Performance")
        selling_recency = st.number_input('Selling Recency (days)', value=10)
        selling_frequency = st.number_input('Selling Frequency (orders)', value=50)
        avg_selling_internal = st.number_input('Avg Selling Interval (days)', value=5.5)
        total_sales_count = st.number_input('Total Sales Count', value=100)
        total_revenue = st.number_input('Total Revenue', value=15000.0)

    with col2:
        st.subheader("Product & Logistics")
        avg_price = st.number_input('Avg Selling Price', value=150.75)
        unique_products = st.number_input('Unique Products Count', value=10)
        unique_categories = st.number_input('Unique Categories Count', value=5)
        avg_freight_value = st.number_input('Avg Freight Value', value=25.50)
        avg_weight = st.number_input('Avg Weight (g)', value=1500.0)
        avg_volume = st.number_input('Avg Volume (cm³)', value=5000.0)

    with col3:
        st.subheader("Delivery & Retention")
        avg_delivery_delay = st.number_input('Avg Delivery Delay (hours)', value=2.0)
        avg_delivery_time = st.number_input('Avg Delivery Time (hours)', value=48.0)
        delay_rate = st.number_input('Delay Rate (0-1)', value=0.05, min_value=0.0, max_value=1.0, step=0.01)
        avg_distance = st.number_input('Avg Distance (km)', value=300.0)
        repeat_buyer_ratio = st.number_input('Repeat Buyer Ratio (0-1)', value=0.3, min_value=0.0, max_value=1.0, step=0.01)

    st.markdown("---")
    st.subheader("Other Required Features")
    col4, col5 = st.columns(2)
    with col4:
         seller_state = st.selectbox('Seller State', ['SP', 'MG', 'RJ', 'PR', 'SC', 'GO', 'ES', 'PE', 'CE', 'BA', 'DF', 'RS', 'PA', 'MT', 'MA', 'MS', 'RN', 'PB', 'SE', 'AL', 'TO', 'RO', 'PI', 'AC', 'AP', 'AM'])
    with col5:
         avg_review_score = st.number_input('Avg Review Score (1-5)', value=4.2, min_value=1.0, max_value=5.0)
         review_count = st.number_input('Review Count', value=80)
         low_rating_ratio = st.number_input('Low Rating Ratio (0-1)', value=0.1)

    # --- Prediction Logic ---
    st.markdown("---")
    if st.button('Predict Churn', type='primary'):
        # Create DataFrame with exact column names expected by the model
        input_data = pd.DataFrame({
            'selling_recency': [selling_recency],
            'selling_frequency': [selling_frequency],
            'avg_selling_internal': [avg_selling_internal],
            'avg_price': [avg_price],
            'total_sales_count': [total_sales_count],
            'total_revenue': [total_revenue],
            'avg_freight_value': [avg_freight_value],
            'avg_delivery_delay': [avg_delivery_delay],
            'avg_delivery_time': [avg_delivery_time],
            'delay_rate': [delay_rate],
            'avg_distance': [avg_distance],
            'repeat_buyer_ratio': [repeat_buyer_ratio],
            'avg_weight': [avg_weight],
            'avg_volume': [avg_volume],
            'unique_products': [unique_products],
            'unique_categories': [unique_categories],
            'seller_state': [seller_state],       
            'avg_review_score': [avg_review_score], 
            'review_count': [review_count],         
            'low_rating_ratio': [low_rating_ratio] 
        })

        st.write("Input Data Preview:")
        st.dataframe(input_data)

        try:
            # Predict
            prediction = model.predict(input_data)
            st.success(f"Prediction Result: {prediction[0]}")

            # Predict Proba (if supported)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_data)
                st.info(f"Probability of Churn: {proba[0][1]:.2%}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Note: If you get a 'Feature Mismatch' error, ensure all columns used in training are present here.")
