import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import warnings

# Suppress FutureWarnings from Prophet and Pandas
warnings.simplefilter(action="ignore", category=FutureWarning)

# -------------------- CONFIGURATION --------------------
st.set_page_config(page_title="RainVision AI", layout="wide")
st.title("🌧️ RainVision AI - Smart Rainfall Forecasting")

# -------------------- COLUMN DETECTION FUNCTIONS --------------------
def detect_date_column(df):
    """Detects a date-like column dynamically."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "month" in col.lower():
            return col
    return None

def detect_value_column(df):
    """Detects a numerical column representing rainfall values."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or "rain" in col.lower() or "value" in col.lower() or "precipitation" in col.lower():
            return col
    return None

# -------------------- FORECASTING FUNCTION --------------------
def forecast_rainfall(df, months=12):
    # Detect column names dynamically
    date_col = detect_date_column(df)
    value_col = detect_value_column(df)

    if not date_col or not value_col:
        st.error("⚠️ Could not automatically detect suitable date or value columns. Please check your CSV format.")
        st.write("🔍 CSV Columns Available:", df.columns)  # Debugging info
        return None

    # Rename detected columns for Prophet
    df.rename(columns={date_col: 'ds', value_col: 'y'}, inplace=True)
    df['ds'] = pd.to_datetime(df['ds'])  # Ensure datetime format
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Your CSV contains missing values. Cleaning them now...")
        df.dropna(inplace=True)

    # Build and train model
    model = Prophet()
    try:
        model.fit(df)
    except ValueError as e:
        st.error(f"❌ Error fitting Prophet model: {e}")
        return None

    # Make predictions
    future = model.make_future_dataframe(periods=months, freq='ME')  # Updated 'M' to 'ME'
    forecast = model.predict(future)

    return forecast

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader("📁 Upload a rainfall CSV file", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)

    st.write("🔍 CSV Columns Detected:", raw_df.columns)

    forecast = forecast_rainfall(raw_df)
    if forecast is not None:
        st.success("✅ Forecast generated successfully!")

        # -------------------- Forecast Plot --------------------
        st.subheader("📈 Forecast Plot")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(forecast['ds'], forecast['yhat'], label="Predicted Rainfall")
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
            ax.set_xlabel("Date")
            ax.set_ylabel("Rainfall Prediction")
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Error plotting forecast: {e}")

        # -------------------- Forecast Table --------------------
        st.subheader("📊 Forecast Data (Next 12 Months)")
        forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
        st.dataframe(forecast_out)

        # -------------------- Download Button --------------------
        csv_download = forecast_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast as CSV", csv_download, "rainfall_forecast.csv", "text/csv")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("🔧 Built with Prophet, Streamlit | © 2025 Atharv Patil")
