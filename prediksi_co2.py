import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model = pickle.load(open('prediksi_co2.sav', 'rb'))

# Load and preprocess the dataset
df = pd.read_excel("CO2 dataset.xlsx")
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index(['Year'], inplace=True)

st.title('Forecasting CO2')
year = st.slider("Tentukan Tahun", 1, 30, step=1)

# Generate predictions
pred_values = model.forecast(year)

# Debugging: Print lengths of predictions and index
st.write(f"Number of predictions: {len(pred_values)}")
last_year = df.index[-1].year
pred_index = pd.date_range(start=f'{last_year + 1}-01-01', periods=year, freq='YS')
st.write(f"Expected prediction index length: {len(pred_index)}")

# Check if the forecast returned the correct number of predictions
if len(pred_values) != year:
    st.error(f"Expected {year} predictions, but got {len(pred_values)}. Please check the model.")
else:
    pred = pd.DataFrame(pred_values, columns=['CO2'])

    # Ensure the 'CO2' column is numeric
    pred['CO2'] = pd.to_numeric(pred['CO2'], errors='coerce')

    # Check for numeric data
    if pred['CO2'].isnull().all():
        st.error("Prediction data contains no numeric values.")
    else:
        pred_index = pd.date_range(start=f'{last_year + 1}-01-01', periods=len(pred), freq='YS')
        pred.index = pred_index

        if st.button("Predict"):
            col1, col2 = st.columns([2, 3])
            with col1:
                st.dataframe(pred)
            with col2:
                fig, ax = plt.subplots()
                df['CO2'].plot(style='--', color='gray', legend=True, label='known', ax=ax)
                pred['CO2'].plot(color='b', legend=True, label='prediction', ax=ax)
                st.pyplot(fig)
