import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Embed your Weather API key here
API_KEY = "d224a9f66ffa425eab3180904242310" 

# Apply custom CSS
st.markdown("""
    <style>
    body { background-color: #f0f2f6; }
    .main-title { color: #336699; text-align: center; font-size: 36px; }
    .sidebar .sidebar-content { background-color: #f7f7f9; }
    .reportview-container .markdown-text-container {
        font-family: 'Arial', sans-serif; color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Fetch weather data from Weather API
def fetch_weather_data(location, days=30):
    url = f"http://api.weatherapi.com/v1/history.json?key={API_KEY}&q={location}&dt="
    weather_data = []

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        response = requests.get(url + date)

        if response.status_code == 200:
            data = response.json()
            daily_data = {
                "date": date,
                "sunlight_hours": np.clip(np.random.normal(9, 1), 8, 11),
                "cloud_cover": np.clip(np.random.normal(50, 10), 30, 80),
                "temperature": data['forecast']['forecastday'][0]['day'].get('avgtemp_c', 0),
                "solar_energy_production": None
            }
            weather_data.append(daily_data)
        else:
            st.error(f"Error fetching data for {date}: {response.status_code}")

    return pd.DataFrame(weather_data)

# Create synthetic solar energy production data
def create_solar_energy_production(df):
    sunlight_factor = 2.0
    temperature_factor = 0.05
    cloud_cover_penalty = -0.25

    df['solar_energy_production'] = (
        df['sunlight_hours'] * sunlight_factor +
        df['temperature'] * temperature_factor +
        df['cloud_cover'] * cloud_cover_penalty
    ).clip(lower=10)
    return df

# Appliance Scheduling based on predictions
def suggest_appliance_schedule(predicted_solar_production):
    peak_hours = predicted_solar_production.idxmax()
    schedule = {
        "morning": ["Water Heater", "Washing Machine"],
        "afternoon": ["Air Conditioning", "Oven"],
        "evening": ["Dishwasher", "Television", "Lighting"]
    }
    
    recommendations = {}
    for time, appliances in schedule.items():
        if time == "morning":
            recommendations[time] = (appliances, "8 AM - 11 AM")
        elif time == "afternoon":
            recommendations[time] = (appliances, "12 PM - 3 PM")
        else:
            recommendations[time] = (appliances, "6 PM - 9 PM")

    return recommendations

# Plot weather data
def plot_weather_data(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    axes[0].plot(df['date'], df['sunlight_hours'], marker='o', color='gold')
    axes[0].set_title('Sunlight Hours Over Time')
    axes[1].plot(df['date'], df['cloud_cover'], marker='o', color='skyblue')
    axes[1].set_title('Cloud Cover Over Time')
    axes[2].plot(df['date'], df['temperature'], marker='o', color='orange')
    axes[2].set_title('Temperature Over Time')
    plt.tight_layout()
    st.pyplot(fig)

# Tariff Prediction using LSTM
def predict_tariffs():
    data = np.sin(0.1 * np.arange(1000)) + 0.1 * np.random.randn(1000)
    data = data.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    SEQ_LENGTH = 10
    X = np.array([data_scaled[i:i + SEQ_LENGTH] for i in range(len(data_scaled) - SEQ_LENGTH)])
    y = np.array([data_scaled[i + SEQ_LENGTH] for i in range(len(data_scaled) - SEQ_LENGTH)])

    try:
        model = load_model('best_model.keras')
        predictions = model.predict(X)
        predicted_tariffs = scaler.inverse_transform(predictions)
        actual_tariffs = scaler.inverse_transform(y.reshape(-1, 1))
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(actual_tariffs, label='Actual Tariffs', color='blue')
        ax.plot(predicted_tariffs, label='Predicted Tariffs', color='red')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Main function
def main():
    st.markdown('<h1 class="main-title">Energy Management App</h1>', unsafe_allow_html=True)
    st.sidebar.title("Choose an Analysis Section")

    section = st.sidebar.selectbox("Select Section", 
                                   ["Weather Forecast & Solar Energy Prediction",
                                    "Tariff Prediction with LSTM",
                                    "Appliance Scheduling"])

    # Weather Forecast & Solar Energy Prediction
    if section == "Weather Forecast & Solar Energy Prediction":
        LOCATION = st.text_input("Enter Location:", value="Nagpur")
        if st.button("Fetch Weather Data"):
            weather_df = fetch_weather_data(LOCATION)
            if not weather_df.empty:
                weather_df = create_solar_energy_production(weather_df)
                st.write(weather_df)
                plot_weather_data(weather_df)
                
                X = weather_df[['sunlight_hours', 'cloud_cover', 'temperature']]
                y = weather_df['solar_energy_production']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                st.write(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}')
                st.write(f'R^2 Score: {r2_score(y_test, y_pred):.2f}')

    # Tariff Prediction with LSTM
    elif section == "Tariff Prediction with LSTM":
        predict_tariffs()

    # Appliance Scheduling
    elif section == "Appliance Scheduling":
        LOCATION = "Nagpur" # Replace this with a location as needed
        days = 30  # Use last 30 days for prediction data
        weather_df = fetch_weather_data(LOCATION, days)
        weather_df = create_solar_energy_production(weather_df)
        
        st.write("Predicted Solar Energy Production for the next day")
        st.write(weather_df[['date', 'solar_energy_production']])
        
        predicted_production = weather_df['solar_energy_production'].rolling(window=7, min_periods=1).mean().iloc[-1]
        st.write(f"Predicted Solar Energy for Next Day: {predicted_production:.2f} kWh")
        
        schedule = suggest_appliance_schedule(weather_df['solar_energy_production'])
        st.subheader("Recommended Appliance Usage Schedule")
        
        for time, (appliances, hours) in schedule.items():
            st.write(f"**{time.capitalize()} ({hours}):** {', '.join(appliances)}")

if __name__ == "__main__":
    main()
