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
from sklearn.preprocessing import MinMaxScaler

# Embed your Weather API key here
API_KEY = "d224a9f66ffa425eab3180904242310"  # Replace with your actual Weather API key

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

# Function to plot data
def plot_data(df):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Sunlight hours plot
    axes[0].plot(df['date'], df['sunlight_hours'], marker='o', color='gold', label='Sunlight Hours')
    axes[0].set_title('Sunlight Hours Over Time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Sunlight Hours')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # Cloud cover plot
    axes[1].plot(df['date'], df['cloud_cover'], marker='o', color='skyblue', label='Cloud Cover')
    axes[1].set_title('Cloud Cover Over Time')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cloud Cover (%)')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)

    # Temperature plot
    axes[2].plot(df['date'], df['temperature'], marker='o', color='orange', label='Temperature')
    axes[2].set_title('Average Temperature Over Time')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Temperature (°C)')
    axes[2].legend()
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

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
        recommendations[time] = (appliances, f"Use between {peak_hours - 1} PM and {peak_hours} PM")
    
    return recommendations

# Streamlit app
def main():
    st.markdown('<h1 class="main-title">Solar Energy Prediction and Tariff Forecasting</h1>', unsafe_allow_html=True)

    LOCATION = st.text_input("Enter Location:", value="Nagpur")
    
    if st.button("Fetch Weather Data"):
        weather_df = fetch_weather_data(LOCATION)
        if not weather_df.empty:
            # Create synthetic solar energy production data
            weather_df = create_solar_energy_production(weather_df)
            weather_df.ffill(inplace=True)  # Forward fill missing values
            
            # Display the weather data
            st.write(weather_df)

            # Plot the data
            plot_data(weather_df)

            # Prepare data for training
            X = weather_df[['sunlight_hours', 'cloud_cover', 'temperature']]
            y = weather_df['solar_energy_production']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a Random Forest Regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R^2 Score: {r2:.2f}")

            # Appliance scheduling suggestion
            appliance_schedule = suggest_appliance_schedule(y_pred)
            st.write("Suggested Appliance Schedule:")
            for time, (appliances, timing) in appliance_schedule.items():
                st.write(f"{time.capitalize()}: {', '.join(appliances)} ({timing})")

    # Tariff prediction section
    st.markdown("## Tariff Prediction")
    st.subheader("Load your trained LSTM model for tariff prediction")
    
    # Generate synthetic data for tariff prediction
    def create_synthetic_data(num_samples=1000):
        time = np.arange(num_samples)
        data = np.sin(0.1 * time) + 0.1 * np.random.randn(num_samples)  # Example sine wave data with noise
        return data

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length), 0])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    # Generate and prepare the data for tariff prediction
    data = create_synthetic_data()
    data = data.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    SEQ_LENGTH = 10
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, time steps, features)

    # Load the pre-trained LSTM model
    try:
        model = load_model('best_model.keras')  # Load your model here
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

    # If model loading was successful, proceed with predictions
    if 'model' in locals():
        # Make predictions
        predictions = model.predict(X)

        # Inverse transform and ensure non-negative values
        predicted_tariffs = scaler.inverse_transform(predictions)
        predicted_tariffs = np.abs(predicted_tariffs)  # Ensure predicted tariffs are non-negative

        # Inverse transform actual tariffs and ensure non-negative values
        actual_tariffs = scaler.inverse_transform(y.reshape(-1, 1))
        actual_tariffs = np.abs(actual_tariffs)  # Ensure actual tariffs are non-negative

        # Plotting results
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(actual_tariffs, label='Actual Tariffs', color='blue')
        ax.plot(predicted_tariffs, label='Predicted Tariffs', color='red')
        ax.set_title('Comparison of Actual and Predicted Tariffs')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Tariff Values')
        ax.legend()

        # Show the plot in Streamlit
        st.pyplot(fig)

        # Additional Plot: Highlight Low and High Tariff Regions
        threshold = np.mean(actual_tariffs)  # Set a threshold as the mean of actual tariffs
        low_tariff_indices = np.where(actual_tariffs.flatten() < threshold)[0]
        high_tariff_indices = np.where(actual_tariffs.flatten() >= threshold)[0]

        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(actual_tariffs, label='Actual Tariffs', color='blue')
        ax.scatter(low_tariff_indices, actual_tariffs[low_tariff_indices], color='green', label='Low Tariff Regions', marker='o')
        ax.scatter(high_tariff_indices, actual_tariffs[high_tariff_indices], color='orange', label='High Tariff Regions', marker='o')
        ax.set_title('Low and High Tariff Regions')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Tariff Values')
        ax.legend()
        st.pyplot(fig)

# Run the main function
if __name__ == "__main__":
    main()
