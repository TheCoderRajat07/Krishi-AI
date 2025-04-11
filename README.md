# KrishiAI - Smart Farming Platform

KrishiAI is a comprehensive smart farming platform that integrates machine learning models with IoT sensor data from ThingSpeak to provide intelligent farming recommendations and real-time monitoring.

## Features

- **ML Predictions & Recommendations**
  - Crop Recommendation based on soil parameters
  - Fertilizer Recommendation for optimal crop growth
  - Crop Price Prediction for better market decisions

- **ThingSpeak Dashboard**
  - Real-time sensor data visualization
  - Temperature, humidity, soil moisture, and light intensity monitoring
  - Historical data analysis with interactive charts

## Technologies Used

- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Data Visualization**: Chart.js
- **ML Models**: Scikit-learn
- **IoT Integration**: ThingSpeak API

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/krishiai.git
   cd krishiai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the application at `http://localhost:8000`

## Project Structure

```
krishiai/
├── app.py                  # FastAPI application
├── requirements.txt        # Python dependencies
├── static/                 # Static files
│   ├── style.css           # CSS styles
│   └── script.js           # JavaScript for static pages
├── templates/              # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # ML predictions page
│   └── dashboard.html      # ThingSpeak dashboard
└── models/                 # ML models
    ├── Crop_recommendation_model.pkl
    ├── Fertilizer_recommendation.pkl
    └── Crop_price_prediction_model.pkl
```

## ThingSpeak Integration

The application connects to ThingSpeak using the following configuration:
- Channel ID: 2914283
- Read API Key: AX3FPC99JAZGSA5S

Sensor data fields:
- Field 1: Temperature (°C)
- Field 2: Humidity (%)
- Field 3: Soil Moisture (%)
- Field 4: Light Intensity (lux)

## ML Models

The application uses three pre-trained machine learning models:

1. **Crop Recommendation Model**: Recommends the best crop to plant based on soil parameters and environmental conditions.
2. **Fertilizer Recommendation Model**: Suggests the appropriate fertilizer based on soil nutrient levels and crop type.
3. **Crop Price Prediction Model**: Predicts future crop prices based on historical data and seasonal trends.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ThingSpeak for IoT data hosting
- Scikit-learn for machine learning capabilities
- FastAPI for the web framework 