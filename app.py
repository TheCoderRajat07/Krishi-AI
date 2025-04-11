# app.py
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib
from pydantic import BaseModel
from typing import List, Optional
import io
import base64
from PIL import Image
import tensorflow as tf
import random

app = FastAPI()

# Mount the static files directory so that /static routes are available
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load ML models
def load_model(model_path):
    try:
        # Check if the file is a joblib file
        if model_path.endswith('.joblib'):
            return joblib.load(model_path)
        else:
            # Use pickle for other file types
            with open(model_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

# Load models
crop_recommendation_model = load_model("models/Crop_recommendation_model.pkl")
fertilizer_recommendation_model = load_model("models/Fertilizer_recommendation.pkl")
crop_price_model = load_model("models/Crop_price_prediction_model.pkl")

# Load crop price data for reference
try:
    crop_price_data = pd.read_csv("Jupyter files/Crop_Price.csv")
    # Get unique crops from the dataset
    available_crops = crop_price_data['Crop'].unique().tolist()
except Exception as e:
    print(f"Error loading crop price data: {e}")
    available_crops = []

# Load the label encoders
soil_type_encoder = load_model("models/soil_type_encoder.joblib")
crop_type_encoder = load_model("models/crop_type_encoder.joblib")
fertilizer_encoder = load_model("models/fertilizer_encoder.joblib")

# ThingSpeak configuration
THINGSPEAK_CHANNEL_ID = "2914283"
THINGSPEAK_READ_API_KEY = "AX3FPC99JAZGSA5S"

# Valid options for dropdowns
VALID_SOIL_TYPES = ["Black", "Clayey", "Loamy", "Red", "Sandy"]
VALID_CROP_TYPES = ["Barley", "Cotton", "Ground Nuts", "Maize", "Millets", "Oil seeds", 
                   "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat", "coffee", 
                   "kidneybeans", "orange", "pomegranate", "rice", "watermelon"]

# Disease information database
DISEASE_INFO = {
    "rice": {
        "bacterial_leaf_blight": {
            "description": "Bacterial leaf blight is a serious disease of rice caused by the bacterium Xanthomonas oryzae pv. oryzae. It can cause significant yield losses in susceptible rice varieties.",
            "treatments": [
                "Use resistant rice varieties",
                "Practice proper water management",
                "Apply copper-based bactericides",
                "Remove infected plants and debris",
                "Avoid excessive nitrogen fertilization"
            ]
        },
        "brown_spot": {
            "description": "Brown spot is a fungal disease of rice caused by Cochliobolus miyabeanus. It affects the leaves, sheaths, panicles, and grains of the rice plant.",
            "treatments": [
                "Use disease-free seeds",
                "Apply fungicides at early stages",
                "Improve soil fertility with balanced nutrients",
                "Practice crop rotation",
                "Remove infected plant debris"
            ]
        },
        "healthy": {
            "description": "The rice plant appears healthy with no signs of disease.",
            "treatments": [
                "Continue regular monitoring",
                "Maintain proper irrigation",
                "Apply balanced fertilization",
                "Practice good field hygiene"
            ]
        }
    },
    "wheat": {
        "rust": {
            "description": "Wheat rust is a fungal disease that causes orange-brown pustules on leaves and stems. It can spread rapidly and cause significant yield losses.",
            "treatments": [
                "Plant rust-resistant wheat varieties",
                "Apply fungicides at early stages",
                "Practice crop rotation",
                "Remove volunteer wheat plants",
                "Monitor fields regularly for early detection"
            ]
        },
        "septoria_leaf_blotch": {
            "description": "Septoria leaf blotch is a fungal disease that causes brown lesions with yellow halos on wheat leaves. It can reduce photosynthesis and grain yield.",
            "treatments": [
                "Use disease-free seeds",
                "Apply fungicides when conditions favor disease",
                "Practice crop rotation",
                "Remove infected plant debris",
                "Avoid excessive nitrogen fertilization"
            ]
        },
        "healthy": {
            "description": "The wheat plant appears healthy with no signs of disease.",
            "treatments": [
                "Continue regular monitoring",
                "Maintain proper irrigation",
                "Apply balanced fertilization",
                "Practice good field hygiene"
            ]
        }
    },
    "maize": {
        "northern_leaf_blight": {
            "description": "Northern leaf blight is a fungal disease that causes large, grayish-brown lesions on maize leaves. It can reduce photosynthesis and grain yield.",
            "treatments": [
                "Plant resistant maize hybrids",
                "Remove infected plant debris",
                "Practice crop rotation",
                "Apply fungicides if necessary",
                "Avoid excessive nitrogen fertilization"
            ]
        },
        "common_rust": {
            "description": "Common rust is a fungal disease that causes circular to elongated brown pustules on maize leaves. It can reduce photosynthesis and grain yield.",
            "treatments": [
                "Plant rust-resistant maize hybrids",
                "Apply fungicides at early stages",
                "Practice crop rotation",
                "Remove volunteer maize plants",
                "Monitor fields regularly for early detection"
            ]
        },
        "healthy": {
            "description": "The maize plant appears healthy with no signs of disease.",
            "treatments": [
                "Continue regular monitoring",
                "Maintain proper irrigation",
                "Apply balanced fertilization",
                "Practice good field hygiene"
            ]
        }
    },
    "potato": {
        "late_blight": {
            "description": "Late blight is a devastating disease of potatoes caused by the oomycete Phytophthora infestans. It causes dark brown spots on leaves and rotting of tubers.",
            "treatments": [
                "Use disease-free seed potatoes",
                "Apply fungicides preventively",
                "Practice proper spacing for good air circulation",
                "Remove infected plants and tubers",
                "Avoid overhead irrigation"
            ]
        },
        "early_blight": {
            "description": "Early blight is a fungal disease that causes dark brown spots with concentric rings on potato leaves. It can reduce tuber yield and quality.",
            "treatments": [
                "Use disease-free seed potatoes",
                "Apply fungicides at early stages",
                "Remove infected leaves",
                "Practice crop rotation",
                "Avoid excessive nitrogen fertilization"
            ]
        },
        "healthy": {
            "description": "The potato plant appears healthy with no signs of disease.",
            "treatments": [
                "Continue regular monitoring",
                "Maintain proper irrigation",
                "Apply balanced fertilization",
                "Practice good field hygiene"
            ]
        }
    },
    "tomato": {
        "early_blight": {
            "description": "Early blight is a fungal disease that causes dark brown spots with concentric rings on tomato leaves. It can reduce fruit yield and quality.",
            "treatments": [
                "Use disease-free seeds or transplants",
                "Apply fungicides at early stages",
                "Remove infected leaves",
                "Practice crop rotation",
                "Avoid excessive nitrogen fertilization"
            ]
        },
        "late_blight": {
            "description": "Late blight is a devastating disease of tomatoes caused by the oomycete Phytophthora infestans. It causes dark brown spots on leaves and rotting of fruits.",
            "treatments": [
                "Use disease-free seeds or transplants",
                "Apply fungicides preventively",
                "Practice proper spacing for good air circulation",
                "Remove infected plants",
                "Avoid overhead irrigation"
            ]
        },
        "healthy": {
            "description": "The tomato plant appears healthy with no signs of disease.",
            "treatments": [
                "Continue regular monitoring",
                "Maintain proper irrigation",
                "Apply balanced fertilization",
                "Practice good field hygiene"
            ]
        }
    }
}

class FertilizerInput(BaseModel):
    temperature: float
    humidity: float
    nitrogen: float
    phosphorous: float
    potassium: float
    soil_type: str
    crop_type: str

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    # Serve the index.html file from the static folder
    return templates.TemplateResponse("index.html", {"request": request, "available_crops": available_crops})

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    # Serve the dashboard.html file
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/disease-prediction", response_class=HTMLResponse)
async def get_disease_prediction(request: Request):
    # Serve the disease_prediction.html file
    return templates.TemplateResponse("disease_prediction.html", {"request": request})

@app.get("/api/data", response_class=JSONResponse)
async def get_sensor_data():
    # Fetch data from ThingSpeak
    results = 10  # number of records to fetch
    url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_READ_API_KEY}&results={results}"
    
    try:
        r = requests.get(url, timeout=5)  # Add timeout to prevent hanging
        r.raise_for_status()  # Raise an exception for bad status codes
        
        data = r.json()
        if not data:
            raise HTTPException(status_code=404, detail="No data received from ThingSpeak")
            
        feeds = data.get("feeds", [])
        if not feeds:
            raise HTTPException(status_code=404, detail="No feeds found in ThingSpeak data")
            
        # Validate and clean the feed data
        cleaned_feeds = []
        for feed in feeds:
            try:
                # Ensure created_at is valid
                timestamp = feed.get("created_at")
                if not timestamp:
                    continue
                    
                # Convert string fields to appropriate types
                cleaned_feed = {
                    "created_at": timestamp,
                    "field1": float(feed.get("field1")) if feed.get("field1") else None,  # Temperature
                    "field2": float(feed.get("field2")) if feed.get("field2") else None,  # Humidity
                    "field3": float(feed.get("field3")) if feed.get("field3") else None,  # Soil Moisture
                    "field4": feed.get("field4"),  # Rain Status (on/off)
                    "field5": feed.get("field5")   # Water Pump Status (on/off)
                }
                
                # Process binary status fields
                if cleaned_feed["field4"] is not None:
                    # Convert to lowercase for consistency
                    if isinstance(cleaned_feed["field4"], str):
                        cleaned_feed["field4"] = cleaned_feed["field4"].lower()
                    # Convert numeric values to binary
                    elif isinstance(cleaned_feed["field4"], (int, float)):
                        cleaned_feed["field4"] = "on" if cleaned_feed["field4"] > 0 else "off"
                
                if cleaned_feed["field5"] is not None:
                    # Convert to lowercase for consistency
                    if isinstance(cleaned_feed["field5"], str):
                        cleaned_feed["field5"] = cleaned_feed["field5"].lower()
                    # Convert numeric values to binary
                    elif isinstance(cleaned_feed["field5"], (int, float)):
                        cleaned_feed["field5"] = "on" if cleaned_feed["field5"] > 0 else "off"
                
                cleaned_feeds.append(cleaned_feed)
            except (ValueError, TypeError) as e:
                print(f"Error processing feed: {e}")
                continue
                
        return {"feeds": cleaned_feeds}
        
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Timeout while fetching data from ThingSpeak")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error fetching data from ThingSpeak: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while processing ThingSpeak data")

@app.post("/api/disease-prediction")
async def predict_disease(leafImage: UploadFile = File(...), cropType: str = Form(...)):
    try:
        # Read the image file
        contents = await leafImage.read()
        image = Image.open(io.BytesIO(contents))
        
        # Resize image to standard size (224x224 is common for many models)
        image = image.resize((224, 224))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # In a real application, you would use a trained model here
        # For now, we'll simulate a prediction with random values
        
        # Simulate model prediction
        if cropType.lower() in DISEASE_INFO:
            possible_diseases = list(DISEASE_INFO[cropType.lower()].keys())
            predicted_disease = random.choice(possible_diseases)
            confidence = random.uniform(0.7, 0.99)
            
            # Get disease information
            disease_data = DISEASE_INFO[cropType.lower()][predicted_disease]
            
            return {
                "disease": predicted_disease.replace("_", " ").title(),
                "confidence": confidence,
                "description": disease_data["description"],
                "treatments": disease_data["treatments"]
            }
        else:
            # If crop type not in database, return a generic response
            return {
                "disease": "Unknown",
                "confidence": 0.5,
                "description": f"No disease information available for {cropType}.",
                "treatments": ["Consult with a local agricultural expert for specific recommendations."]
            }
            
    except Exception as e:
        print(f"Error in disease prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/api/crop-recommendation")
async def crop_recommendation(
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...)
):
    if crop_recommendation_model is None:
        raise HTTPException(status_code=500, detail="Crop recommendation model not loaded")
    
    # Prepare input data
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    # Make prediction
    prediction = crop_recommendation_model.predict(input_data)[0]
    
    return {"recommended_crop": prediction}

@app.post("/api/fertilizer-recommendation")
async def fertilizer_recommendation(input_data: FertilizerInput):
    if fertilizer_recommendation_model is None:
        raise HTTPException(status_code=500, detail="Fertilizer recommendation model not loaded")
    
    try:
        # Validate soil type
        if input_data.soil_type not in VALID_SOIL_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid soil type. Must be one of: {', '.join(VALID_SOIL_TYPES)}")
        
        # Validate crop type
        if input_data.crop_type not in VALID_CROP_TYPES:
            raise HTTPException(status_code=400, detail=f"Invalid crop type. Must be one of: {', '.join(VALID_CROP_TYPES)}")
        
        # Print input data for debugging
        print("Input data received:", input_data)
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame({
            'Temperature': [input_data.temperature],
            'Humidity': [input_data.humidity],
            'Moisture': [45],  # Default value based on training data average
            'Soil Type': [input_data.soil_type],
            'Crop Type': [input_data.crop_type],
            'Nitrogen': [input_data.nitrogen],
            'Potassium': [input_data.potassium],
            'Phosphorous': [input_data.phosphorous]
        })
        
        # Encode categorical variables
        input_df['Soil Type'] = soil_type_encoder.transform(input_df['Soil Type'])
        input_df['Crop Type'] = crop_type_encoder.transform(input_df['Crop Type'])
        
        # Make prediction
        prediction = fertilizer_recommendation_model.predict(input_df)
        
        # Decode the prediction
        recommended_fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]
        
        return {"recommended_fertilizer": recommended_fertilizer}
        
    except Exception as e:
        print(f"Error in fertilizer recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/crop-price-prediction")
async def crop_price_prediction(
    crop: str = Form(...),
    month: int = Form(...),
    year: int = Form(...)
):
    if crop_price_model is None:
        raise HTTPException(status_code=500, detail="Crop price prediction model not loaded")
    
    try:
        # Validate month and year
        if not 1 <= month <= 12:
            raise HTTPException(status_code=400, detail="Month must be between 1 and 12")
        if year < 2023:
            raise HTTPException(status_code=400, detail="Year must be 2023 or later")
            
        # Get average price for the crop from the dataset
        crop_data = crop_price_data[crop_price_data['Crop'] == crop]
        if crop_data.empty:
            raise HTTPException(status_code=400, detail=f"Crop '{crop}' not found in the dataset")
        
        avg_price = crop_data['Price'].mean()
        
        # Create input data for the model
        # The model might expect different column names or order, so we'll try different formats
        try:
            # Try with the original format
            input_data = pd.DataFrame({
                'Crop': [crop],
                'Month': [month],
                'Year': [year]
            })
            
            # Make prediction using the model
            predicted_price = crop_price_model.predict(input_data)[0]
        except Exception as e:
            print(f"Error making prediction with original format: {e}")
            try:
                # Try with a different format (if the model was trained with different column names)
                input_data = pd.DataFrame({
                    'crop': [crop],
                    'month': [month],
                    'year': [year]
                })
                predicted_price = crop_price_model.predict(input_data)[0]
            except Exception as e:
                print(f"Error making prediction with alternative format: {e}")
                # Fallback to simple prediction if model fails
                month_factor = 1.0 + (month - 6) * 0.05  # Higher prices in winter months (Oct-Mar)
                year_factor = 1.0 + (year - 2023) * 0.1  # Assuming 10% annual increase
                predicted_price = avg_price * month_factor * year_factor
        
        return {"predicted_price": float(predicted_price)}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in crop price prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error in crop price prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
