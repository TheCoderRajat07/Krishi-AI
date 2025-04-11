import joblib
from sklearn.preprocessing import LabelEncoder

# Create encoders for each categorical variable
soil_type_encoder = LabelEncoder()
crop_type_encoder = LabelEncoder()
fertilizer_encoder = LabelEncoder()

# Fit the encoders with the known categories
soil_types = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
crop_types = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 
              'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat', 'coffee', 
              'kidneybeans', 'orange', 'pomegranate', 'rice', 'watermelon']
fertilizer_types = ['10-10-10', '10-26-26', '14-14-14', '14-35-14', '15-15-15', 
                   '17-17-17', '20-20', '28-28', 'DAP', 'Potassium chloride', 
                   'Potassium sulfate.', 'Superphosphate', 'TSP', 'Urea']

# Fit the encoders
soil_type_encoder.fit(soil_types)
crop_type_encoder.fit(crop_types)
fertilizer_encoder.fit(fertilizer_types)

# Save the encoders
joblib.dump(soil_type_encoder, 'models/soil_type_encoder.joblib')
joblib.dump(crop_type_encoder, 'models/crop_type_encoder.joblib')
joblib.dump(fertilizer_encoder, 'models/fertilizer_encoder.joblib')

print("Encoders have been created and saved successfully!") 