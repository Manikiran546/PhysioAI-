import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import json
import os

def create_synthetic_data(n_samples=10000):
    """Generate synthetic fitness data for training"""
    np.random.seed(42)
    
    # Generate synthetic data
    age = np.random.randint(18, 65, n_samples)
    height = np.random.randint(150, 200, n_samples)
    weight = np.random.randint(45, 120, n_samples)
    gender = np.random.randint(0, 2, n_samples)
    activity_level = np.random.randint(1, 6, n_samples)
    fitness_goal = np.random.randint(0, 3, n_samples)
    
    # Calculate TDEE using Mifflin-St Jeor Equation + activity multiplier
    # Fixed: Use vectorized operations instead of conditional on arrays
    bmr_male = (10 * weight) + (6.25 * height) - (5 * age) + 5
    bmr_female = (10 * weight) + (6.25 * height) - (5 * age) - 161
    
    # Use gender to select the appropriate BMR
    bmr = np.where(gender == 0, bmr_male, bmr_female)
    
    activity_multipliers = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    tdee = []
    
    for i in range(n_samples):
        multiplier = activity_multipliers[activity_level[i]]
        base_calories = bmr[i] * multiplier
        
        # Adjust based on fitness goal
        if fitness_goal[i] == 0:  # Weight loss
            calories = base_calories * 0.85
        elif fitness_goal[i] == 1:  # Maintain
            calories = base_calories
        else:  # Muscle gain
            calories = base_calories * 1.15
            
        tdee.append(calories + np.random.normal(0, 50))  # Add noise
    
    # Prepare features and target
    X = np.column_stack([age, height, weight, gender, activity_level, fitness_goal])
    y = np.array(tdee)
    
    return X, y

def create_and_train_ann():
    """Create and train the ANN model"""
    print("Generating synthetic training data...")
    X, y = create_synthetic_data(10000)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Build ANN model
    print("Building ANN model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("Training model...")
    # Train model with fewer epochs for faster testing
    history = model.fit(
        X_scaled, 
        y, 
        epochs=50,  # Reduced from 100 to 50 for faster training
        batch_size=32, 
        validation_split=0.2, 
        verbose=1
    )
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    print("Saving model...")
    model.save('models/calorie_predictor.h5')
    print("Model saved as 'models/calorie_predictor.h5'")
    
    # Save scaler parameters
    scaler_info = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    
    with open('models/scaler_params.json', 'w') as f:
        json.dump(scaler_info, f)
    print("Scaler parameters saved as 'models/scaler_params.json'")
    
    # Print final training metrics
    final_loss = history.history['loss'][-1]
    final_mae = history.history['mae'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_mae = history.history['val_mae'][-1]
    
    print(f"\nTraining completed!")
    print(f"Final Training Loss (MSE): {final_loss:.2f}")
    print(f"Final Training MAE: {final_mae:.2f}")
    print(f"Final Validation Loss (MSE): {final_val_loss:.2f}")
    print(f"Final Validation MAE: {final_val_mae:.2f}")
    
    return model, scaler

def load_ann_model():
    """Load the trained ANN model and scaler"""
    try:
        model = tf.keras.models.load_model('models/calorie_predictor.h5')
        
        with open('models/scaler_params.json', 'r') as f:
            scaler_info = json.load(f)
        
        scaler = StandardScaler()
        scaler.mean_ = np.array(scaler_info['mean'])
        scaler.scale_ = np.array(scaler_info['scale'])
        
        print("ANN model and scaler loaded successfully")
        return model, scaler
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating new model...")
        return create_and_train_ann()

def predict_calories(model, scaler, input_data):
    """Predict daily calorie needs"""
    input_array = np.array([[
        input_data['age'],
        input_data['height'], 
        input_data['weight'],
        input_data['gender'],
        input_data['activity_level'],
        input_data['fitness_goal']
    ]])
    
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled, verbose=0)[0][0]
    
    return max(1200, round(prediction))  # Ensure minimum calories

# Test the model when run directly
if __name__ == "__main__":
    print("=== ANN Model Training Script ===")
    print("This script will create and train a new ANN model for calorie prediction.")
    
    # Create and train model
    model, scaler = create_and_train_ann()
    
    # Test prediction
    test_input = {
        'age': 25,
        'height': 175,
        'weight': 70,
        'gender': 0,
        'activity_level': 3,
        'fitness_goal': 2
    }
    
    print("\n=== Testing Model ===")
    calories = predict_calories(model, scaler, test_input)
    print(f"Test prediction for 25yo male, 175cm, 70kg, active, muscle gain: {calories} kcal")
    
    print("\n=== Model Summary ===")
    model.summary()
    
    # Test multiple scenarios
    print("\n=== Testing Different Scenarios ===")
    test_cases = [
        {'age': 30, 'height': 160, 'weight': 55, 'gender': 1, 'activity_level': 2, 'fitness_goal': 0},
        {'age': 40, 'height': 180, 'weight': 85, 'gender': 0, 'activity_level': 4, 'fitness_goal': 1},
        {'age': 22, 'height': 170, 'weight': 65, 'gender': 1, 'activity_level': 5, 'fitness_goal': 2}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        calories = predict_calories(model, scaler, test_case)
        gender_str = "Male" if test_case['gender'] == 0 else "Female"
        goal_str = {0: "Weight Loss", 1: "Maintenance", 2: "Muscle Gain"}[test_case['fitness_goal']]
        activity_str = {1: "Sedentary", 2: "Light", 3: "Moderate", 4: "Active", 5: "Very Active"}[test_case['activity_level']]
        
        print(f"Case {i}: {test_case['age']}yo {gender_str}, {test_case['height']}cm, {test_case['weight']}kg, {activity_str}, {goal_str}: {calories} kcal")