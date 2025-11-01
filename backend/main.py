from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from ann_model import create_and_train_ann, load_ann_model, predict_calories
from chatbot import load_fitness_chatbot, generate_food_plan, generate_weekly_plan
import json

app = FastAPI(title="Fitness Recommendation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class UserInput(BaseModel):
    age: int
    height: float
    weight: float
    gender: int
    activity_level: int
    fitness_goal: int

class FitnessResponse(BaseModel):
    predicted_calories: int
    food_plan: str
    weekly_plan: str
    status: str

# Global variables for loaded models
ann_model = None
scaler = None
chatbot_loaded = False

@app.on_event("startup")
async def startup_event():
    global ann_model, scaler, chatbot_loaded
    try:
        ann_model, scaler = load_ann_model()
        print("ANN model loaded successfully")
    except:
        print("Training new ANN model...")
        ann_model, scaler = create_and_train_ann()
    
    chatbot_loaded = load_fitness_chatbot()
    if chatbot_loaded:
        print("Chatbot model loaded successfully")
    else:
        print("Chatbot model failed to load, using fallback responses")

@app.get("/")
async def root():
    return {"message": "Fitness Recommendation API is running!"}

@app.post("/predict-fitness-plan", response_model=FitnessResponse)
async def predict_fitness_plan(user_input: UserInput):
    try:
        # Step 1: Predict calories using ANN
        predicted_calories = predict_calories(ann_model, scaler, user_input.model_dump())

        # Step 2: Generate food plan using LLM - CORRECTED: removed extra parameter
        food_plan = generate_food_plan(
            predicted_calories, 
            user_input.fitness_goal
        )
        
        # Step 3: Generate weekly plan using LLM - CORRECTED: removed extra parameter
        weekly_plan = generate_weekly_plan(
            predicted_calories,
            user_input.fitness_goal
        )
        
        return FitnessResponse(
            predicted_calories=predicted_calories,
            food_plan=food_plan,
            weekly_plan=weekly_plan,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plan: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "ann_model_loaded": ann_model is not None, 
        "chatbot_loaded": chatbot_loaded
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)