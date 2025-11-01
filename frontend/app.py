import streamlit as st
import requests
import json
import pandas as pd
from io import StringIO

# Configure the page
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="💪",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .calorie-display {
        font-size: 2rem;
        font-weight: bold;
        color: #2ca02c;
        text-align: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .plan-box {
        padding: 1rem;
        background-color: #fafafa;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:8000"

def check_backend_connection():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

def get_fitness_recommendation(user_data):
    try:
        response = requests.post(
            f"{API_URL}/predict-fitness-plan",
            json=user_data
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from backend: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">💪 AI-Powered Fitness Coach</div>', unsafe_allow_html=True)
    
    # Check backend connection
    if not check_backend_connection():
        st.error("⚠️ Backend server is not running. Please start the FastAPI server first.")
        st.info("Run this command in your backend directory: uvicorn main:app --reload")
        return
    
    # Sidebar for user input
    st.sidebar.header("🏃‍♂️ Your Profile")
    
    with st.sidebar.form("user_input_form"):
        age = st.slider("Age", min_value=18, max_value=80, value=25)
        height = st.slider("Height (cm)", min_value=140, max_value=220, value=175)
        weight = st.slider("Weight (kg)", min_value=40, max_value=150, value=70)
        gender = st.radio("Gender", options=["Male", "Female"])
        activity_level = st.select_slider(
            "Activity Level",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {
                1: "Sedentary", 
                2: "Lightly Active", 
                3: "Moderately Active", 
                4: "Very Active", 
                5: "Extremely Active"
            }[x]
        )
        fitness_goal = st.radio(
            "Fitness Goal",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "Weight Loss", 
                1: "Maintain Weight", 
                2: "Muscle Gain"
            }[x]
        )
        
        submitted = st.form_submit_button("Generate Fitness Plan")
    
    # Main content area
    if submitted:
        # Prepare user data
        user_data = {
            "age": age,
            "height": height,
            "weight": weight,
            "gender": 0 if gender == "Male" else 1,
            "activity_level": activity_level,
            "fitness_goal": fitness_goal
        }
        
        with st.spinner("🤖 AI is generating your personalized fitness plan..."):
            result = get_fitness_recommendation(user_data)
        
        if result and result["status"] == "success":
            # Display calorie prediction
            st.markdown('<div class="section-header">🎯 Your Daily Calorie Target</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="calorie-display">{result["predicted_calories"]} kcal/day</div>', unsafe_allow_html=True)
            
            # Display food plan
            st.markdown('<div class="section-header">🍽️ Your Daily Food Plan</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="plan-box">{result["food_plan"]}</div>', unsafe_allow_html=True)
            
            # Display weekly plan
            st.markdown('<div class="section-header">📅 Your Weekly Fitness Plan</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="plan-box">{result["weekly_plan"]}</div>', unsafe_allow_html=True)
            
            # Additional information
            with st.expander("💡 How this plan was generated"):
                st.markdown("""
                **AI-Powered Recommendations:**
                - **Calorie Prediction**: Artificial Neural Network trained on fitness data
                - **Food Plan**: Generated using advanced language models
                - **Weekly Plan**: Adaptive planning based on your goals
                
                **Note**: This is an AI-generated recommendation. Consult with healthcare professionals before starting any new fitness program.
                """)
        
        else:
            st.error("Failed to generate fitness plan. Please try again.")
    
    else:
        # Welcome message
        st.markdown("""
        ## Welcome to Your AI Fitness Coach! 🏋️‍♀️
        
        This intelligent system uses advanced AI to create personalized fitness plans just for you:
        
        - **🤖 Neural Networks** for accurate calorie prediction
        - **💬 Language Models** for personalized meal and workout plans
        - **📊 Adaptive Planning** that evolves with your progress
        
        ### How it works:
        1. Fill out your profile in the sidebar
        2. Click 'Generate Fitness Plan'
        3. Get your AI-powered personalized plan!
        
        ### Ready to start your fitness journey?
        Fill out the form on the left and let AI guide you to your goals! 🎯
        """)
        
        # Sample output preview
        with st.expander("👀 See a sample output"):
            st.markdown("""
            **Sample Calorie Target:** 2800 kcal/day
            
            **Sample Food Plan:**
            | Meal | Food Items | Calories | Protein | Carbs | Fat |
            | Breakfast | Oats + banana + milk | 400 | 18g | 60g | 8g |
            | Lunch | Brown rice + grilled chicken + veggies | 700 | 45g | 80g | 20g |
            
            **Sample Weekly Plan:**
            For Week 2, increase daily intake to 2940 kcal (+5%). Add one additional strength workout focusing on compound lifts.
            """)

if __name__ == "__main__":
    main()