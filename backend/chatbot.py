from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# Global variable for the generator
generator = None

def load_fitness_chatbot():
    """Load a text generation model"""
    global generator
    try:
        print("Loading chatbot model...")
        
        # Use distilgpt2 for better compatibility
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            max_new_tokens=300,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=50256
        )
        print("✓ Chatbot model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        generator = None
        return False

def generate_food_plan(calories, fitness_goal):
    """Generate food plan using LLM - ONLY send calories"""
    goal_text = {0: "weight loss", 1: "maintenance", 2: "muscle gain"}
    
    if generator is None:
        return f"Error: Chatbot model not loaded. Basic meal plan for {calories} calories ({goal_text[fitness_goal]}):\n- Breakfast: Balanced meal\n- Lunch: Protein + vegetables\n- Dinner: Light meal with protein"
    
    try:
        # ONLY send calories - let LLM generate everything
        prompt = f"Create a detailed daily meal plan for {calories} calories for {goal_text[fitness_goal]} with specific food items and portions:"
        
        print(f"Generating food plan for {calories} calories, goal: {goal_text[fitness_goal]}")
        
        response = generator(
            prompt,
            max_new_tokens=350,
            num_return_sequences=1
        )[0]['generated_text']
        
        # Extract only the newly generated part (after the prompt)
        if prompt in response:
            response = response.split(prompt)[1].strip()
        
        print(f"Generated food plan: {response}")
        return response
            
    except Exception as e:
        print(f"Error in LLM food generation: {e}")
        return f"Daily meal plan for {calories} calories ({goal_text[fitness_goal]}):\n- Breakfast: Oatmeal with fruits\n- Lunch: Grilled chicken with vegetables\n- Dinner: Fish with quinoa and salad\n- Snacks: Greek yogurt, nuts"

def generate_weekly_plan(calories, fitness_goal):
    """Generate weekly plan using LLM - ONLY send calories"""
    goal_text = {0: "weight loss", 1: "maintenance", 2: "muscle gain"}
    
    if generator is None:
        return f"Error: Chatbot model not loaded. Basic weekly plan for {calories} calories ({goal_text[fitness_goal]}):\nMonday: Cardio + Strength\nTuesday: Rest day\nWednesday: Full body workout\nThursday: Active recovery\nFriday: Strength training\nSaturday: Cardio\nSunday: Rest"
    
    try:
        # ONLY send calories - let LLM generate everything
        prompt = f"Create a detailed weekly fitness plan for {calories} calories for {goal_text[fitness_goal]} with daily workouts and activities:"
        
        print(f"Generating weekly plan for {calories} calories, goal: {goal_text[fitness_goal]}")
        
        response = generator(
            prompt,
            max_new_tokens=350,
            num_return_sequences=1
        )[0]['generated_text']
        
        # Extract only the newly generated part (after the prompt)
        if prompt in response:
            response = response.split(prompt)[1].strip()
        
        print(f"Generated weekly plan: {response}")
        return response
            
    except Exception as e:
        print(f"Error in LLM weekly generation: {e}")
        return f"Weekly fitness plan for {calories} calories ({goal_text[fitness_goal]}):\nMonday: Cardio (30min) + Upper body\nTuesday: Lower body strength\nWednesday: Rest or light yoga\nThursday: HIIT workout\nFriday: Full body strength\nSaturday: Outdoor activity\nSunday: Rest and recovery"