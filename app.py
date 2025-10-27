from flask import Flask, request, jsonify, render_template
from google import genai
from dotenv import load_dotenv
import json
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)

# Get API key from environment variable
API_KEY = os.environ.get('GEMINI_API_KEY', 'abcd')
logger.info(f"API Key loaded: {'Yes' if API_KEY and API_KEY != 'abcd' else 'No (using fallback)'}")

# Initialize Gemini client
client = None
if API_KEY and API_KEY != 'abcd':
    try:
        client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})
        logger.info("Gemini client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        client = None
else:
    logger.error("No valid API key found! Please set GEMINI_API_KEY in .env file")

# Load the dataset
with open('mental_health_data.json', 'r') as file:
    data = json.load(file)

training_data = []
for intent in data['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        for response in intent['responses']:
            training_data.append({
                'input': pattern,
                'output': response
            })

class FineTunedChat:
    def __init__(self, client, training_data):
        self.client = client
        self.training_data = training_data

    def get_response(self, user_input):
        for item in self.training_data:
            if user_input.lower() in item['input'].lower():
                return item['output']
        return None  # Return None if no exact match found
    
    def generate_gemini_response(self, user_input, instruction):
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=instruction + user_input,
            )
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return "I'm sorry, I'm having trouble connecting right now. Please try again later."

# Initialize variables for the new client approach
fine_tuned_chat = None

# Only try to initialize if we have a valid client
if client:
    try:
        fine_tuned_chat = FineTunedChat(client, training_data)
        logger.info("Fine-tuned chat initialized successfully with new client")
    except Exception as e:
        logger.error(f"Failed to initialize fine-tuned chat: {e}")
        fine_tuned_chat = None
else:
    logger.error("No valid client - Gemini features disabled")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/basic-screen')
def BasicScreen():
    return render_template('basicScreen.html')

@app.route('/depression')
def Depression():
    return render_template('info-depress.html')

@app.route('/anxiety')
def Anxiety():
    return render_template('info-anxiety.html')

@app.route('/autism')
def Autism():
    return render_template('info-autism.html')

@app.route('/test-depression')
def TestDepression():
    return render_template('test_dep.html')

@app.route('/test-anxiety')
def TestAnxiety():
    return render_template('test_anx.html')

@app.route('/test-stress')
def TestStress():
    return render_template('test_stress.html')

@app.route('/ChatBot')
def ChatBot():
    return render_template('bot_index.html')

@app.route('/message', methods=['POST'])
def message():
    user_input = request.json.get('message')
    logger.info(f"User Input: {user_input}")

    # Check if fine_tuned_chat is available
    if not fine_tuned_chat:
        logger.error("Fine-tuned chat not available")
        return jsonify({'reply': "I'm sorry, the AI service is currently unavailable. Please try again later."})

    response_text = fine_tuned_chat.get_response(user_input)

    if response_text:
        logger.info("Using response from training data")
        reply = response_text
    else:
        logger.info("No match in training data, calling Gemini API")
        instruction = '''You are a mental health specialist providing support in a conversational manner. When responding to users, aim to create a warm and empathetic interaction, much like a dialogue between a mental health professional and a patient. 

                1. **Show Empathy**: Acknowledge the user's feelings and concerns with empathy. Use supportive and reassuring language.
                2. **Be Concise and Relevant**: Provide clear, direct responses without overwhelming details. Focus on the user's immediate concerns and offer practical advice.
                3. **Encourage Open Dialogue**: Invite users to share more if they wish. Ask open-ended questions to understand their needs better, but avoid pushing them to share more than they're comfortable with.
                4. **Prioritize Emotional Safety**: Ensure responses are sensitive to the user's emotional state. If a topic is too complex or requires professional help, gently guide the user towards seeking support from a licensed professional.
                5. **Respectful Engagement**: Engage with users respectfully and maintain a professional tone, even if the conversation becomes challenging.

                Respond to the user's query accordingly.'''
        
        reply = fine_tuned_chat.generate_gemini_response(user_input, instruction)
        logger.info(f"Gemini API Response: {reply[:100]}...")
    
    logger.info(f"Bot Reply: {reply[:100]}...")
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)