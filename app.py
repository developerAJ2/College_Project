from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import json
import os

app = Flask(__name__)

# Get API key from environment variable
API_KEY = os.environ.get('GEMINI_API_KEY', 'abcd')  # Fallback for development
genai.configure(api_key=API_KEY)

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
    def __init__(self, model, training_data):
        self.model = model
        self.training_data = training_data

    def get_response(self, user_input):
        for item in self.training_data:
            if user_input.lower() in item['input'].lower():
                return item['output']
        return None  # Return None if no exact match found

# Initialize the Google Gemini LLM
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

fine_tuned_chat = FineTunedChat(model, training_data)

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
    response_text = fine_tuned_chat.get_response(user_input)

    if response_text:
        reply = response_text
    else:
        instruction = '''You are a mental health specialist providing support in a conversational manner. When responding to users, aim to create a warm and empathetic interaction, much like a dialogue between a mental health professional and a patient. 

                1. **Show Empathy**: Acknowledge the user's feelings and concerns with empathy. Use supportive and reassuring language.
                2. **Be Concise and Relevant**: Provide clear, direct responses without overwhelming details. Focus on the user's immediate concerns and offer practical advice.
                3. **Encourage Open Dialogue**: Invite users to share more if they wish. Ask open-ended questions to understand their needs better, but avoid pushing them to share more than they're comfortable with.
                4. **Prioritize Emotional Safety**: Ensure responses are sensitive to the user's emotional state. If a topic is too complex or requires professional help, gently guide the user towards seeking support from a licensed professional.
                5. **Respectful Engagement**: Engage with users respectfully and maintain a professional tone, even if the conversation becomes challenging.

                Respond to the user's query accordingly.'''
        response = chat.send_message(instruction + user_input)
        reply = response.text

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)