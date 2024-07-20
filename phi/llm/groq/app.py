from flask import Flask, request, jsonify
from micro.llm.groq import Groq

app = Flask(__name__)

@app.route('/invoke', methods=['POST'])
def invoke_model():
    data = request.json
    messages = data.get('messages')
    groq_instance = Groq()  # Initialize your model here
    try:
        response = groq_instance.response(messages=messages)
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
