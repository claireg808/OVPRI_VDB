# receive response from the LLM

import json
from flask import Flask, request, jsonify
from rag import answer_query

app = Flask(__name__)
history = []

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    history.append(user_input)

    response, log_entry = answer_query(
        query=user_input,
        history=history[:-1]
    )

    with open('logs/rag_logs.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')

    return jsonify({'response': response})

# add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response