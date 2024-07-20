from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from phi.assistant import Assistant

app = Flask(__name__)

# Import the necessary components from your assistant module
from assistant import get_auto_rag_assistant  # type: ignore

# Setup for file uploads
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize variables to hold the assistant instance and its configuration
assistant_instance: Assistant = None
llm_model_global = None
embeddings_model_global = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/setup', methods=['POST'])
def setup_assistant():
    global assistant_instance, llm_model_global, embeddings_model_global
    data = request.json
    llm_model = data.get('llm_model')
    embeddings_model = data.get('embeddings_model')
    llm_model_global = llm_model
    embeddings_model_global = embeddings_model
    assistant_instance = get_auto_rag_assistant(llm_model=llm_model, embeddings_model=embeddings_model)
    return jsonify({"message": "Assistant setup with LLM model: {} and embeddings model: {}".format(llm_model, embeddings_model)}), 200

@app.route('/add-knowledge', methods=['POST'])
def upload_pdf():
    global assistant_instance
    # Check if the assistant is configured
    if not assistant_instance:
        return jsonify({"error": "Assistant is not configured. Please set up first."}), 400
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the file to add it to the knowledge base
        from micro.document.reader.pdf import PDFReader
        reader = PDFReader()
        documents = reader.read(filepath)
        if documents:
            assistant_instance.knowledge_base.load_documents(documents, upsert=True)
            return jsonify({"message": "File uploaded and processed successfully", "filename": filename}), 200
        else:
            return jsonify({"error": "Failed to process the PDF"}), 500
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/query', methods=['POST'])
def ask_question():
    global assistant_instance
    # Check if the assistant is configured
    if not assistant_instance:
        return jsonify({"error": "Assistant is not configured. Please set up first."}), 400
    
    data = request.json
    question = data.get('question')
    
    # Check if a question was provided
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # Assuming `run()` method takes a question and returns a response.
        # This might need to be adjusted if your assistant's API differs.
        response = assistant_instance.run(question, stream=False)
        return jsonify({"answer": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_knowledge_base', methods=['POST'])
def clear_knowledge_base():
    global assistant_instance
    # Check if the assistant is configured
    if not assistant_instance:
        return jsonify({"error": "Assistant is not configured. Please set up first."}), 400
    
    try:
        # Assuming the knowledge base has a clear method
        assistant_instance.knowledge_base.vector_db.clear()
        return jsonify({"message": "Knowledge base cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5003)


