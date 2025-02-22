from flask import Flask, request, jsonify, render_template
import os
import PyPDF2
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Initialize SentenceTransformer for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2') 

# FAISS setup
index = faiss.IndexFlatL2(384) 
pdf_texts = []

# Hugging Face API setup
HF_API_KEY = 'hf_api key' 
HF_MODEL_NAME = 'mistral-7b-instruct-v0.3'
HF_API_URL = f'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3'

def extract_pdf_text(file_path):
    
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_pdf_embeddings(text):
    
    embeddings = embedding_model.encode([text])
    return embeddings

def generate_answer_from_hf_api(question, context):
    
    headers = {
        'Authorization': f'Bearer {HF_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    
    prompt = f"Question: {question} Context: {context}"

    
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": 200}
    }
    
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        generated_text = response.json()[0]['generated_text']
        
        answer = generated_text.split("Answer:")[-1].strip()  
        return answer
    else:
        return "Error generating response from Hugging Face API."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400
    file = request.files['pdf']
    question = request.form['question']

    
    pdf_path = os.path.join('uploads', file.filename)
    file.save(pdf_path)
    pdf_text = extract_pdf_text(pdf_path)
    pdf_embeddings = get_pdf_embeddings(pdf_text)

    # Store the embeddings in FAISS
    pdf_texts.append(pdf_text)
    index.add(np.array(pdf_embeddings).astype('float32'))

    # Search for the most relevant content in the stored PDF
    question_embeddings = embedding_model.encode([question])
    _, I = index.search(np.array(question_embeddings).astype('float32'), k=1)

    # Get the relevant text from the PDF
    relevant_text = pdf_texts[I[0][0]]
    
    # Generate an answer using the Hugging Face API
    generated_answer = generate_answer_from_hf_api(question, relevant_text)

    return jsonify({"answer": generated_answer})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000)
