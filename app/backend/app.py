from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load Sentence Transformer model
transformers_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
movies = pickle.load(open('../data/movie_list.pkl', 'rb'))
description_embedding = pickle.load(open('../data/descriptions_embeddings.pkl', 'rb'))
metadata_embedding = pickle.load(open('../data/metadata_embeddings.pkl', 'rb'))

# Route to receive prompt and embed it
@app.route('/embed', methods=['POST'])
def embed_prompt():
    data = request.json
    prompt = data['prompt'].lower()

    # Embed the prompt using Sentence Transformer
    prompt_embedding = transformers_model.encode([prompt])
    movie_embedding = description_embedding + metadata_embedding
    similarity_scores = np.matmul(movie_embedding, np.transpose(prompt_embedding))
    
    return jsonify({"recommended_movie_ids": recommend(similarity_scores)
})

def recommend(similarity_scores):
    distances = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])
    recommended_movie_ids = []

    for index, item in distances[:5]:
        recommended_movie_ids.append(index)
            
    return recommended_movie_ids

if __name__ == "__main__":
    app.run(debug=True)
