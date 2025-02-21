import streamlit as st
import json
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat

# Load dataset
with open('databot.json') as f:
    dataset = json.load(f)

# Fungsi normalisasi teks
def clean_text(text):
    text = text.lower()  # Konversi ke huruf kecil
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca kecuali spasi
    return text

# Extract patterns and responses (dengan normalisasi)
patterns = []
responses = []
for intent in dataset["intents"]:
    for pattern in intent["patterns"]:
        cleaned_pattern = clean_text(pattern)  # Bersihkan setiap pattern
        patterns.append(cleaned_pattern)
        responses.append(" ".join(intent["responses"]))

# Train Word2Vec model dengan teks yang sudah dibersihkan
sentences = [pattern.split() for pattern in patterns]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def get_sentence_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# Precompute sentence vectors
pattern_vectors = np.array([get_sentence_vector(pattern, word2vec_model) for pattern in patterns])


def find_top_similar_responses(user_input, top_n=3):
    user_input = clean_text(user_input)  # Normalisasi input
    user_vector = get_sentence_vector(user_input, word2vec_model).reshape(1, -1)
    similarities = cosine_similarity(user_vector, pattern_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [responses[i] for i in top_indices]

# Function to generate a response using Ollama with streaming
def generate_response_with_ollama(user_input):
    user_input = clean_text(user_input)  # Normalisasi input
    top_responses = find_top_similar_responses(user_input)
    prompt = f"""
### instruksi:
berikut adalah data yang kamu miliki:
{chr(10).join(f"- {resp}" for resp in top_responses)}

berikan jawaban dari pertanyaan berdasarkan data yang kamu miliki. Jangan membuat jawaban sendiri jika tidak ada di data.

{user_input}
"""
    print(prompt)

    response = chat(
        model='adijayainc/bhsa-llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )
    
    for chunk in response:
        yield chunk['message']['content']

# Streamlit App
st.title("Chatbot Kapal")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Apa yang bisa saya bantu?")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from Ollama and stream it
    response_container = st.chat_message("assistant")
    response_text = ""
    with response_container:
        response_area = st.empty()
        for chunk in generate_response_with_ollama(prompt):
            response_text += chunk
            response_area.markdown(response_text)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})