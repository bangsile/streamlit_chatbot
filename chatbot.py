import streamlit as st
import json
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from ollama import chat
from datetime import datetime

# Load dataset chatbot (termasuk jadwal kapal)
with open('databot.json') as f:
    dataset = json.load(f)

# Mapping hari dalam bahasa Inggris ke bahasa Indonesia
hari_mapping = {
    "monday": "senin", "tuesday": "selasa", "wednesday": "rabu",
    "thursday": "kamis", "friday": "jumat", "saturday": "sabtu",
    "sunday": "minggu"
}

# Ambil hari ini dalam format bahasa Indonesia
hari_ini_en = datetime.today().strftime('%A').lower()
hari_ini_id = hari_mapping.get(hari_ini_en, "senin")  # Default ke Senin jika tidak ditemukan

# Fungsi untuk mengganti "hari ini" dan "malam ini" dengan waktu yang sesuai
def replace_time_keywords(user_input):
    user_input = user_input.lower()

    # Ganti "hari ini" dengan hari dari sistem
    user_input = user_input.replace("hari ini", f"hari {hari_ini_id}")

    # Ganti "malam ini" dengan hari + waktu yang sesuai
    user_input = user_input.replace("pagi ini", f"hari {hari_ini_id} pagi")
    user_input = user_input.replace("siang ini", f"hari {hari_ini_id} siang")
    user_input = user_input.replace("sore ini", f"hari {hari_ini_id} sore")
    user_input = user_input.replace("malam ini", f"hari {hari_ini_id} malam")

    return user_input

# **Buat dictionary pattern → response**
pattern_response_map = {}

for intent in dataset["intents"]:
    for pattern in intent["patterns"]:
        pattern_response_map[pattern.lower()] = intent["responses"]

def get_response_from_pattern(user_input):
    user_input = replace_time_keywords(user_input)  # **Ganti "hari ini" dan "malam ini" dengan waktu spesifik**

    # Pisahkan hari dan waktu dari pertanyaan pengguna
    words = user_input.split()
    hari = None
    waktu = None
    for word in words:
        if word in hari_mapping.values():  # Cek apakah ada nama hari
            hari = word
        if word in ["pagi", "siang", "sore", "malam"]:  # Cek apakah ada waktu
            waktu = word
    
    # Pastikan ada hari dan waktu
    if not hari or not waktu:
        return None  # Tidak bisa menentukan jadwal tanpa hari dan waktu yang jelas

    # Buat kembali pola yang dicari
    search_pattern = f"jadwal kapal hari {hari} {waktu}"

    # Cek apakah ada pola yang cocok secara eksak
    for pattern, responses in pattern_response_map.items():
        if pattern == search_pattern:
            return "\n".join(responses)

    return "⚠️ Maaf, tidak ada jadwal kapal untuk waktu tersebut."  # Jika tidak ditemukan


# Load pre-trained Word2Vec model
word2vec_model = Word2Vec.load("word2vec_model.bin")

def get_sentence_vector(sentence, model):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# Precompute sentence vectors untuk dataset
patterns = list(pattern_response_map.keys())
pattern_vectors = np.array([get_sentence_vector(pattern, word2vec_model) for pattern in patterns])

def find_top_similar_responses(user_input, top_n=3):
    user_vector = get_sentence_vector(user_input, word2vec_model).reshape(1, -1)
    similarities = cosine_similarity(user_vector, pattern_vectors).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [pattern_response_map[patterns[i]] for i in top_indices]

# Function to generate a response
def generate_response(user_input):
    user_input = replace_time_keywords(user_input)  # **Ganti "hari ini" dan "malam ini" dengan yang sesuai**

    # **Cek apakah ada pattern yang cocok lebih dulu**
    response = get_response_from_pattern(user_input)
    if response:
        return response  # Jika ada, langsung return response

    # Jika tidak ada pattern yang cocok, gunakan Word2Vec dan Ollama
    top_responses = find_top_similar_responses(user_input)
    if not top_responses:
        return "⚠️ Maaf, saya tidak memiliki informasi terkait pertanyaan Anda."

    prompt = f"""
### Instruksi:
Berikut adalah data yang kamu miliki:
{chr(10).join(f"- {resp}" for resp in top_responses)}

Berikan jawaban dari pertanyaan berdasarkan data yang kamu miliki. Jangan membuat jawaban sendiri jika tidak ada di data.

{user_input}
"""
    print(prompt)

    response = chat(
        model='adijayainc/bhsa-llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    response_text = ""
    for chunk in response:
        response_text += chunk['message']['content']
    
    return response_text

# Streamlit App
st.title("Chatbot Pelayanan (PT. Aksar Saputra Lines)")

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

    # Get response and display it
    response_text = generate_response(prompt)

    response_container = st.chat_message("assistant")
    with response_container:
        st.markdown(response_text)

    # Save assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
