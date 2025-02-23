import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Parameter
vector_size = 10  # Ukuran vektor diperkecil agar lebih mudah dilihat
window_size = 2
negative = 5
epochs = 100

# Data yang digunakan
sentences = ["kapan kapal aksar 09 berangkat", 
             "kapan aksar 9 berangkat", 
             "kapan kapal aksar 9 berangkat"]

# Tokenisasi menjadi daftar kata-kata
tokenized_sentences = [sentence.split() for sentence in sentences]

# Membuat vocab
word_counts = defaultdict(int)
for sentence in tokenized_sentences:
    for word in sentence:
        word_counts[word] += 1

vocab = list(word_counts.keys())
vocab_size = len(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

# Inisialisasi bobot manual dengan seed agar tetap sama
np.random.seed(42)
W1 = np.random.uniform(-0.5/vector_size, 0.5/vector_size, (vocab_size, vector_size))

# Train dengan Gensim
gensim_model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=window_size, 
                         negative=negative, sg=0, hs=0, sample=0, min_count=1, epochs=epochs)

# Masukkan bobot manual ke model Gensim agar identik
for word in vocab:
    gensim_model.wv[word] = W1[word2idx[word]]

# Fungsi untuk mendapatkan vektor sebuah kata
def get_word_vector(word, model_type="manual"):
    if word in vocab:
        return W1[word2idx[word]] if model_type == "manual" else gensim_model.wv[word]
    return np.zeros(vector_size)

# Fungsi mendapatkan vektor rata-rata dari sebuah kalimat
def get_sentence_vector(sentence, model_type="manual"):
    vectors = [get_word_vector(word, model_type) for word in sentence if word in vocab]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

# Fungsi menghitung cosine similarity secara manual
def cosine_similarity_manual(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Data uji baru
test_sentence = ["kapal", "aksar", "09", "berangkat", "kapan"]

# Menampilkan vektor setiap kata
print("\n===== Vektor Kata =====")
for word in vocab:
    vec_manual = get_word_vector(word, "manual")
    vec_gensim = get_word_vector(word, "gensim")
    print(f"Kata: {word}")
    print(f"  Manual  : {vec_manual}")
    print(f"  Gensim  : {vec_gensim}")
    print("-" * 50)

# Menampilkan vektor rata-rata setiap kalimat
print("\n===== Vektor Kalimat =====")
for sentence in tokenized_sentences:
    vec_manual = get_sentence_vector(sentence, "manual")
    vec_gensim = get_sentence_vector(sentence, "gensim")
    print(f"Kalimat: {' '.join(sentence)}")
    print(f"  Manual  : {vec_manual}")
    print(f"  Gensim  : {vec_gensim}")
    print("-" * 50)

# Menampilkan vektor rata-rata untuk test_sentence
vec_test_manual = get_sentence_vector(test_sentence, "manual")
vec_test_gensim = get_sentence_vector(test_sentence, "gensim")

print("\n===== Vektor Test Sentence =====")
print(f"Kalimat: {' '.join(test_sentence)}")
print(f"  Manual  : {vec_test_manual}")
print(f"  Gensim  : {vec_test_gensim}")
print("-" * 50)

# Menampilkan Cosine Similarity
print("\n===== Cosine Similarity (Word2Vec Manual & Cosine Similarity Manual) =====")
for sentence in tokenized_sentences:
    vec1_manual = get_sentence_vector(sentence, "manual")
    sim_manual = cosine_similarity_manual(vec1_manual, vec_test_manual)
    print(f"Similarity antara '{' '.join(sentence)}' dan '{' '.join(test_sentence)}' (Manual): {sim_manual:.4f}")

print("\n===== Cosine Similarity (Word2Vec Gensim & Scikit-Learn) =====")
for sentence in tokenized_sentences:
    vec1_gensim = get_sentence_vector(sentence, "gensim")
    sim_sklearn = cosine_similarity([vec1_gensim], [vec_test_gensim])[0][0]
    print(f"Similarity antara '{' '.join(sentence)}' dan '{' '.join(test_sentence)}' (Gensim + Scikit-Learn): {sim_sklearn:.4f}")