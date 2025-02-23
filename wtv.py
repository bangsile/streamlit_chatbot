from gensim.models import Word2Vec

# Contoh data: beberapa kalimat tokenized
sentences = [
    ["saya", "suka", "belajar", "nlp"],
    ["nlp", "menggunakan", "word2vec"],
    ["word2vec", "memiliki", "cbow", "dan", "skip-gram"]
]

# Membuat model CBOW
model = Word2Vec(sentences, vector_size=3, window=2, min_count=1, sg=0)  # sg=0 untuk CBOW

# Melihat vektor kata "nlp"
print("vektor kata 'nlp': ", model.wv["nlp"])

# Melihat kata yang paling mirip dengan "nlp"
print(model.wv.most_similar("nlp"))
