import gensim
import numpy as np
from nltk import word_tokenize



def word_embedding(sentences):
    # Word Embedding modelini yükleyin
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin')
    word_embedding_model = gensim.models.KeyedVectors.load('GoogleNews-vectors-negative300.bin')

    embedded_sentences = []
    for sentence in sentences:
        # Cümlenin kelimelerini vektörlere dönüştürün
        tokens = word_tokenize(sentence.lower())
        word_vectors = [word_embedding_model[word] for word in tokens if word in word_embedding_model.vocab]

        # Cümlenin vektörünü hesaplayın (kelime vektörlerinin toplamı)
        sentence_vector = np.sum(word_vectors, axis=0)

        # Gömülü cümleyi listeye ekleyin
        embedded_sentences.append(sentence_vector)

    return embedded_sentences