
import sys

import nltk
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QSlider, QVBoxLayout, QComboBox, \
    QTextEdit, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import matplotlib.pyplot as plt
from nltk.cluster import cosine_distance
from nltk.tokenize import sent_tokenize
from nltk.metrics import edit_distance, jaccard_distance
import gensim
import numpy as np
import gensim.downloader as api
import re
from collections import Counter

from math import log10
from numpy.linalg import norm
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
import math
import re
from collections import Counter
from rouge import Rouge

WORD = re.compile(r"\w+")

ps = PorterStemmer()

class DocumentUploader(QMainWindow):


    def __init__(self):
        super().__init__()
        self.threshold = 0.5  # Varsayılan threshold değeri
        self.initUI()
        # Word Embedding modelini yükleyin
        self.word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


    def initUI(self):
        self.setWindowTitle('Document Uploader')
        self.setGeometry(300, 300, 600, 400)

        self.upload_button = QPushButton('Doküman Yükle', self)
        self.upload_button.clicked.connect(self.showFileDialog)

        self.graph_label = QLabel('Doküman Grafi:', self)
        #self.graph_text = QTextEdit(self)
        #self.graph_text.setReadOnly(True)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        self.summary_label = QLabel('Özet:', self)
        self.summary_text = QTextEdit(self)
        self.summary_text.setReadOnly(True)

        self.rouge_score_label = QLabel('Rouge Skoru:', self)
        self.rouge_score_label_text = QTextEdit(self)
        self.rouge_score_label_text.setReadOnly(True)

        self.similarity_threshold_label = QLabel('Cümle Benzerliği Threshold:', self)
        self.similarity_threshold_slider = QSlider(Qt.Horizontal)
        self.similarity_threshold_slider.setMinimum(0)
        self.similarity_threshold_slider.setMaximum(100)
        self.similarity_threshold_slider.setValue(int(self.threshold * 100))
        self.similarity_threshold_slider.valueChanged.connect(self.updateSimilarityThreshold)

        self.score_threshold_label = QLabel('Cümle Skor Threshold:', self)
        self.score_threshold_slider = QSlider(Qt.Horizontal)
        self.score_threshold_slider.setMinimum(0)
        self.score_threshold_slider.setMaximum(100)
        self.score_threshold_slider.setValue(int(self.threshold * 100))
        self.score_threshold_slider.valueChanged.connect(self.updateScoreThreshold)

        self.algorithm_label = QLabel('Cümle Benzerliği Algoritması:', self)
        self.algorithm_combo = QComboBox(self)
        self.algorithm_combo.addItem('cosine_similarity')
        self.algorithm_combo.addItem('jaccard_distance')
        self.algorithm_combo.currentIndexChanged.connect(self.updateAlgorithm)

        self.select_reference_button = QPushButton('Referans Metin Seç', self)
        self.select_reference_button.clicked.connect(self.showFileDialog)

        layout = QVBoxLayout()
        layout.addWidget(self.upload_button)
        layout.addWidget(self.select_reference_button)
        layout.addWidget(self.graph_label)
        layout.addWidget(self.canvas)

        layout.addWidget(self.summary_label)
        layout.addWidget(self.summary_text)

        layout.addWidget(self.rouge_score_label)
        layout.addWidget(self.rouge_score_label_text)

        #layout.addWidget(self.graph_text)
        layout.addWidget(self.similarity_threshold_label)
        layout.addWidget(self.similarity_threshold_slider)
        layout.addWidget(self.score_threshold_label)
        layout.addWidget(self.score_threshold_slider)
        layout.addWidget(self.algorithm_label)
        layout.addWidget(self.algorithm_combo)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def showFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Doküman Seç", "", "Metin Dosyaları (*.txt)", options=options)
        if file_path:
            reference_path, _ = QFileDialog.getOpenFileName(self, "Referans Metin Seç", "", "Metin Dosyaları (*.txt)",
                                                            options=options)
            if reference_path:
                with open(reference_path, 'r', encoding='utf-8') as ref_file:
                    reference_text = ref_file.read()
                    self.processDocument(file_path, reference_text)
            else:
                self.processDocument(file_path, "")  # Boş referans metin ile işlem yapılacak



    def processDocument(self, file_path, reference_text):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            """print("tip")
            type(text)
            print("metin:")
            print(text)"""
            self.sentences = self.splitIntoSentences(text)
            self.title = self.extract_title(text)
            #embedded_Sentences = self.word_embedding(sentences)
            #print("Embedded sentences:\n")
            preprocessed_sentences = self.preprocess_text(text)
            embedded_sentences = self.word_embedding(preprocessed_sentences)

            sentences = preprocessed_sentences

            self.graph_sentences_by_similarity(text, embedded_sentences, self.splitIntoSentences(text))

            # reference_text = "summary"
            rouge_score = self.calculateRougeScore(self.summary, reference_text)
            print("rouge score -->")
            print(rouge_score)

            # title = None
            #score = self.calculate_sentence_score(sentences)
            #print("Sentence score:", score)

            #self.calculate_sentence_score(self.sentences)

    """def processReferenceDocument(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            reference_text = file.read()
            self.processDocument(file_path, reference_text)"""



    def calculateRougeScore(self, summary_text, reference_text):
        #return Rouge.get_scores(summary_text, reference_text)
        """print("asdfd")
        print(summary_text)
        print("afdsa")
        print(reference_text)"""
        summary_text = str(summary_text)
        reference_text = str(reference_text)
        print("asdfd")
        print(summary_text)
        rouge = Rouge()
        rouge_scores = rouge.get_scores(summary_text, reference_text)

        str_rouge_scores = str(rouge_scores)
        # Display Rouge scores
        self.rouge_score_label_text.setText(str_rouge_scores)

        return rouge_scores


    def splitIntoSentences(self, text):
        # Cümleleri ayırma işlemi burada gerçekleştirilir
        sentences = sent_tokenize(text)
        return sentences

    def calculateSimilarity(self, sentence1, sentence2):
        algorithm = self.algorithm_combo.currentText()

        #print(sentence1)
        #print(" --> ")
        #print(sentence2)

        # Seçilen benzerlik algoritmasına göre işlemler burada gerçekleştirilebilir
        if algorithm == 'cosine_similarity':
            # Edit mesafesi benzerlik ölçümü
            """distance = edit_distance(sentence1, sentence2)
            similarity = 1 / (distance + 1)  # Yüksek edit mesafesi düşük benzerlik olarak yorumlanır"""

            similarity_matrix = self.calculate_cosine_similarity(sentence1, sentence2)

            similarity = similarity_matrix
            """# Convert matrices to numpy arrays
            array1 = np.array(sentence1)
            array2 = np.array(sentence2)

            # Calculate cosine similarity
            similarity = cosine_similarity(array1, array2)"""


        elif algorithm == 'jaccard_distance':
            # Jaccard benzerlik ölçümü
            set1 = set(sentence1.split())
            set2 = set(sentence2.split())
            similarity = 1 - jaccard_distance(set1, set2)  # Jaccard mesafesi yüksek benzerlik olarak yorumlanır
        else:
            similarity = 0.0

        min_value = np.min(similarity)
        max_value = np.max(similarity)
        normalized_matrix = (similarity - min_value) / (max_value - min_value)

        return normalized_matrix

    def calculate_cosine_similarity(self, matrix1, matrix2):
        similarity_matrix = np.zeros((matrix1.shape[0], matrix2.shape[0]))
        for i in range(matrix1.shape[0]):
            for j in range(matrix2.shape[0]):
                similarity_matrix[i][j] = 1 - cosine_distance(matrix1[i], matrix2[j])
        return similarity_matrix



    def updateSimilarityThreshold(self, value):
        self.threshold = value / 100.0
        # Benzerlik threshold'ını güncelleme işlemleri burada gerçekleştirilebilir

    def updateScoreThreshold(self, value):
        self.threshold = value / 100.0
        # Skor threshold'ını güncelleme işlemleri burada gerçekleştirilebilir

    def updateAlgorithm(self, index):
        # Seçilen benzerlik algoritmasına göre işlemler burada gerçekleştirilebilir
        pass

    """def get_tfidf_matrix(self, sentences):
        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        return tfidf_vectorizer.fit_transform(sentences)

    def get_cosine_similarity_matrix(self, sentences):
        tfidf_matrix = self.get_tfidf_matrix(sentences)
        return cosine_similarity(tfidf_matrix)"""

    def graph_sentences_by_similarity(self, text, embedded_sentences, sentences):
        print("graph_sentences_by_similarity")
        # Create a graph. Each node is a sentence.
        G = nx.Graph()
        # Add edges between similar sentences based on similarity threshold
        edge_list = []
        for i in range(len(embedded_sentences)):
            for j in range(i + 1, len(embedded_sentences)):
                similarity = self.calculateSimilarity(embedded_sentences[i], embedded_sentences[j])
                average_similarity = np.mean(similarity, dtype=np.float64)
                if average_similarity > self.threshold:
                    edge_list.append((i, j, embedded_sentences[i][j]))
        G.add_weighted_edges_from(edge_list)

        # Extract total number of connections and number of connections of nodes above threshold
        self.total_connections = G.number_of_edges()
        self.above_threshold_connections = len(edge_list)

        # Calculate sentence score
        sentence_scores = self.calculate_sentence_score(text, sentences)

        self.summary = self.generate_summary(self.sentences, sentence_scores)
        print("-->")
        print(self.summary)

        ###################
        ###################
        str_summary = str(self.summary)
        output = ""
        for item in eval(str_summary):
            output += item[0] + " "
        self.summary_text.setText(output)
        ###################
        ###################

        # Draw the graph
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        pos = nx.spring_layout(G)
        labels = {node: sentence for node, sentence in zip(G.nodes(), sentences)}
        nx.draw(G, pos, labels=labels, with_labels=True, ax=ax, font_size=7)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12)

        # Draw edge labels
        for edge in G.edges():
            x = (pos[edge[0]][0] + pos[edge[1]][0]) / 2  # x-koordinatını ortalamasını al
            y = (pos[edge[0]][1] + pos[edge[1]][1]) / 2  # y-koordinatını ortalamasını al
            similarity = G.get_edge_data(*edge)['weight']
            ax.text(x, y, f"{similarity:.2f}", horizontalalignment='center', verticalalignment='center', fontsize=8)

        self.canvas.setFixedSize(1000, 800)
        self.canvas.draw()

    def preprocess_text(self,text):
        print("preprocess_text")


        preprocessed_sentences = []
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))

        for sentence in self.sentences:
            # Küçük harfe dönüştürme
            sentence = sentence.lower()

            # Noktalama işaretlerini kaldırma
            sentence = sentence.translate(str.maketrans("", "", punctuation))

            # Kelimelere ayırma
            words = word_tokenize(sentence)

            # Kök bulma (Stemming)
            stemmed_words = [stemmer.stem(word) for word in words]

            # Gereksiz kelimeleri çıkarma (Stop-word Elimination)
            filtered_words = [word for word in stemmed_words if word not in stop_words]

            # Ön işleme adımlarının tamamlanmış cümleyi birleştirme
            preprocessed_sentence = " ".join(filtered_words)
            preprocessed_sentences.append(preprocessed_sentence)

        # Ön işleme adımlarından geçirilmiş cümleleri döndürme
        return preprocessed_sentences

    def word_embedding(self, sentences):
        embedded_sentences = []
        for sentence in sentences:
            # Cümlenin kelimelerini vektörlere dönüştürün
            tokens = word_tokenize(sentence.lower())

            word_vectors = []
            for word in tokens:
                if self.word_embedding_model.__contains__(word):
                    word_vectors.append(self.word_embedding_model[word])

            # Cümlenin vektörünü hesaplayın (kelime vektörlerinin toplamı)
            sentence_vector = np.sum(word_vectors, axis=0)

            # Gömülü cümleyi listeye ekleyin
            embedded_sentences.append(sentence_vector)

        return embedded_sentences

    def extract_title(self,document):
        first_empty_line_index = document.find('\n\n')

        if first_empty_line_index == -1:
            return None

        title = document[:first_empty_line_index].strip()
        return title

    def generate_summary(self, sentences, sentence_scores):

        s = list(zip(sentences, sentence_scores))
        s.sort(reverse=True)

        summary_sentences = []
        len_sentences = len(s)
        for i in range(len_sentences // 2):
            summary_sentences.append(s[i])

        return summary_sentences

    def calculate_sentence_score(self, text, sentence):
        def calculate_proper_noun_score(sentence):
            tokens = nltk.word_tokenize(sen)
            tagged = nltk.pos_tag(tokens)
            proper_nouns = [word for word, pos in tagged if pos == 'NNP']
            proper_noun_score = len(proper_nouns) / len(tokens)
            return proper_noun_score

        def calculate_numeric_data_score(sentence):
            tokens = nltk.word_tokenize(sen)
            numeric_data = [token for token in tokens if token.isdigit()]
            numeric_data_score = len(numeric_data) / len(tokens)
            return numeric_data_score

        def calculate_node_similarity_score(sentence, threshold):
            if(self.total_connections != 0):
                return self.above_threshold_connections / self.total_connections
            else:
                return 0


        def calculate_title_word_score(sentence, title):
            title_words = word_tokenize(title)
            title_word_scores = []

            tokens = word_tokenize(sentence)
            matching_words = [token for token in tokens if token in title_words]
            title_word_score = len(matching_words) / len(tokens)
            return title_word_score



        def create_tf_idf_matrix(sentences) -> dict:

            # Calculate term frequency (TF) for each sentence
            tfs = []
            for sentence in sentences:
                tf = Counter(word_tokenize(sentence))
                tfs.append(tf)

            # Calculate inverse document frequency (IDF)
            idf = {}
            for tf in tfs:
                for word in tf:
                    if word not in idf:
                        idf[word] = math.log(len(sentences) / (sum([1 for tf in tfs if word in tf]) + 1))

            # Calculate TF-IDF for each sentence
            tf_idfs = []
            t_idfs_sentence_by_sentence = []
            for i, tf in enumerate(tfs):
                tf_idf = {}
                for word in tf:
                    tf_idf[word] = tf[word] * idf[word]
                tf_idfs.append(tf_idf)


            return tf_idfs

        sentence_scores = []
        p5  = create_tf_idf_matrix(sentence)
        for i, sen in enumerate(sentence):
            p1 = calculate_proper_noun_score(sen)
            p2 = calculate_numeric_data_score(sen)
            p3 = calculate_node_similarity_score(sen, self.threshold)
            p4 = calculate_title_word_score(sen, self.title)
            # print('TF-IDF matrix', tf_idf_matrix)
            # print('First document tfidf',tf_idf_matrix[list(tf_idf_matrix.keys())[0]])
            print("p degerleri")
            print(p1)
            print(p2)
            print(p3)
            print(p4)
            print(p5[i])

            sum_p5 =0

            for value in p5[i].values():
                if isinstance(value, float):
                    sum_p5 += value

            p5_score = sum_p5/ len(p5[i])
            sentence_score = (p1+p2+p3+p4+p5_score)/5
            print(sentence_score)
            sentence_scores.append(sentence_score)


        return sentence_scores