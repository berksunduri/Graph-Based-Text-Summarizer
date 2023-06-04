from tkinter import *
from tkinter import filedialog, messagebox
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch
import re
import numpy as np
from rouge import rouge
from rouge import Rouge
from rouge_score import rouge_scorer


class Application:
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        self.sentence_embeddings = None
        #uploadButton
        self.button = Button(frame, text="Upload", command=self.upload_file)
        self.button.pack(side=LEFT)
        #referanceUploadButton
        self.reference_button = Button(frame, text="Upload Reference", command=self.upload_reference)
        self.reference_button.pack(side=LEFT)  # New button for uploading reference text
        #graphbutton
        self.graphdisplay = Button(frame, text="Display Graph", command=self.display_graph)
        self.graphdisplay.pack(side=LEFT)

        #text
        self.summary_text = Text(frame)
        self.summary_text.pack()
        #rougeCalculateButton
        self.rouge_button = Button(frame, text="Calculate ROUGE", command=self.calculate_and_display_rouge)
        self.rouge_button.pack(side=LEFT)

    def upload_file(self):
        filepath = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
        if filepath:
            with open(filepath, 'r') as file:
                text = file.read()
            self.original_text = text
            #texti Tokenle
            self.sentences = sent_tokenize(text)


            #cumle embeddinglerini bul
            sentence_embeddings = [self.get_sentence_embedding(s) for s in self.sentences]
            self.sentence_embeddings = sentence_embeddings


            title_words = []

            # graf olustur nodelari ayarla
            self.G = nx.Graph()
            for idx, sentence in enumerate(self.sentences):
                self.G.add_node(idx, sentence=sentence, embedding=sentence_embeddings[idx])

            # nodelara cosinus benzerligine gore baglanti ekle
            for i in range(len(self.sentences)):
                for j in range(i + 1, len(self.sentences)):
                    sim = cosine_similarity(sentence_embeddings[i].mean(axis=0).reshape(1, -1),
                                            sentence_embeddings[j].mean(axis=0).reshape(1, -1))[0][0]
                    if sim > 0.5:  # 0.5 threshold(istege gore degistir)
                        self.G.add_edge(i, j, weight=sim)

            # cumle skorlari hesap
            sentence_scores = self.calculate_sentence_scores(self.sentences, title_words)

            # nodelari skorlara gore ayarla
            for idx, score in enumerate(sentence_scores):
                self.G.nodes[idx]["score"] = score

            # ozeti olustur
            summary = self.summarize()

            self.summary_text.delete('1.0', END)
            self.summary_text.insert(END, summary)
            #ozeti printle
            print(summary)

    def upload_reference(self):
        filepath = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
        if filepath:
            with open(filepath, 'r') as file:
                self.reference_text = file.read()

    def display_graph(self):
        G = self.G

        # node pozisyonlari
        pos = nx.spring_layout(G)

        # nodelar
        nx.draw_networkx_nodes(G, pos, node_size=500)

        # baglantilar
        nx.draw_networkx_edges(G, pos, edgelist=G.edges())

        # nodelara cumle skorlari ekle
        node_labels = nx.get_node_attributes(G, 'score')
        node_labels = {k: "{:.1f}".format(v) for k, v in node_labels.items()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color='r')

        # semantik benzerlik skorlarini baglantilara ekle
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: "{:.1f}".format(v) for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.axis('off')
        plt.show()

    def summarize(self, top_n=5):
        # cumleleri skorlara gore sirala
        sorted_sentences = sorted(self.G.nodes(data=True), key=lambda x: x[1]['score'], reverse=True)

        # top_n cumleleri sirala
        top_sentences = sorted_sentences[:top_n]

        # cumleleri text siralamasina gore sirala
        top_sentences = sorted(top_sentences, key=lambda x: x[0])

        # cumleleri birlestir ozeti olustur
        summary = ' '.join([node[1]['sentence'] for node in top_sentences])
        self.summary = summary
        return summary

    def calculate_rouge(self, generated_summary, reference_summary):
        rouge = Rouge()
        scores = rouge.get_scores(generated_summary, reference_summary, avg=True)

        scores_percentage = {key: {k: v * 100 for k, v in value.items()} for key, value in scores.items()}
        return scores_percentage

    def calculate_and_display_rouge(self):
        if self.summary and self.reference_text:
            rouge_scores = self.calculate_rouge(self.summary, self.reference_text)
            print(rouge_scores)
            messagebox.showinfo("ROUGE Scores", f"ROUGE-1: {rouge_scores['rouge-1']}\nROUGE-L: {rouge_scores['rouge-l']}")


    @staticmethod
    def get_sentence_embedding(sentence):
        model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        inputs = tokenizer(sentence, return_tensors='pt')
        outputs = model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    @staticmethod
    def get_continuous_chunks(text):
        chunked = ne_chunk(pos_tag(word_tokenize(text)))
        continuous_chunk = []
        current_chunk = []

        for i in chunked:
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def calculate_sentence_scores(self, sentences, title_words):
        scores = []

        # numerik veriyi olustur
        named_entities = []
        numerical_data = []
        for sentence in sentences:
            named_entities.extend(self.get_continuous_chunks(sentence))
            numerical_data.extend(re.findall(r'\d+', sentence))

        # TF-IDF Skoru hesaplama
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(sentences)
        tfidf_scores = np.array(X.sum(axis=0)).flatten()
        tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), tfidf_scores))

        # top 10% sozcukler tema sozcukleri olacak
        num_theme_words = int(len(tfidf_dict) * 0.10)
        theme_words = sorted(tfidf_dict, key=tfidf_dict.get, reverse=True)[:num_theme_words]

        for sentence in sentences:
            words = word_tokenize(sentence)
            sentence_length = len(words)
            # TF-IDF HESAPLAMALARI
            p1 = sum(1 for word in words if word in named_entities) / sentence_length
            p2 = sum(1 for word in words if word in numerical_data) / sentence_length
            p3 = len([e for e in self.G.edges() if e[0] == sentences.index(sentence)]) / len(self.G.edges())
            p4 = sum(1 for word in words if word in title_words) / sentence_length
            p5 = sum(1 for word in words if word in theme_words) / sentence_length

            score = p1 + p2 + p3 + p4 + p5
            scores.append(score)

        return scores


if __name__ == "__main__":
    root = Tk()
    app = Application(root)
    root.mainloop()