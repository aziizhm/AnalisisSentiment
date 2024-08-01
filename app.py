import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# Fungsi untuk mendownload data NLTK jika belum ada
def download_nltk_data():
    nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
    if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
        nltk.download('punkt')
    if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'stopwords')):
        nltk.download('stopwords')

download_nltk_data()

# Fungsi untuk pelabelan
def pelabelan(rating):
    if rating < 3:
        return 'negatif'
    elif rating > 3:
        return 'positif'
    return None  

# Fungsi untuk case folding
def case_folding(text):
    return text.lower()

# Fungsi untuk cleaning
def cleaning(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # menghapus @..
    text = re.sub(r'#', '', text)  # menghapus hashtag
    text = re.sub(r'https?:\/\/\S+', '', text)  # menghapus url
    text = re.sub(r':\)', '', text)  # menghapus icon
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)  # menghapus single character
    text = re.sub(r'\s+', ' ', text)  # menghapus spasi ganda
    text = re.sub(r'[^\w\s]', '', text)  # menghapus punctuation
    text = re.sub(r'(.)\1+', r'\1\1', text)  # menghapus kata yg berulang seperti ooo menjadi oo
    text = re.sub(r'\.+', '', text)  # menghilangkan teks yang mengandung akhiran MULTIPLE TITIK (...)
    text = re.sub(r'\d+', '', text)  # menghilangkan teks yang mengandung angka
    text = re.sub(r'\b\w{1,2}\b', '', text)  # menghilangkan teks yang mengandung 1-2 huruf
    text = text.encode('ascii', 'ignore').decode('ascii')  # menghapus non-ascii (EMOTICON)
    return text

# Fungsi untuk tokenization
def tokenize(text):
    return word_tokenize(text)

# Fungsi untuk stopword removal
stop_words = set(stopwords.words('indonesian'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# Fungsi untuk stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

# Judul aplikasi
st.title("Aplikasi Analisis Sentimen Menggunakan Algortima Naive Bayess Classifier")

uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Display original data
    st.subheader('Original Data')
    st.write(df)

    # Pastikan kolom 'rating' dan 'content' ada dalam DataFrame
    if 'rating' in df.columns and 'content' in df.columns:
        # Menggunakan fungsi apply untuk membuat kolom 'Label' baru
        df['Label'] = df['rating'].apply(pelabelan)

        # Tampilkan data setelah pelabelan
        st.subheader('Data Setelah Labeling')
        st.write(df[['rating', 'content', 'Label']])

        # Menghapus baris dengan Label None (untuk rating = 3)
        df = df[df['Label'].notna()]

        # Case folding
        df['casefolding'] = df['content'].apply(case_folding)
        st.subheader('Data Setelah Case Folding')
        st.write(df[['content', 'casefolding']])

        # Cleaning
        df['cleaning'] = df['casefolding'].apply(cleaning)
        df = df.dropna()
        df = df[df['cleaning'] != '']
        df.drop_duplicates(subset=['cleaning'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.subheader('Data Setelah Cleaning')
        st.write(df[['casefolding', 'cleaning']])

        # Tokenization
        df['token'] = df['cleaning'].apply(tokenize)
        st.subheader('Data Setelah Tokenization')
        st.write(df[['cleaning', 'token']])

        # Stopword removal
        df['stopword'] = df['token'].apply(remove_stopwords)
        st.subheader('Data Setelah Stopword Removal')
        st.write(df[['token', 'stopword']])

        # Stemming
        df['stemming'] = df['stopword'].apply(stemming)
        st.subheader('Data Setelah Stemming')
        st.write(df[['stopword', 'stemming']])

        # Tab untuk memilih hasil yang ingin dilihat
        tab1, tab2 = st.tabs(["Modeling", "Diagram & Wordcloud"])

        with tab1:
            
            # Membagi data menjadi data training dan testing
            X_train, X_test, y_train, y_test = train_test_split(df['cleaning'], df['Label'], test_size=0.20, random_state=0)

            # Tampilkan ukuran data training dan testing
            st.subheader("Ukuran Data Training dan Testing")
            st.write(f"X_train shape: {X_train.shape}")
            st.write(f"y_train shape: {y_train.shape}")
            st.write(f"X_test shape: {X_test.shape}")
            st.write(f"y_test shape: {y_test.shape}")

            # TF-IDF Vectorization
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_train = tfidf_vectorizer.fit_transform(X_train)
            tfidf_test = tfidf_vectorizer.transform(X_test)

            # Count Vectorizer
            vectorizer = CountVectorizer()
            vectorizer.fit(X_train)

            X_train_vectorized = vectorizer.transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)

            # Menampilkan array dari X_train
            st.subheader("Array dari X_train (TF-IDF)")
            st.write(pd.DataFrame(tfidf_train.toarray(), columns=tfidf_vectorizer.get_feature_names_out()))

            st.subheader("Array dari X_train (Count Vectorizer)")
            st.write(pd.DataFrame(X_train_vectorized.toarray(), columns=vectorizer.get_feature_names_out()))

            # Naive Bayes Model
            nb = MultinomialNB()
            nb.fit(tfidf_train, y_train)

            # Predict and evaluate
            y_pred = nb.predict(tfidf_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="binary", pos_label="negatif")
            recall = recall_score(y_test, y_pred, average="binary", pos_label="negatif")
            f1 = f1_score(y_test, y_pred, average="binary", pos_label="negatif")
            conf_matrix = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred, zero_division=0)

            st.subheader("Hasil Evaluasi Model Naive Bayes")
            st.write(f"Akurasi: {accuracy}")
            st.write(f"Presisi: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1-Score: {f1}")
            st.write(f"Confusion Matrix:\n {conf_matrix}")
            st.write("Classification Report:\n", class_report)

        with tab2:
            # Hitung jumlah label positif dan negatif
            jumlah_positif = df[df['Label'] == 'positif']['Label'].count()
            jumlah_negatif = df[df['Label'] == 'negatif']['Label'].count()

            # Hitung total jumlah data
            total_data = len(df)

            # Hitung persentase label positif dan negatif
            persentase_positif = (jumlah_positif / total_data) * 100
            persentase_negatif = (jumlah_negatif / total_data) * 100

            # Buat diagram pie
            labels = ['Positif\nTotal: {}'.format(jumlah_positif), 'Negatif\nTotal: {}'.format(jumlah_negatif)]
            sizes = [persentase_positif, persentase_negatif]
            colors = ['skyblue', 'lightcoral']
            explode = (0.1, 0)  # Pisahkan potongan untuk label 'Positif'

            plt.figure(figsize=(6, 6))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
            plt.axis('equal')  # Biar diagramnya menjadi lingkaran
            plt.title('Persentase Label Positif dan Negatif')
            st.pyplot(plt)

            # Wordcloud untuk seluruh data
            teks = ' '.join(df['stemming'].apply(lambda x: ' '.join(x)))
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(teks)

            st.subheader("Wordcloud pada Data Komentar")
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Wordcloud pada data komentar')
            st.pyplot(plt)

            # Wordcloud untuk data positif
            data_positif = df[df['Label'] == 'positif']
            teks_positif = ' '.join(data_positif['stemming'].apply(lambda x: ' '.join(x)))
            wordcloud_positif = WordCloud(width=800, height=400, background_color='white').generate(teks_positif)

            st.subheader("Wordcloud Data Positif")
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_positif, interpolation='bilinear')
            plt.axis('off')
            plt.title('Wordcloud Data Positif')
            st.pyplot(plt)

            # Wordcloud untuk data negatif
            data_negatif = df[df['Label'] == 'negatif']
            teks_negatif = ' '.join(data_negatif['stemming'].apply(lambda x: ' '.join(x)))
            wordcloud_negatif = WordCloud(width=800, height=400, background_color='white').generate(teks_negatif)

            st.subheader("Wordcloud Data Negatif")
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud_negatif, interpolation='bilinear')
            plt.axis('off')
            plt.title('Wordcloud Data Negatif')
            st.pyplot(plt)
