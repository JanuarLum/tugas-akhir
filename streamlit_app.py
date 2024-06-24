import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import streamlit as st
import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataset = pd.read_excel('/content/drive/MyDrive/data_ps_uk.xlsx')
dataset1 = pd.read_excel('/content/drive/MyDrive/data_ps_us.xlsx')
dataset2 = pd.read_excel('/content/drive/MyDrive/dta_playstore.xlsx')

st.write("Data dari Inggris")
st.write(dataset)
st.write("Data dari Amerika")
st.write(dataset1)
st.write("Data dari Indonesia")
st.write(dataset2)

data = dataset[['content', 'score']]
data.dropna(inplace=True)
data1 = dataset1[['content', 'score']]
data.dropna(inplace=True)
data2 = dataset2[['content', 'score']]
data.dropna(inplace=True)

def pelabelan(rate):
  if rate < 3:
    return 'negatif'
  else:
    return 'positif'

data['label'] = data['score'].apply(pelabelan)
data1['label'] = data['score'].apply(pelabelan)
data2['label'] = data['score'].apply(pelabelan)

st.write(data)
st.write(data1)
st.write(data2)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

data['content']
data1['content']
data2['content']

lemma = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words())

def CleanReview(txt):
  txt = re.sub(r'http\S+', ' ', txt)
  txt = re.sub('[^a-zA-Z]', ' ', txt)
  txt = str(txt).lower()
  txt = word_tokenize(txt)
  txt = [item for item in txt if item not in stop_words]
  txt = [lemma.lemmatize(word=w,pos='v') for w in txt]
  txt = [i for i in txt if len(i) > 2]
  txt = ' '.join(txt)
  return txt

data['CleanReview'] = data['content'].apply(CleanReview)
data1['CleanReview'] = data1['content'].apply(CleanReview)
data2['CleanReview'] = data2['content'].apply(CleanReview)

st.write("Data setelah preprocessing")
st.write(data)
st.write(data1)
st.write(data2)

x = data['CleanReview']
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

vectorizer = CountVectorizer()
vectorizer.fit(x_train)

x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(x_test)

from textblob import TextBlob

data_pstore = list(data['content'])
polaritas = 0

status = []
total_positif = total_negatif = total = 0

for i, pstore in enumerate(data_pstore):
  analysis = TextBlob(pstore)
  polaritas += analysis.polarity

  if analysis.sentiment.polarity > 0.0:
    total_positif += 1
    status.append('Positif')
  else:
    total_negatif += 1
    status.append('Negatif')

  total += 1

st.write('Hasil Analisis Data:')
st.write('Positif =', total_positif)
st.write('Negatif =', total_negatif)
st.write('Total Data =', total)

# data_pstore1 = list(data1['content'])
# polaritas = 0

# status = []
# total_positif1 = total_negatif1 = total1 = 0

# for i, pstore in enumerate(data_pstore1):
#   analysis = TextBlob(pstore)
#   polaritas += analysis.polarity

#   if analysis.sentiment.polarity > 0.0:
#     total_positif1 += 1
#     status.append('Positif')
#   else:
#     total_negatif1 += 1
#     status.append('Negatif')

#   total1 += 1

# st.write('Hasil Analisis Data:')
# st.write('Positif =', total_positif1)
# st.write('Negatif =', total_negatif1)
# st.write('Total Data =', total1)

# data['klasifikasi'] = status
# data.isnull().sum()

data.isnull().sum()

def klasifikasi(rate):
  if rate < 3:
    return 0
  # elif rate == 3:
  #   return 'netral'
  else:
    return 1

data['nilai'] = data['score'].apply(klasifikasi)
st.write(data)

for c in [0.01, 0.05, 0.25, 0.5, 0.75, 1]:
  svm = LinearSVC(C=c)
  svm.fit(x_train, y_train)

svm = LinearSVC(C = 1.0)
svm.fit(x_train, y_train)

st.write('Tingkat akurasi model :', accuracy_score(y_test, svm.predict(x_test)))

y_pred = svm.predict(x_test)
st.write('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(x_test, y_test)))
st.write(classification_report(y_test, y_pred))

# ! pip install vaderSentiment

# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# def sentiment_scores(sentence):

# 	# Create a SentimentIntensityAnalyzer object.
# 	sid_obj = SentimentIntensityAnalyzer()

# 	sentiment_dict = sid_obj.polarity_scores(sentence)

# 	st.write("Overall sentiment dictionary is : ", sentiment_dict)
# 	st.write("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
# 	st.write("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

# 	st.write("Sentence Overall Rated As", end = " ")

# 	# decide sentiment as positive, negative and neutral
# 	if sentiment_dict['compound'] >= 0.05 :
# 		st.write("Positive")
# 	else :
# 		st.write("Negative")

# # Driver code
# if __name__ == "__main__" :

# 	sentence = "coursera is the bad app"
# 	sentiment_scores(sentence)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import string
from nltk.corpus import stopwords

# Pastikan untuk mengunduh stopwords dari NLTK
nltk.download('stopwords')

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Menghapus stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Asumsikan file Excel memiliki kolom 'Comment' dan 'Sentiment'
data['content'] = data['content'].apply(clean_text)
comments = data['content']
sentiments = data['nilai']

# Preprocessing: Mengubah teks menjadi vektor fitur
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comments)
y = sentiments

# Membagi data menjadi data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model KNN
knn = KNeighborsClassifier(n_neighbors=3)

# Melatih model
knn.fit(X_train, y_train)

# Memprediksi pada data testing
y_pred = knn.predict(X_test)

# Evaluasi model
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report:\n", classification_report(y_test, y_pred))

# Fungsi untuk memprediksi sentimen dari komentar baru
def predict_sentiment(comment):
    comment_cleaned = clean_text(comment)
    comment_vec = vectorizer.transform([comment_cleaned])
    sentiment = knn.predict(comment_vec)
    return sentiment[0]

# Prediksi sentimen untuk semua komentar dalam dataset
data['Predicted_Sentiment'] = data['content'].apply(predict_sentiment)

# Menyimpan hasil prediksi ke dalam file Excel baru
output_file_path = 'predicted_sentiments.xlsx'
data.to_excel(output_file_path, index=False)

st.write(f"Predicted sentiments saved to {output_file_path}")

