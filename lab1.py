import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Nuskaitome duomenų rinkinį iš CSV failo
def load_articles_from_csv(file_path):
    data_set = pd.read_csv(file_path)  # Nuskaitomas duomenų rinkinys iš CSV failo
    return data_set  # Grąžinamas nuskaitytas duomenų rinkinys

# Funkcija, skirta rasti labiausiai atitinkančius straipsnius pagal vartotojo įvestą frazę
def find_most_similar_articles(query, vectorizer, tfidf_matrix, data_set, threshold=0.3):
    query_tfidf = vectorizer.transform([query])  # Transformuojama vartotojo įvesta frazė į TF-IDF formatą
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()  # Skaičiuojami panašumo balai tarp frazės ir straipsnių
    similar_indices = similarity_scores.argsort()[::-1]  # Surikiuojami indeksai pagal panašumą mažėjimo tvarka
    similar_articles = data_set.iloc[similar_indices]  # Surandami panašiausių straipsnių duomenys pagal surikiuotus indeksus
    similar_articles = similar_articles[similarity_scores[similar_indices] > threshold]  # Pasirenkami tik tie straipsniai, kurių panašumas yra didesnis nei nurodytas slenkstis
    return similar_articles  # Grąžinami panašiausių straipsnių duomenys

def get_top_words_and_articles(vectorizer, tfidf_matrix, data_set):
    # Apskaičiuojame svorius kiekvienam žodžiui TF-IDF matricoje
    word_scores = tfidf_matrix.sum(axis=0).tolist()[0]

    # Susiejame žodžius su jų svoriais
    words = vectorizer.get_feature_names_out()
    word_score_dict = dict(zip(words, word_scores))

    # Rikiuojame žodžius pagal svorius
    sorted_word_scores = sorted(word_score_dict.items(), key=lambda x: x[1], reverse=True)

    # Pasiimame 5 dažniausiai pasikartojančius žodžius
    top_words = [word for word, _ in sorted_word_scores[:5]]

    # Sukuriame sąrašą straipsnių, kuriuose pasikartoja šie žodžiai
    articles_with_top_words = {word: [] for word in top_words}
    for index, row in data_set.iterrows():
        for word in top_words:
            if word in row['Text']:
                articles_with_top_words[word].append(row)

    # Sukuriame galutinį sąrašą straipsnių su pasikartojančiais žodžiais, atsižvelgdami į žodžių dažnumą
    top_articles = []
    for word, articles in articles_with_top_words.items():
        articles_sorted_by_word_count = sorted(articles, key=lambda x: x['Text'].count(word), reverse=True)
        top_articles.extend(articles_sorted_by_word_count[:5])

    return top_words, top_articles  # Grąžiname 5 pirmus straipsnius su dažniausiai pasikartojančiais žodžiais

# Puslapio pavadinimas
st.title('Straipsnių paieška')

# Nurodome kelią iki CSV failo
csv_file_path = "C:\\Users\\meila\\OneDrive\\Stalinis kompiuteris2\\I_laboras_DI\\Vegan_Articles.csv"

# Nuskaitome straipsnius iš CSV failo
data_set = load_articles_from_csv(csv_file_path)

# Sukuriame TF-IDF vektorizatorių ir TF-IDF matricą
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data_set['Text'])

# Įvesties laukas vartotojo frazei
query = st.text_input('Įveskite paieškos frazę')

# Jei įvesta frazė, rasti panašius straipsnius ir juos atvaizduoti
if query:
    similar_articles = find_most_similar_articles(query, vectorizer, tfidf_matrix, data_set, threshold=0.3)
    if similar_articles.shape[0] > 0:
        st.subheader('Panašūs straipsniai:')
        for index, row in similar_articles.iterrows():
            st.write('**Title:**', row['Title'])
            st.write('**Link:**', row['Link'])
            st.write('**Date:**', row['Date'])
            st.write('**Text:**', row['Text'])
    else:
        st.write("No similar results.")

# Atvaizduoti 5 dažniausiai pasikartojančius žodžius ir su jais susijusius straipsnius
top_words, top_articles = get_top_words_and_articles(vectorizer, tfidf_matrix, data_set)
st.subheader('5 dažniausiai pasikartojantys žodžiai ir su jais susiję straipsniai:')
for word in top_words:
    st.write(f'**{word}:**')
    articles_for_word = [article for article in top_articles if word in article['Text']][:5]  # Pasiimame tik 5 straipsnius
    for article in articles_for_word:
        st.write(f'- [{article["Title"]}]({article["Link"]})')