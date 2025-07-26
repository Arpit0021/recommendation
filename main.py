import streamlit as st
import pandas as pd
import requests
import nltk
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Setup ---
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

tmdb_api_key = "670f0efc82e0fb4d4d56a2303dff5ab5"

omdb_api_key = "9f607133"

# Load Data
movies = pd.DataFrame(pickle.load(open('movie_dict.pkl', 'rb')))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies['overview'] = movies.get('overview', pd.Series(['No description available'] * len(movies))).fillna(
    'No description available')


# --- Utility Functions ---

@st.cache_resource
def fetch_movie_info(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={omdb_api_key}"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        if data.get('Response') == 'True':
            return {
                "Poster": data.get('Poster', 'https://via.placeholder.com/150') if data.get(
                    'Poster') != 'N/A' else 'https://via.placeholder.com/150',
                "Plot": data.get('Plot', 'No plot available'),
                "Actors": data.get('Actors', 'N/A'),
                "Rating": data.get('imdbRating', 'N/A'),
                "Genre": data.get('Genre', 'N/A'),
                "Release": data.get('Released', 'N/A'),
                "Runtime": data.get('Runtime', 'N/A')
            }
    return {key: 'N/A' if key != "Poster" else 'https://via.placeholder.com/150' for key in
            ['Poster', 'Plot', 'Actors', 'Rating', 'Genre', 'Release', 'Runtime']}


@st.cache_resource
def get_tmdb_movie_id(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={tmdb_api_key}&query={title}"
    res = requests.get(url).json()
    return res.get('results', [{}])[0].get('id')


@st.cache_resource
def get_cast_images(title):
    movie_id = get_tmdb_movie_id(title)
    if not movie_id:
        return []
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={tmdb_api_key}"
    cast = requests.get(url).json().get('cast', [])[:5]
    return [(c['name'], f"https://image.tmdb.org/t/p/w200{c['profile_path']}" if c.get(
        'profile_path') else "https://via.placeholder.com/100") for c in cast]


def find_similar_title_parts(title):
    base = title.lower().split(":")[0].replace("part", "").replace("the", "").strip()
    return movies[movies['title'].str.lower().str.contains(base)]['title'].tolist()


def recommend(title):
    title_lower = title.lower()
    match = movies[movies['title'].str.lower() == title_lower]
    if match.empty:
        return [], [], [], []

    idx = match.index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:]
    series_titles = find_similar_title_parts(title)

    recommendations = []
    posters, summaries, sentiments = [], [], []

    for i, _ in distances:
        rec_title = movies.iloc[i]['title']
        if rec_title not in recommendations and (rec_title in series_titles or len(recommendations) < 5):
            recommendations.append(rec_title)
        if len(recommendations) == 5:
            break

    for rec in recommendations:
        info = fetch_movie_info(rec)
        posters.append(info['Poster'])
        summaries.append(movies[movies['title'] == rec]['overview'].values[0])
        sentiments.append(0)

    return recommendations, posters, summaries, sentiments


def sentiment_analysis(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.2:
        return "😊🌟 This movie gives off a *Great Vibe*!"
    elif score <= -0.2:
        return "🙁⚠ This movie might feel a little *Sad or Intense*, but that's okay!"
    return "😐🤔 This movie feels a bit *Neutral*."


# --- UI Styling ---
st.markdown("""
<style>
.stApp {
    background-image: url("https://wallpapers.com/images/high/netflix-background-gs7hjuwvv2g0e9fj.webp");
    background-size: cover;
    background-position: center;
}
.cast-frame {
    border: 3px solid white;
    padding: 0;
    box-shadow: 0 0 20px rgba(255, 255, 255, 0.7);
    margin-bottom: 10px;
}
.movie-poster, .recommended-poster {
    width: 150px;
    height: 225px;
    object-fit: cover;
}
</style>
""", unsafe_allow_html=True)

# --- App ---
st.title("🎬 Movie Recommender System")
selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    info = fetch_movie_info(selected_movie)
    names, posters, summaries, sentiments = recommend(selected_movie)
    cast_info = get_cast_images(selected_movie)

    # Movie Info
    st.markdown("## 🎥 Selected Movie Details")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(info["Poster"], width=250)
    with col2:
        st.markdown(f"**Title:** {selected_movie}")
        st.markdown(f"**Rating:** {info['Rating']}/10" if info['Rating'] != 'N/A' else "Rating: N/A")
        st.markdown(f"**Runtime:** {info['Runtime']}")
        st.markdown(f"**Genre:** {info['Genre']}")
        st.markdown(f"**Release Date:** {info['Release']}")
        st.markdown(f"**Actors:** {info['Actors']}")
        st.markdown(f"**Summary:** {info['Plot']}")

    # Cast Section
    if cast_info:
        st.markdown("## 🎭 Top Characters")
        cols = st.columns(len(cast_info))
        for col, (name, img_url) in zip(cols, cast_info):
            with col:
                st.markdown(f'<div class="cast-frame"><img src="{img_url}" width="150" height="225" /></div>',
                            unsafe_allow_html=True)
                st.markdown(name)

    # Sentiment
    st.markdown("## 🧠 Sentiment Analysis")
    st.markdown(f"*Sentiment Analysis Result:* {sentiment_analysis(info['Plot'])}")

    # Recommendations
    if names:
        st.markdown("---")
        st.markdown("## 🤝 Recommended Movies for You")
        cols = st.columns(len(names))
        for i in range(len(names)):
            with cols[i]:
                st.image(posters[i], width=150, use_container_width=True)
                st.markdown(names[i])

