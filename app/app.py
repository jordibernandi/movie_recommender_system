import pickle
import streamlit as st
import requests
import numpy as np
import os

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Hide the scrollbar but keep scrolling functionality */
    ::-webkit-scrollbar {
        width: 0px;
        background: transparent; /* Make the scrollbar transparent */
    }
    </style>
    """,
    unsafe_allow_html=True  # Allow HTML tags in Markdown
)

# Load data using st.cache_data to prevent reloading on every run
@st.cache_data(show_spinner=True)
def load_data():
    return {
        'movies': pickle.load(open('data/movie_list.pkl', 'rb')),
        'similarity_tfidf': pickle.load(open('data/similarity_tfidf.pkl', 'rb')),
        'similarity_bert': pickle.load(open('data/similarity_bert.pkl', 'rb'))
    }

st.sidebar.title('Team 5')

# Load data
data = load_data()
movies = data['movies']

if 'watched_movies' not in st.session_state:
    st.session_state.watched_movies = []
if 'summed_matrix_histories' not in st.session_state:
    st.session_state.summed_matrix_histories = np.zeros(movies.shape[0])

def recommend(movie, use_history):
    if embed_type == 'TF-IDF':
        similarity = data['similarity_tfidf']
    else:
        similarity = data['similarity_bert']

    index = movies[movies['title'] == movie].index[0]

    if use_history:
        st.session_state.watched_movies.append(index)
        st.session_state.summed_matrix_histories = st.session_state.summed_matrix_histories + similarity[index]
        final_matrix = st.session_state.summed_matrix_histories
    else:
        final_matrix = similarity[index]

    distances = sorted(list(enumerate(final_matrix)), reverse=True, key=lambda x: x[1])
    recommended_movie_ids = []

    count = 0
    for index, item in distances[1:]:
        if index not in st.session_state.watched_movies:
            recommended_movie_ids.append(index)
            count = count + 1
            if count == 5:
                break
            
    return recommended_movie_ids


def display_selection_page():
    st.header('Movie Recommender System - Selection')

    global embed_type
    embed_type = st.sidebar.selectbox(
        'Embedding type:',
        ['TF-IDF', 'BERT']
    )
    
    use_history = st.checkbox("Use multiple histories")

    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list 
    )

    if st.button('Show Recommendation'):
        recommended_movie_ids = recommend(selected_movie, use_history)
        display_recommendations(recommended_movie_ids)

    if st.session_state.watched_movies:
        display_watched_movies()

# Display watched movies and reset button
def display_watched_movies():
    st.sidebar.write("Watched movies:")
    for i in st.session_state.watched_movies:
        st.sidebar.write(movies['title'][i])

    if st.sidebar.button("Reset"):
        st.session_state.watched_movies = []
        st.session_state.summed_matrix_histories = np.zeros(movies.shape[0])

# Display movie recommendations
def display_recommendations(recommended_movie_ids):
    
    columns = st.columns(5)

    for i, index in enumerate(recommended_movie_ids):
        movie = movies.iloc[index]
        if not movie.empty:
            title = movie['title']
            director = ', '.join(movie['director']) if movie['director'] else "-"
            cast = ', '.join(movie['cast']) if movie['cast'] else "-"
            genre = ', '.join(movie['listed_in']) if movie['listed_in'] else "-"
            country = ', '.join(movie['country']) if movie['country'] else "-"
            release = movie['release_year'] if movie['release_year'] else "-"

            # Display each movie in a separate column
            with columns[i]:
                st.text(capitalize_sentence(title))
                image_path = 'data/images/' + str(index) + '.jpg'
                if os.path.exists(image_path):
                    st.image(image_path, use_column_width=True)
                else:
                    st.image('data/images/empty.jpg', use_column_width=True)
                st.write("**Country:**", capitalize_sentence(country))
                st.write("**Genre:**", capitalize_sentence(genre))
        else:
            st.write(f"Movie '{index}' not found in the dataset.")

def capitalize_sentence(string):
    # Split the string into sentences
    sentences = string.split(' ')

    # Capitalize the first letter of each sentence
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]

    # Join the capitalized sentences back into a single string
    return ' '.join(capitalized_sentences)

# Display content based on the selected page
def display_prompt_page():
    st.header("Movie Recommender System - Prompt")

    movie_prompt = st.text_area("Describe your ideal movie", value="", height=200)

    if st.button('Show Recommendation'):
        with st.spinner("Generating recommendation..."):
            generate_recommendation(movie_prompt)

# Generate recommendation using GPT model and display embedding
def generate_recommendation(movie_prompt):    
        response = requests.post("http://localhost:5000/embed", json={"prompt": movie_prompt})
        recommended_movies = response.json()["recommended_movie_ids"]
        display_recommendations(recommended_movies)

# Main function to display selected page
def main():
    global embed_type
    
    page = st.sidebar.selectbox(
        "Method type",
        ["Selection", "Prompt"]
    )

    if page == "Selection":
        display_selection_page()
    elif page == "Prompt":
        display_prompt_page()

# Run the app
if __name__ == "__main__":
    main()
