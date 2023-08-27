import numpy as np
import pandas as pd
import re


def load_data():
    df_movies = pd.read_csv("./dataset/movies.csv", delimiter=',', quotechar='"')
    df_ratings = pd.read_csv("./dataset/ratings.csv")

    content_movies = prepare_content_movies(df_movies, df_ratings)
    content_user = prepare_content_users(content_movies, df_ratings)

    df_ratings['index'] = df_ratings.index
    movies_train = df_ratings.drop(columns=['userId', 'rating', 'timestamp']).merge(content_movies, on='movieId', validate='m:1')
    movies_train.sort_values(by='index', inplace=True)
    movies_train.drop(columns=['index'], inplace=True)

    user_train = df_ratings.drop(columns=['movieId', 'rating', 'timestamp']).merge(content_user, on='userId', validate='m:1')
    user_train.sort_values(by='index', inplace=True)
    user_train.drop(columns=['index'], inplace=True)

    y_train = df_ratings['rating']

    return movies_train, user_train, y_train, df_movies


def prepare_content_movies(movies, ratings):
    df = movies
    df["year"] = movies["title"].apply(test_year_title)

    df_avg = ratings.groupby(['movieId'])['rating'].mean().to_frame()
    df = df.merge(df_avg, how='left', on='movieId')

    genre_columns = set('|'.join(movies["genres"]).split('|'))
    genre_columns = list(genre_columns)
    genre_columns.sort()

    np_array = np.zeros((len(df), len(genre_columns)))
    df = df.reindex(columns= df.columns.tolist() + genre_columns)
    df[genre_columns] = np_array

    for i, row in enumerate(movies.itertuples()):
        genres = row.genres.split('|')
        for g in genres:
            df.at[i, g] = 1

    return df.drop(columns=['title', 'genres'])


def prepare_content_users(prepared_movies, ratings):
    df = ratings.groupby(['userId'])['rating'].count().to_frame()
    df.reset_index(inplace=True)
    df_mean = ratings.groupby(['userId'])['rating'].mean().to_frame()
    df = df.merge(df_mean, how='left', on='userId')
    df.rename(columns={'rating_x' : 'rating_count', 'rating_y' : 'rating_mean'}, inplace=True)

    movies_vecs = prepared_movies.drop(columns=['year', 'rating'])
    user_vecs = ratings.merge(movies_vecs, how='outer', on='movieId')
    user_vecs.iloc[:, 4:] = user_vecs.iloc[:, 4:].multiply(user_vecs['rating'], axis=0)
    user_vecs.mask(user_vecs == 0, np.nan, inplace=True)
    user_vecs = user_vecs.drop(columns=['movieId', 'rating', 'timestamp']).groupby(['userId']).mean()
    user_vecs.fillna(0, inplace=True)
    return df.merge(user_vecs, how='left', on='userId')


def test_year_title(x):
    match = re.search(r"\((\d{4})\)", x)
    if match:
        return match.group(1)
    return 0