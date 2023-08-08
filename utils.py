import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


def prepare_y_r(movies, ratings):
    Y = ratings.pivot(index="movieId", columns="userId", values="rating")
    Y = Y.reindex(movies["movieId"])
    Y = Y.fillna(0)
    R = Y.mask(Y > 0, 1)
    return Y, R

def prepare_x(movies, features):
    np_array = np.zeros((len(movies), len(features)))
    X = pd.DataFrame(np_array, columns=features, index=movies["movieId"])
    for i, row in enumerate(movies.itertuples(), start=1):
        genres = row.genres.split('|')
        for genre in genres:
            X.at[i, genre] = 1
    return X
