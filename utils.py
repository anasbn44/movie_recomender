import numpy as np
import pandas as pd



def prepare_data(movies, ratings):
    Y = ratings.pivot(index="movieId", columns="userId", values="rating")
    Y = Y.reindex(movies["movieId"])
    Y = Y.fillna(0)
    R = Y.mask(Y > 0, 1)
    return Y, R

def prepare_x(movies, features):
    pass