from unittest import TestCase
import pandas as pd
pd.set_option('display.max_columns', None)

import utils


class Test(TestCase):
    def setUp(self) -> None:
        self.ratings = pd.read_csv("dataset/ratings.csv")
        self.movies = pd.read_csv("dataset/movies.csv")
        self.genre_columns = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
                              "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                              "Thriller", "War", "IMAX", "Western", "(no genres listed)"]

    def test_prepare_data(self):
        self.assertIsNotNone(utils.prepare_data(self.movies, self.ratings), "wrong")

    def test_prepare_x(self):
        self.assertIsNotNone(utils.prepare_x(self.movies, self.genre_columns), "wrong")

