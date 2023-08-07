from unittest import TestCase
import pandas as pd

import utils


class Test(TestCase):
    def setUp(self) -> None:
        self.ratings = pd.read_csv("dataset/ratings.csv")
        self.movies = pd.read_csv("dataset/movies.csv")

    def test_prepare_data(self):
        self.assertEqual(utils.prepare_data(self.movies, self.ratings), (self.movies.shape[0], self.ratings.userId.nunique()), "wrong")
