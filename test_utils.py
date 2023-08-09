from unittest import TestCase
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

import utils


class Test(TestCase):
    def setUp(self) -> None:
        self.ratings = pd.DataFrame(np.array([[1,1,4], [2,1,4.5], [2,2,2.5], [3,2,5]]),
                                    columns=["userId","movieId","rating"], index=np.arange(1, 5))
        self.movies = pd.DataFrame([[1,"Nixon (1995)","Drama"], [2,"Four Rooms (1995)","Comedy"]],
                                   columns=["movieId","title","genres"], index=np.arange(1,3))
        self.genre_columns = ["Comedy", "Drama"]
        self.Y = pd.DataFrame(np.array([[4, 4.5, 0], [0, 2.5, 5]]),
                              columns=np.arange(1, 4), index=np.arange(1, 3))
        self.R = self.Y.mask(self.Y > 0, 1)
        self.X = pd.DataFrame(np.array([[0, 1], [1, 0]]), columns=self.genre_columns, index=np.arange(1, 3))

    def test_prepare_data(self):
        Y, R = utils.prepare_y_r(self.movies, self.ratings)
        print(Y.values)
        self.assertEqual(Y.values, self.Y.values, "wrong")

    def test_prepare_x(self):
        self.assertEqual(utils.prepare_x(self.movies, self.genre_columns), self.X, "wrong")

