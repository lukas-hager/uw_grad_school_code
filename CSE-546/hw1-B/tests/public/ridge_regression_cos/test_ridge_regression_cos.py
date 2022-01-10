from typing import Callable
from unittest import TestCase, TestLoader, TestSuite

import numpy as np
from gradescope_utils.autograder_utils.decorators import partial_credit, visibility

import homeworks.ridge_regression_cos.ridge_regression_cos as ridge_regression_cos


class TestRidgeRegressionCos(TestCase):
    @visibility("visible")
    @partial_credit(1)
    def test_transform_data(self, set_score: Callable[[int], None]):
        try:
            x = np.array(
                [
                    [2.0, 3.0, 1.0, 2.0],
                    [3.0, 2.0, 2.0, 1.0],
                    [3.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 2.0],
                ],
            )
            G = np.array([[2.0, 1.0], [1.0, 2.0], [1.0, 1.0], [1.0, 1.0],],)
            b = np.array([1.0, -1.0])
            expected = np.array(
                [
                    [0.0044257, -0.83907153],
                    [0.84385396, -0.91113026],
                    [-0.83907153, 0.96017029],
                    [0.75390225, 0.28366219],
                    [0.75390225, 0.28366219],
                ]
            )

            actual = ridge_regression_cos.transform_data(x, G, b)
            np.testing.assert_array_almost_equal(actual, expected, decimal=4)
        except:  # noqa: E722
            raise

        set_score(1)

    @visibility("visible")
    @partial_credit(1)
    def test_split_into_validation(self, set_score: Callable[[int], None]):
        try:
            x = np.array(
                [
                    [2.0, 3.0, 1.0, 2.0],
                    [3.0, 2.0, 2.0, 1.0],
                    [3.0, 1.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 1.0],
                    [1.0, 1.0, 1.0, 2.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [2.0, 1.0, 2.0, 2.0],
                ],
            )
            y = np.array([2, 3, 0, 1, 4, 9, 7])
            expected_train_size = 4
            expected_val_size = 3

            (
                (x_train, y_train),
                (x_val, y_val),
            ) = ridge_regression_cos.split_into_validation(x, y, fraction_train=0.6)
            assert (
                x_train.shape[0] == expected_train_size
            ), f"x_train doesn't match expected shape. It has length of {x_train.shape[0]}, but expected {expected_train_size}"
            assert (
                y_train.shape[0] == expected_train_size
            ), f"y_train doesn't match expected shape. It has length of {y_train.shape[0]}, but expected {expected_train_size}"
            assert (
                x_val.shape[0] == expected_val_size
            ), f"x_val doesn't match expected shape. It has length of {x_val.shape[0]}, but expected {expected_val_size}"
            assert (
                y_val.shape[0] == expected_val_size
            ), f"y_val doesn't match expected shape. It has length of {y_val.shape[0]}, but expected {expected_val_size}"
        except:  # noqa: E722
            raise

        set_score(1)


# Create a Suite for this problem
suite_cos = TestLoader().loadTestsFromTestCase(TestRidgeRegressionCos)

RidgeRegressionCosTestSuite = TestSuite([suite_cos])
