import pytest
from package import pz_Evaluation as yf


def test_algorithm_flavor():
    assert yf.algorithm_flavor('fzb_base_base') == 'fzboost'
    assert yf.algorithm_flavor('knn_base_HSC') == 'knn'
    assert yf.algorithm_flavor('a_b_c') == 'a'

