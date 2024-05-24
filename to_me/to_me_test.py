import time

import pytest
import torch
from to_me_imp import bipartite_soft_matching

def test_bipartite_soft_matching_with_valid_input():
    k = torch.randn(1, 100, 1408)
    k = k.to('cuda:0')
    k = k.half()
    r = 50
    merge, prune = bipartite_soft_matching(k, r, r_type='other', metric='dot_product', penalty_type='exponential')
    ti = time.time()
    result = merge(k)
    tf = time.time()
    time_merge = tf - ti
    print()
    print(result.shape)
    print(f'Time merge: {time_merge}')
    ti = time.time()
    result = prune(k)
    tf = time.time()
    time_prune = tf - ti
    print(result.shape)
    print(f'Time prune: {time_prune}')
    print()
    print(f'Time diference, merge - prune: {time_merge - time_prune}')


def test_bipartite_soft_matching_with_large_r():
    k = torch.randn(1, 4, 2)
    r = 3
    merge = bipartite_soft_matching(k, r)
    result = merge(k)
    assert result.shape == (1, 1, 2)

def test_bipartite_soft_matching_with_zero_r():
    k = torch.randn(1, 4, 2)
    r = 0
    merge = bipartite_soft_matching(k, r)
    result = merge(k)
    assert result.shape == (1, 4, 2)

def test_bipartite_soft_matching_with_negative_r():
    k = torch.randn(1, 4, 2)
    r = -1
    with pytest.raises(IndexError):
        merge = bipartite_soft_matching(k, r)
        merge(k)
        print()

def test_bipartite_soft_matching_with_non_tensor_input():
    k = [1, 2, 3]
    r = 1
    with pytest.raises(TypeError):
        merge = bipartite_soft_matching(k, r)
        merge(k)