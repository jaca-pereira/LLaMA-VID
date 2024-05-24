import unittest

import torch

from to_me.token_merging import merge_wavg, bipartite_soft_matching, merge_source, kth_bipartite_soft_matching, \
    random_bipartite_soft_matching


class MyTestCase(unittest.TestCase):
    def test_wavg(self, half=False):
        video = torch.randn(1, 300*256, 1408, device='cuda:0')
        if half:
            video = video.half()
        r = 150*256
        merge, _ = bipartite_soft_matching(
            video,
            r,
            False,
            False,
        )
        x, size = merge_wavg(merge, video, None)
        size = x.size(-2)
        size_target = video.shape[-2] - r
        self.assertEqual(size, size_target)

    def test_source(self, half=False):
        video = torch.randn(1, 300*256, 1408, device='cuda:0')
        if half:
            video = video.half()
        r = 150*256
        merge, _ = bipartite_soft_matching(
            video,
            r,
            False,
            False,
        )
        source = merge_source(merge, video, None)

    def test_kth_bipartite_matching(self, half=False):
        video = torch.randn(1, 300*256, 1408, device='cuda:0')
        if half:
            video = video.half()
        k = 2
        merge, _ = kth_bipartite_soft_matching(video, k)
        x, size = merge_wavg(merge, video, None)
        size = x.size(-2)
        size_target = video.shape[-2]/k
        self.assertEqual(size, size_target)

    def test_random_bipartite_soft_matching(self, half=False):
        video = torch.randn(1, 300*256, 1408, device='cuda:0')
        if half:
            video = video.half()
        r = 150*256
        merge, _ = random_bipartite_soft_matching(video, r)
        x, size = merge_wavg(merge, video, None)
        size = x.size(-2)
        size_target = video.shape[-2] - r
        self.assertEqual(size, size_target)

if __name__ == '__main__':
    unittest.main(half = True)
