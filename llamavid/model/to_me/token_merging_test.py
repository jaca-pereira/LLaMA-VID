import unittest

import torch

from llamavid.model.to_me.token_merging import bipartite_soft_matching, merge_wavg, merge_source, \
    kth_bipartite_soft_matching, random_bipartite_soft_matching, bipartite_soft_matching_threshold


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
        self.assertEqual(size_target, size)

    def test_bipartite_soft_matching_double(self):
        video = torch.randn(1, 300*256, 1408, device='cuda:0')
        video = video.half()
        r = 1.0
        merge, _ = bipartite_soft_matching_threshold(
            video,
            r,
            False,
            False,
        )
        x, size = merge_wavg(merge, video, None)
        print()
        print(size.size(-2))

        r = size.size(-2)//2
        merge, _ = bipartite_soft_matching(
            x,
            r,
            False,
            False,
        )
        x, size = merge_wavg(merge, x, None)
        print()
        print(size.size(-2))
        print()
        print(size[0, :10])

    def test_bipartite_soft_matching_threshold(self):
        video = torch.randn(1, 300*256, 1408, device='cuda:0')
        video = video.half()
        threshold = 0.1
        merge, _ = bipartite_soft_matching_threshold(
            video,
            threshold,
            False,
            False,
        )
        x, size = merge_wavg(merge, video, None)
        print()
        print(video.shape[-2])
        print(x.shape[-2])
        self.assertEqual(True, True)


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

    def test_kth_bipartite_matching(self):
        video = torch.randn(1, 300*256, 1408, device='cuda:0')
        k = 8
        merge, _ = kth_bipartite_soft_matching(video, k)
        x, size = merge_wavg(merge, video, None)
        size = x.size(-2)
        size_target = video.shape[-2]/k
        print()
        print(size)
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
