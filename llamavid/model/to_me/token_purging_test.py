import time
import unittest
import torch
from llamavid.model.to_me import token_pruning  # adjust this import according to your project structure

class TestTokenPurging(unittest.TestCase):
    def test_prune_top_k_tokens(self):
        # Create mock data
        video_tokens = torch.randn(1, 900*256, 1048, device='cuda:0')  # batch_size=10, num_video_tokens=5, dim=300
        text_tokens = torch.randn(1, 1000, 1048, device='cuda:0')  # batch_size=10, num_text_tokens=7, dim=300
        video_tokens  =video_tokens.half()
        text_tokens  =text_tokens.half()
        k = 3000

        t0 = time.time()
        # Call the function with the mock data
        result = token_purging.text_topk_pruning(video_tokens, text_tokens, k)
        t1 = time.time()
        print("Time taken: ", t1-t0)

        # Check the shape of the result
        self.assertEqual(result.shape, (1, k, 1048))

if __name__ == '__main__':
    unittest.main()
