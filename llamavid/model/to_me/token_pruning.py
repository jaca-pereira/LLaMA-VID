import torch
import torch.nn.functional as F


def prune_top_k_tokens(video_tokens: torch.Tensor, text_tokens: torch.Tensor, k: int, labels: torch.Tensor=None):
    """
    Returns the top_k video tokens that have the highest average cosine similarity with the text tokens.
    """

    # Normalize the tokens and store in new variables
    normalized_video_tokens = video_tokens / video_tokens.norm(p=2, dim=-1, keepdim=True)
    normalized_text_tokens = text_tokens / text_tokens.norm(p=2, dim=-1, keepdim=True)

    # Calculate cosine similarity
    sim = normalized_video_tokens @ normalized_text_tokens.transpose(-1, -2)  # shape becomes [batch_size, num_video_tokens, num_text_tokens]

    # Calculate the sum cosine similarity
    sum_sim = torch.sum(sim, dim=-1)  # shape becomes [batch_size, num_video_tokens]

    # Get the indices of the top_k video tokens
    _, top_k_idx = torch.topk(sum_sim, k, dim=-1)  # shape becomes [batch_size, k]

    # Gather the top k tokens from the original video tokens
    top_k_tokens = video_tokens[top_k_idx]
    if labels is not None:
        top_k_labels = labels[top_k_idx]
    else:
        top_k_labels = None
    return top_k_tokens, top_k_labels
