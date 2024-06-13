import torch


def text_topk_pruning(video_tokens: torch.Tensor, text_tokens: torch.Tensor, k: int):
    """
    Returns the top_k video tokens that have the highest average cosine similarity with the text tokens.
    """

    # Normalize the tokens and store in new variables
    normalized_video_tokens = video_tokens / video_tokens.norm(p=2, dim=-1, keepdim=True)
    normalized_text_tokens = text_tokens / text_tokens.norm(p=2, dim=-1, keepdim=True)
    normalized_text_tokens = torch.nn.functional.pad(normalized_text_tokens, (0, normalized_video_tokens.shape[-1] - normalized_text_tokens.shape[-1]))
    normalized_text_tokens = normalized_text_tokens.half()

    sim = normalized_video_tokens @ normalized_text_tokens.transpose(-1, -2)  # shape becomes [batch_size, num_video_tokens, num_text_tokens]


    sim = torch.sum(sim, dim=-1)

    _, top_k_idx = torch.topk(sim, k, dim=-1)

    top_k_tokens = video_tokens[top_k_idx]

    return top_k_tokens
