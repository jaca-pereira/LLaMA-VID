import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_top_k_tokens(source_top_k_tokens, video):
    """ Plots the source patches of top k tokens in the video. """
    fig, axs = plt.subplots(1, source_top_k_tokens.shape[1], figsize=(20, 5))
    for i, ax in enumerate(axs):
        ax.imshow(video[source_top_k_tokens[:, i].long()])
        ax.axis("off")
    plt.show()


def text_topk_pruning(video_tokens: torch.Tensor, text_tokens: torch.Tensor, k: int, source: torch.Tensor=None, video: torch.Tensor=None):
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

    if source is not None:
        source_top_k_tokens = source[top_k_idx]
        plot_top_k_tokens(source_top_k_tokens, video)


    return top_k_tokens
