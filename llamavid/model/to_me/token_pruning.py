import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def prune_top_k_tokens(video_tokens: torch.Tensor, text_tokens: torch.Tensor, k: int, labels: torch.Tensor = None):
    """
    Returns the top_k video tokens that have the highest average cosine similarity with the text tokens.
    """
    sqz = False
    if video_tokens.dim() == 2:
        video_tokens = video_tokens.unsqueeze(0)
        sqz = True
    # Normalize the tokens and store in new variables
    normalized_video_tokens = video_tokens / video_tokens.norm(p=2, dim=-1, keepdim=True)
    normalized_text_tokens = text_tokens / text_tokens.norm(p=2, dim=-1, keepdim=True)
    normalized_text_tokens = torch.nn.functional.pad(normalized_text_tokens, (0, normalized_video_tokens.shape[-1] - normalized_text_tokens.shape[-1]))
    normalized_text_tokens = normalized_text_tokens.half()
    # Calculate cosine similarity
    sim = normalized_video_tokens @ normalized_text_tokens.transpose(-1,
                                                                     -2)  # shape becomes [batch_size, num_video_tokens, num_text_tokens]

    # Calculate the sum cosine similarity
    sim = torch.sum(sim, dim=-1)  # shape becomes [batch_size, num_video_tokens]

    # Get the indices of the top_k video tokens
    _, top_k_idx = torch.topk(sim, k, dim=-1)  # shape becomes [batch_size, k]

    top_k_tokens = None
    top_k_labels = None

    for n in range(video_tokens.shape[0]):
        if top_k_tokens is None:
            top_k_tokens = video_tokens[n][top_k_idx[n]]
            top_k_tokens = top_k_tokens.unsqueeze(0)
            if labels is not None:
                top_k_labels = labels[top_k_idx[n]].unsqueeze(0)
        else:
            top_k_tokens = torch.cat((top_k_tokens, video_tokens[n][top_k_idx[n]].unsqueeze(0)), dim=0)
            if labels is not None:
                top_k_labels = torch.cat((top_k_labels, labels[top_k_idx[n]].unsqueeze(0)), dim=0)
    if sqz:
        top_k_tokens = top_k_tokens.squeeze(0)
        if labels is not None:
            top_k_labels = top_k_labels.squeeze(0)
    return top_k_tokens, top_k_labels


def plot_source_top_k_tokens(video_tokens: torch.Tensor, text_tokens: torch.Tensor, video: torch.Tensor,
                             source: torch.Tensor, k: int = 100):
    """
    Returns the top_k video tokens that have the highest average cosine similarity with the text tokens.
    """
    if text_tokens.dim() == 3:
        text_tokens = text_tokens.squeeze(0)
    # Normalize the tokens and store in new variables
    normalized_video_tokens = video_tokens / video_tokens.norm(p=2, dim=-1, keepdim=True)
    normalized_text_tokens = text_tokens / text_tokens.norm(p=2, dim=-1, keepdim=True)

    normalized_text_tokens = torch.nn.functional.pad(normalized_text_tokens, (0, normalized_video_tokens.shape[-1] - normalized_text_tokens.shape[-1]))
    normalized_text_tokens = normalized_text_tokens.half()

    # Calculate cosine similarity
    sim = normalized_video_tokens @ normalized_text_tokens.transpose(-1,
                                                                     -2)  # shape becomes [batch_size, num_video_tokens, num_text_tokens]

    # Calculate the sum cosine similarity
    sim = torch.sum(sim, dim=-1)  # shape becomes [batch_size, num_video_tokens]

    # Get the indices of the top_k video tokens
    _, top_k_idx = torch.topk(sim, k, dim=-1)  # shape becomes [batch_size, k]
    # get original source tokens for top_k_idx
    source_topk_idx = source[top_k_idx]
    for i in range(k):
        map_idx = source_topk_idx[i]
        top_k_idx_i = torch.where(map_idx == 1)
        if len(top_k_idx_i[0]) == 0:
            continue
        elif len(top_k_idx_i[0]) == 1:
            top_k_idx[i] = top_k_idx_i[0].item()
        else:
            top_k_idx[i] = top_k_idx_i[0][0].item()
    # Calculate frame indices and patch indices
    frame_indices = top_k_idx // 256
    patch_indices = top_k_idx % 256

    # Convert patch indices to 2D indices
    row_indices = patch_indices // 16
    col_indices = patch_indices % 16

    # plot the patches from video with the same indexes as top_k_idx tokens from video_tokens
    fig, axs = plt.subplots(4,25, figsize=(20, 10))
    for i in range(k):
        frame_index = frame_indices[i].item()
        row_index = row_indices[i].item() * 14
        col_index = col_indices[i].item() * 14
        patch = video[frame_index, :, max((row_index-14), 0):min((row_index + 14), 224),
                max((col_index-14), 0):min((col_index+14), 224)]  # assuming video is a [num_frames, channels, height, width] tensor
        #patch = F.interpolate(patch.unsqueeze(0), size=(112, 112), mode='bilinear', align_corners=False)
        patch = patch.squeeze(0)
        patch = patch.permute(1, 2, 0).to('cpu').numpy().astype(float)
        axs_x = i // 25
        axs_y = i % 25
        axs[axs_x][axs_y].imshow(patch)
        axs[axs_x][axs_y].axis('off')
    plt.show()
    plt.close()
    exit(1)
