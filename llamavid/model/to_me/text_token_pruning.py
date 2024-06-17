import math

import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_top_k_tokens(source_top_k_tokens: torch.Tensor, video: torch.Tensor):
    """ Plots the source patches of top k tokens in the video. """
    idx_patches = {}
    num_patches = 0
    for i, map_idx in enumerate(source_top_k_tokens):
        source_top_k_tokens_idx = torch.where(map_idx == 1)
        if len(source_top_k_tokens_idx) == 0 or len(source_top_k_tokens_idx[0]) == 0:
            continue
        idx_patches[i] = []
        for tpl in source_top_k_tokens_idx:
            for idx in tpl:
                idx_patches[i].append(idx.item())
                num_patches += 1

    grid_size = math.ceil(math.sqrt(num_patches))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    current_patch_n = 0
    for n, i in enumerate(idx_patches):
        idx_patches_i = torch.tensor(idx_patches[i], device=source_top_k_tokens.device)

        frame_indices = idx_patches_i // 256
        patch_indices = idx_patches_i % 256

        # Convert patch indices to 2D indices
        row_indices = patch_indices // 16
        col_indices = patch_indices % 16
        for j in range(len(frame_indices)):
            frame_index = frame_indices[j].item()
            row_index = row_indices[j].item() * 14
            col_index = col_indices[j].item() * 14
            patch = video[frame_index, :, max((row_index - 14), 0):min((row_index + 14), 224),
                    max((col_index - 14), 0):min((col_index + 14),
                                                 224)]  # assuming video is a [num_frames, channels, height, width] tensor
            # patch = F.interpolate(patch.unsqueeze(0), size=(112, 112), mode='bilinear', align_corners=False)
            patch = patch.squeeze(0)
            patch = patch.permute(1, 2, 0).to('cpu').numpy().astype(float)
            row_idx = current_patch_n // grid_size
            col_idx = current_patch_n % grid_size
            axs[row_idx, col_idx].imshow(patch)
            axs[row_idx, col_idx].axis('off')
            current_patch_n += 1
    plt.show()
    plt.close()
    exit(1)


def text_topk_pruning(video_tokens: torch.Tensor, text_tokens: torch.Tensor, k: int, video: torch.Tensor=None, source: torch.Tensor=None):
    """
    Returns the top_k video tokens that have the highest average cosine similarity with the text tokens.
    """

    # Normalize the tokens and store in new variables
    normalized_video_tokens = video_tokens / video_tokens.norm(p=2, dim=-1, keepdim=True)
    normalized_text_tokens = text_tokens / text_tokens.norm(p=2, dim=-1, keepdim=True)
    normalized_text_tokens = torch.nn.functional.pad(normalized_text_tokens, (0, normalized_video_tokens.shape[-1] - normalized_text_tokens.shape[-1]))

    sim = normalized_video_tokens @ normalized_text_tokens.transpose(-1, -2)  # shape becomes [batch_size, num_video_tokens, num_text_tokens]


    sim = torch.sum(sim, dim=-1)

    _, top_k_idx = torch.topk(sim, k, dim=-1)

    top_k_tokens = video_tokens[top_k_idx]

    if source is not None:
        source_top_k_tokens = source[top_k_idx]
        plot_top_k_tokens(source_top_k_tokens, video)

    return top_k_tokens
