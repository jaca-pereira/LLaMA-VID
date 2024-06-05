import matplotlib.pyplot as plt
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

def plot_source_top_k_tokens(video_tokens: torch.Tensor, text_tokens: torch.Tensor, video: torch.Tensor):
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
    _, top_k_idx = torch.topk(sum_sim, 10, dim=-1)  # shape becomes [batch_size, k]

    # Calculate frame indices and patch indices
    frame_indices = top_k_idx // 256
    patch_indices = top_k_idx % 256

    # Convert patch indices to 2D indices
    row_indices = patch_indices // 16
    col_indices = patch_indices % 16

    # plot the patches from video with the same indexes as top_k_idx tokens from video_tokens
    fig, axs = plt.subplots(1, 10, figsize=(20, 5))
    for i in range(10):
        frame_index = frame_indices[i].item()
        row_index = row_indices[i].item() * 14
        col_index = col_indices[i].item() * 14
        patch = video[frame_index, :, row_index:row_index + 14, col_index:col_index + 14]  # assuming video is a [num_frames, channels, height, width] tensor
        patch = F.interpolate(patch.unsqueeze(0), size=(112, 112), mode='bilinear', align_corners=False)
        patch = patch.squeeze(0)
        patch = patch.permute(1, 2, 0).to('cpu').numpy().astype(float)
        axs[i].imshow(patch)
        axs[i].axis('off')
    plt.show()
    plt.close()
    exit(1)
