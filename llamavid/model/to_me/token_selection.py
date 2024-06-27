# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

#TODO change header for "adapted from" and add the original license

import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def do_nothing(x: torch.Tensor, mode: str=None) -> torch.Tensor:
    return x

def kth_bipartite_soft_matching(
    metric: torch.Tensor, k: int
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with the two sets as (every kth element, the rest).
    If n is the number of tokens, resulting number of tokens will be n // z.

    Input size is [batch, tokens, channels].
    z indicates the stride for the first set.
    z = 2 is equivalent to regular bipartite_soft_matching with r = 0.5 * N
    """
    if k <= 1:
        return do_nothing, do_nothing

    def split(x):
        t_rnd = (x.shape[1] // k) * k
        x = x[:, :t_rnd, :].view(x.shape[0], -1, k, x.shape[2])
        a, b = (
            x[:, :, : (k - 1), :].contiguous().view(x.shape[0], -1, x.shape[-1]),
            x[:, :, (k - 1), :],
        )
        return a, b

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        r = a.shape[1]
        scores = a @ b.transpose(-1, -2)

        _, dst_idx = scores.max(dim=-1)
        dst_idx = dst_idx[..., None]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, _, c = src.shape
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return dst

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        dst = x

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c)).to(x.dtype)

        src = src.view(n, -1, (k - 1), c)
        dst = dst.view(n, -1, 1, c)

        out = torch.cat([src, dst], dim=-2)
        out = out.contiguous().view(n, -1, c)

        return out

    return merge, unmerge


def bipartite_soft_matching(
    metric: torch.Tensor,
    bucket_size: int = 4096
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    bucket_size indicates the size of the bucket for the bipartite matching.

    """


    with torch.no_grad():
        metric.div_(metric.norm(dim=-1, keepdim=True))

        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        # Get the number of items in `a` and `b`
        n = a.shape[0]
        padding_size = (bucket_size - (n % bucket_size)) % bucket_size  # Ensure padding_size is within [0, bucket_size)

        # Pad `a` and `b` to be divisible by `bucket_size`
        if padding_size > 0:
            a = F.pad(a, (0, 0, 0, padding_size), 'constant', 0)
            b = F.pad(b, (0, 0, 0, padding_size), 'constant', 0)

        n_padded = a.shape[0]
        n_buckets = n_padded // bucket_size

        # Reshape `a` and `b` into (n_buckets, bucket_size, d)
        a = a.view(n_buckets, bucket_size, -1)
        b = b.view(n_buckets, bucket_size, -1)

        batch_size = min(2**15, n_padded) # Avoid CUDA OOM
        num_batches = math.ceil(n_padded / batch_size)

        node_max = None
        node_idx = None
        stop = False
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_padded)
            scores_batch = torch.full((end - start, end - start), -math.inf, device=metric.device)
            n_buckets_batch = (end - start) // bucket_size
            # Compute dot product similarity within each bucket
            for j in range(n_buckets_batch):
                scores_batch[j * bucket_size:(j + 1) * bucket_size, j * bucket_size:(j + 1) * bucket_size] = a[j] @ b[j].transpose(1, 0)
            if end > n:
                n = n-start
                scores_batch = scores_batch[:n, :n]
                stop = True
            if node_max is None:
                node_max, node_idx = scores_batch.max(dim=-1)
            else:
                node_max_batch, node_idx_batch = scores_batch.max(dim=-1)
                node_max = torch.cat([node_max, node_max_batch], dim=0)
                node_idx = torch.cat([node_idx, node_idx_batch], dim=0)
            if stop:
                break

        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        src_idx = edge_idx # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        t1, c = src.shape
        src = src.gather(dim=-2, index=src_idx.expand(t1, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(t1, c), src, reduce=mode)
        return dst

    return merge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None, mode: str = "sum"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode=mode)
    size = merge(size, mode=mode)

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="sum")
    return source
