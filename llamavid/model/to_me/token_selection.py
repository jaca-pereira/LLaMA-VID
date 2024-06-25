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
    r: int,
    bucket_size: int = 4096
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    bucket_size indicates the size of the bucket for the bipartite matching.

    """

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[0]
    r = min(r, t // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]

        # apply penalty to tokens from different clips

        n = a.shape[0]
        padding_size = (bucket_size - (n % bucket_size)) % bucket_size

        # padd a and b to be divisible by bucket_size
        a_padded = F.pad(a, (0, 0, 0, padding_size))
        b_padded = F.pad(b, (0, 0, 0, padding_size))

        n_padded = a_padded.shape[0]
        n_buckets = n_padded // bucket_size

        # Reshape a_padded and b_padded into (n_buckets, bucket_size, d)
        a_buckets = a_padded.view(n_buckets, bucket_size, -1)
        b_buckets = b_padded.view(n_buckets, bucket_size, -1)

        # Compute dot product similarity within each bucket
        scores_padded = torch.full((n_padded, n_padded), -math.inf, device=metric.device)
        for i in range(n_buckets):
            a_normalized = F.normalize(a_buckets[i])
            b_normalized = F.normalize(b_buckets[i].transpose(1, 0))
            scores_padded[i * bucket_size:(i + 1) * bucket_size, i * bucket_size:(i + 1) * bucket_size] = a_normalized @ b_normalized

        # Trim the padding
        scores = scores_padded[:n, :n]

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=0)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

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
