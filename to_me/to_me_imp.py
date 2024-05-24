import torch
import math

def bipartite_soft_matching (k: torch.Tensor , r:int , r_type='threshold', metric= 'dot_product', penalty_type='linear') -> torch.Tensor :
    """ Input is k from attention , size [ batch , tokens , channels ]. """
    k = k/k. norm (dim=-1 , keepdim=True)
    a, b = k[ ... , ::2 , :] , k[ ... , 1::2 , :]

    if metric == 'dot_product':
        scores = a @ b. transpose (-1 , -2)
    elif metric == 'cosine_similarity':
        scores = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    else:
        scores = torch.cdist(a, b, p=2)

    # Create a distance penalty matrix
    penalty = torch.abs(torch.arange(a.size(-2))[:, None] - torch.arange(b.size(-2))[None, :])
    penalty = penalty.to(k.device)  # Ensure penalty is on the same device as k

    if penalty_type == 'exponential':
        # Apply the exponential decay to the penalty
        penalty = torch.exp(-penalty)
    elif penalty_type == 'quadratic':
        penalty = torch.pow(penalty, 2)
    # Apply the distance penalty to the scores
    scores -= penalty

    scores [ ... , 0 , :] = - math . inf # don ’t merge cls token
    node_max , node_idx = scores . max ( dim =-1)

    if r_type == 'threshold':
        # Change here: select scores higher than r
        src_idx = (node_max > r).nonzero()[..., None]
        unm_idx = (node_max <= r).nonzero()[..., None]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        r = len(node_max[node_max > r])
    else:
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        unm_idx = unm_idx.sort(dim=-2)[0]  # Sort cls token back to idx 0

    def merge (x: torch . Tensor ) -> torch . Tensor :
        """ Input is of shape [ batch , tokens , channels ]. """
        src , dst = x[ ... , ::2 , :] , x[ ... , 1::2 , :]
        n , t1 , c = src.shape
        unm = src.gather( dim =-2 , index = unm_idx . expand (n , t1 - r , c))
        src = src.gather( dim =-2 , index = src_idx . expand (n , r , c))
        dst = dst.scatter_add(-2 , dst_idx . expand (n , r , c) , src )
        return torch.cat([ unm , dst ] , dim =-2)

    def prune (x: torch . Tensor ) -> torch . Tensor :
        """ Input is of shape [ batch , tokens , channels ]. """
        src , dst = x[ ... , ::2 , :] , x[ ... , 1::2 , :]
        n , t1 , c = src . shape
        unm = src . gather ( dim =-2 , index = unm_idx . expand (n , t1 - r , c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_(-2, dst_idx.expand(n, r, c), src)
        return torch.cat([unm, dst], dim=-2)

    return merge, prune


