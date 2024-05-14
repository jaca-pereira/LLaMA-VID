import torch
from .fast_kmeans import batch_fast_kmedoids_with_split

def get_cluster_inter(cluster_num,
                      distance='euclidean',
                      threshold=1e-6,
                      iter_limit=80,
                      id_sort=True,
                      norm_p=2.0):
    return TokenClusterInter(cluster_num, distance, threshold, iter_limit, id_sort, norm_p)
class TokenClusterInter(torch.nn.Module):
    def __init__(self,
                 cluster_num=49,
                 distance='euclidean',
                 threshold=1e-6,
                 iter_limit=80,
                 id_sort=True,
                 norm_p=2.0):
        super().__init__()
        self.cluster_num = cluster_num
        self.distance = distance
        self.threshold = threshold
        self.iter_limit = iter_limit
        self.id_sort = id_sort
        self.norm_p = norm_p

    def forward(self, x):
        assert x.dim() == 3 or x.dim() == 4, "x does not have 3 or 4 dimensions."
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[2] % 2 == 1:
            x = x[:, :, 1:]
        batch_size, num_frames, num_tokens, width = x.shape
        x = x[:, :80]
        res_x = x
        # after_block_frames x [B, frame_duration, num_tokens - 1, width]
        frame_split = torch.split(res_x, 10, dim=1)
        res_tmp = torch.cat(frame_split, dim=0).contiguous().reshape(batch_size*10, -1, width)
        batch_index = torch.arange(res_tmp.shape[0], dtype=torch.long, device=x.device).unsqueeze(-1)

        assign, mediods_ids = batch_fast_kmedoids_with_split(res_tmp, 500,
                                                             distance=self.distance, threshold=self.threshold,
                                                             iter_limit=self.iter_limit,
                                                             id_sort=self.id_sort,
                                                             norm_p=self.norm_p,
                                                             split_size=10,
                                                             pre_norm=self.pre_norm)

        x = x[assign, ...]
        x = x.reshape(batch_size, num_frames, self.cluster_num, width)

        return x