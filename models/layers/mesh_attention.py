import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from models.layers.mesh import Mesh
from multiprocessing import Pool


class MeshAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v,
                 attn_max_dist=None,
                 dropout=0.1,
                 use_values_as_is=False,
                 use_positional_encoding=False,
                 max_relative_position=8,
                 multiprocess_dist_matrices=False):
        super().__init__()
        self.attn_max_dist = attn_max_dist  # if None it is global attention
        self.use_positional_encoding = use_positional_encoding
        self.max_relative_position = max_relative_position
        self.multi_head_attention = MultiHeadAttention(
            n_head, d_model, d_k, d_v,
            dropout, use_values_as_is,
            use_positional_encoding,
            max_relative_position)

        if multiprocess_dist_matrices:
            self.pool = Pool()
            self.map_func = self.pool.map
        else:
            self.map_func = lambda iterable, args_list: list(map(iterable, args_list))

    def forward(self, x, meshes, dist_matrices=None):
        """
        x: [batch, features, edges, 1] or [batch, features, edges]
        meshes: list of mesh objects
        :param dist_matrices:  list of dist_matrix , each of size n_edges X n_edges.
                               if None and it's needed, it's calculated inside the forward function
        """
        singleton_dim = False
        if x.ndim == 4:
            singleton_dim = True
            x = x.squeeze(3)

        if dist_matrices is None:
            if self.attn_max_dist is not None or self.use_positional_encoding:
                pos_cutoff = self.max_relative_position if self.use_positional_encoding else None
                local_cutoff = self.attn_max_dist
                cutoff = max(filter(None, [pos_cutoff, local_cutoff]))
                tups = [(mesh, cutoff) for mesh in meshes]
                dist_matrices = self.map_func(Mesh.apsp_packed, tups)

        if self.attn_max_dist is not None:
            mask = self.__create_local_edge_mask(x, meshes, self.attn_max_dist, dist_matrices)
        else:
            mask = self.__create_global_edge_mask(x, meshes)

        if False and mask is not None and random.random() < 0.05:
            print("mean edges in attention mask:",
                  mask.float().sum(1).mean().item(),
                  "percentage of max_edges:",
                  mask.float().mean().item())  # how many edges affect every edge in the attention?

        s = x.transpose(1, 2)  # s is sequence-like x: [batch, edges, features]
        s, attn = self.multi_head_attention.forward(s, s, s, mask, dist_matrices)
        # attn: [batch, n_head, edges, edges]. last dim is softmaxed (sums to 1)
        x = s.transpose(1, 2)
        if singleton_dim:
            x = x.unsqueeze(3)
        attn_per_edge = self.__attention_per_edge(attn, mask)
        return x, attn, attn_per_edge, dist_matrices

    @staticmethod
    def __create_global_edge_mask(x, meshes):
        """
        create binary mask of size [n_batch, max_n_edges, max_n_edges]
        for mesh i with E actual edges, mask[i,:E,:E] = 1
        """
        n_batch, max_n_edges = x.shape[0], x.shape[2]
        n_edges_per_mesh = [_mesh.edges_count for _mesh in meshes]
        if all([n == max_n_edges for n in n_edges_per_mesh]):
            return None  # equivalent to a mask of all 1s

        mask = torch.zeros(n_batch, max_n_edges, max_n_edges, dtype=torch.bool, device=x.device)
        for i_mesh, n_edges in enumerate(n_edges_per_mesh):
            mask[i_mesh, :n_edges, :n_edges] = 1
        # print("all same?", len(set([m.edges_count for m in meshes])) == 1,
        #       "| mask full?", torch.all(mask).item(),
        #       "| max edges:", max_n_edges,
        #       "| edge counts:", [m.edges_count for m in meshes])
        return mask

    @staticmethod
    def __create_local_edge_mask(x, meshes, max_dist, dists_matrices):
        """
        create binary mask of size [n_batch, max_n_edges, max_n_edges]
        for mesh i with E actual edges, mask[i,:E,:E] = 1
        and masks away all connections that are more distant than max_dist
        """
        n_batch, max_n_edges = x.shape[0], x.shape[2]
        mask = torch.zeros(n_batch, max_n_edges, max_n_edges, dtype=torch.bool, device=x.device)
        for i_mesh in range(n_batch):
            n_edges = meshes[i_mesh].edges_count
            d_matrix = dists_matrices[i_mesh]
            mask[i_mesh, :n_edges, :n_edges] = torch.BoolTensor(d_matrix <= max_dist)
        return mask

    @staticmethod
    def __attention_per_edge(attn, mask):
        """
        This operation does not propagate gradients.
        attn: [batch, n_head, edges, edges]. last dim is softmaxed (sums to 1)
        mask: [batch, edges, edges]. which edges are valid (exist in mesh) and relevant to each other.
        """
        attn = attn.detach()
        if mask is None:
            return torch.mean(attn, (1, 2))

        mask = mask.detach()
        mask = mask.unsqueeze(1)  # For head axis broadcasting.
        attn_sum = torch.sum(attn.masked_fill(mask == 0, 0.), (1, 2))
        valid_elements = torch.sum(mask, (1, 2)).float()
        attn_per_edge = attn_sum / valid_elements
        return attn_per_edge


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    from https://github.com/jadore801120/attention-is-all-you-need-pytorch
    by Yu-Hsiang Huang.
    use_values_as_is is our addition.
    """

    def __init__(self, n_head, d_model, d_k, d_v,
                 dropout=0.1, use_values_as_is=False,
                 use_positional_encoding=False,
                 max_relative_position=8):
        super().__init__()

        self.attention_type = type
        self.n_head = n_head
        self.d_k = d_k

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        if not use_values_as_is:
            self.d_v = d_v
            self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        else:
            self.d_v = d_model
            self.w_vs = lambda x: self.__repeat_single_axis(x, -1, n_head)
            self.fc = lambda x: self.__average_head_results(x, n_head)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        positional_encoding = None
        if use_positional_encoding:
            positional_encoding = PositionalEncoding(max_relative_position, d_k)

        self.attention = PositionalScaledDotProductAttention(temperature=d_k ** 0.5,
                                                             positional_encoding=positional_encoding)

    @staticmethod
    def __repeat_single_axis(x, axis, n_rep):
        rep_sizes = [1] * x.ndim
        rep_sizes[axis] = n_rep
        x_rep = x.repeat(rep_sizes)
        return x_rep

    @staticmethod
    def __average_head_results(x, n_head):
        shape = list(x.shape)[:-1] + [n_head, -1]
        avg_x = x.view(shape).mean(-2)
        return avg_x

    def forward(self, q, k, v, mask=None, dist_matrices=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask, dist_matrices=dist_matrices)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    from https://github.com/jadore801120/attention-is-all-you-need-pytorch
    by Yu-Hsiang Huang
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionalEncoding(nn.Module):
    def __init__(self, max_pos, d_k):
        super().__init__()
        self.w_rpr = nn.Linear(d_k, max_pos + 1, bias=False)

    def __call__(self, q, dist_matrices):
        return self.forward(q, dist_matrices)

    def forward(self, q, dist_matrices):
        """
        :param q:  [batch, heads, seq, d_k]
        :param dist_matrices:  list of dist_matrix , each of size n_edges X nedges
        :return: resampled_q_dot_rpr: [batch, heads, seq, seq]
        """
        q_dot_rpr = self.w_rpr(q)
        attn_rpr = self.resample_rpr_product(q_dot_rpr, dist_matrices)
        return attn_rpr

    @staticmethod
    def resample_rpr_product(q_dot_rpr, dist_matrices):
        '''
        :param q_dot_rpr:  [batch, heads, seq, max_pos+1]
        :param dist_matrices: list of dist_matrix , each of size n_edges X nedges
        :return: resampled_q_dot_rpr: [batch, heads, seq, seq]
        '''
        bs, n_heads, max_seq, _ = q_dot_rpr.shape
        max_pos = q_dot_rpr.shape[-1] - 1

        seq_lens = np.array([d.shape[0] for d in dist_matrices])
        if (seq_lens == max_seq).all():
            pos_inds = np.stack(dist_matrices)
        else:
            pos_inds = np.ones((bs, max_seq, max_seq), dtype=np.int32) * np.iinfo(np.int32).max
            for i_b in range(bs):
                dist_matrix = dist_matrices[i_b]
                n_edges = dist_matrix.shape[0]
                pos_inds[i_b, :n_edges, :n_edges] = dist_matrix

        pos_inds[pos_inds > max_pos] = max_pos
        batch_inds = np.arange(bs)[:, None, None]
        edge_inds = np.arange(max_seq)[None, :, None]
        resampled_q_dot_rpr = q_dot_rpr[batch_inds, :, edge_inds, pos_inds].permute(0, 3, 1, 2)
        return resampled_q_dot_rpr


class PositionalScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention with optional positional encodings '''

    def __init__(self, temperature, positional_encoding=None, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.positional_encoding = positional_encoding

    def forward(self, q, k, v, mask=None, dist_matrices=None):
        """
        q: [batch, heads, seq, d_k]  queries
        k: [batch, heads, seq, d_k]  keys
        v: [batch, heads, seq, d_v]  values
        mask: [batch, 1, seq, seq]   for each edge, which other edges should be accounted for. "None" means all of them.
                                mask is important when using local attention, or when the meshes are of different sizes.
        rpr: [batch, seq, seq, d_k]  positional representations
        """
        attn_k = torch.matmul(q / self.temperature, k.transpose(2, 3))  # b x n x lq x dv
        if self.positional_encoding is None:
            attn = attn_k
        else:
            attn_rpr = self.positional_encoding(q / self.temperature, dist_matrices)
            attn = attn_k + attn_rpr

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn
