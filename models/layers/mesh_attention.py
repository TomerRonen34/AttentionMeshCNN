import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MeshAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, attn_max_dist=None, dropout=0.1):
        super().__init__()
        self.attn_max_dist = attn_max_dist  # if None it is global attention
        self.multi_head_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout,self.attn_max_dist)

    @staticmethod
    def __create_global_edge_mask(x, meshes):
        """
        create binary mask of size [n_batch, max_n_edges, max_n_edges]
        for mesh i with E actual edges, mask[i,:E,:E] = 1
        """
        n_batch, max_n_edges = x.shape[0], x.shape[2]
        mask = torch.zeros(n_batch, max_n_edges, max_n_edges, dtype=torch.bool, device=x.device)
        for i_mesh in range(n_batch):
            n_edges = meshes[i_mesh].edges_count
            mask[i_mesh, :n_edges, :n_edges] = 1
        # TODO:  in shrec all meshes are always the same size, what happens in other datasets?
        print("all same?", len(set([m.edges_count for m in meshes])) == 1,
              "| mask full?", torch.all(mask).item(),
              "| max edges:", max_n_edges,
              "| edge counts:", [m.edges_count for m in meshes])
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
        attn: [batch, n_head, edges, edges]. last dim is softmaxed (sums to 1)
        mask: [batch, edges, edges]. which edges are valid (exist in mesh) and relevant to each other.
        """
        if mask is None:
            return torch.mean(attn, (1, 2))

        mask = mask.unsqueeze(1).float()  # For head axis broadcasting.
        attn_sum = torch.sum(attn * mask, (1, 2))
        valid_elements = torch.sum(mask, (1, 2))
        attn_per_edge = attn_sum / valid_elements
        return attn_per_edge

    def forward(self, x, meshes):
        """
        x: [batch, features, edges, 1]
        meshes: list of mesh objects
        """
        if self.attn_max_dist is not None:
            dist_matrices = [m.all_pairs_shortest_path() for m in meshes]
            mask = self.__create_local_edge_mask(x, meshes, self.attn_max_dist, dist_matrices)
        else:
            mask = self.__create_global_edge_mask(x, meshes)
            dist_matrices = None

        print("mean edges in attention mask:", mask.float().sum(1).mean().item())  # how many edges affect every edge in the attention?

        s = x.squeeze(3).transpose(1, 2)  # s is sequence-like x: [batch, edges, features]
        s, attn = self.multi_head_attention.forward(s, s, s, mask, dist_matrices)
        # attn: [batch, n_head, edges, edges]. last dim is softmaxed (sums to 1)
        x = s.transpose(1, 2).unsqueeze(3)
        attn_per_edge = self.__attention_per_edge(attn, mask)
        return x, attn, attn_per_edge


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module
    from https://github.com/jadore801120/attention-is-all-you-need-pytorch
    by Yu-Hsiang Huang
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, max_neighbor=5):
        super().__init__()

        self.attention_type = type
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = PositionalScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.base_rpr = Parameter(torch.Tensor(max_neighbor + 1, d_k))

    def forward(self, q, k, v, mask=None,dist_matrices=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        rpr = None
        if self.base_rpr is not None:
            rpr = create_rpr(self.base_rpr, dist_matrices)

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

        q, attn = self.attention(q, k, v, mask=mask, rpr=rpr)

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


class PositionalScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, rpr=None):
        """
        q: [batch, heads, seq, d_k]  queries
        k: [batch, heads, seq, d_k]  keys
        v: [batch, heads, seq, d_v]  values
        mask: [batch, 1, seq, seq]   for each edge, which other edges should be accounted for. "None" means all of them.
                                mask is important when using local attention, or when the meshes are of different sizes.
        rpr: [batch, seq, seq, d_k]  positional representations
        """
        if rpr is None:
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # b x n x lq x dv
        else:
            attn1 = torch.matmul(q / self.temperature, k.transpose(2, 3))

            _q = q.unsqueeze(-2)  # q turns from [batch, heads, seq, d_k] to [batch, heads, seq, 1, d_k]
            _rpr = rpr.transpose(-1, -2).unsqueeze(1)  # rpr turns from [batch, seq, seq, d_k] to [batch, 1, seq, d_k, seq]
            attn2 = torch.matmul(_q / self.temperature, _rpr)  # attn2: [heads, batch, seq, 1, seq]
            attn2 = attn2.squeeze(-2)    # attn2: [batch, heads, seq, seq]

            attn = attn1 + attn2

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


def create_rpr(base_rpr, dist_matrices):
    '''

    :param base_rpr:  max_neighbours+1, dk
    :param dist_matrices: list of dist_matrix , each of size nedges X nedges
    :return:
    '''
    bs = len(dist_matrices)  # batch size
    max_neighbor = base_rpr.shape[0] - 1
    d_k = base_rpr.shape[1]
    max_edges = max([dist_matrix.shape[0] for dist_matrix in dist_matrices])
    rpr = torch.zeros(bs,max_edges,max_edges,d_k)
    for i_mesh in range(bs):
        dist_matrix = dist_matrices[i_mesh]
        n_edges = dist_matrix.shape[0]
        dist_matrix[dist_matrix > max_neighbor] = max_neighbor
        rpr[i_mesh, :n_edges, :n_edges,:] = base_rpr[dist_matrix, :]
    return rpr

# outputs_part1 = tf.matmul(weights, values)  # bs, hd, lq, dv
#
# weights = tf.reshape(tf.transpose(weights, [2, 0, 1, 3]), [lq, bs * hd, lk])  # lq, bs*hd, lk
# outputs_part2 = tf.matmul(weights, rpr_v)  # lq, bs*hd, dv
# outputs_part2 = tf.reshape(tf.transpose(outputs_part2, [1, 0, 2]), [bs, hd, lq, dv])
#
# outputs = outputs_part1 + outputs_part2  # bs, hd, lq, dv
# weights = tf.reshape(tf.transpose(weights, [1, 0, 2]), [bs, hd, lq, lk])  # bs, hd, lq, lk