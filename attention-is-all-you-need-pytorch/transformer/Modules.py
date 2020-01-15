import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"


class PositionalScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, rpr=None, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.rpr = rpr

    def forward(self, q, k, v, mask=None):
        # logits_part1 = tf.matmul(queries, keys, transpose_b=True)  # bs, hd, lq, lk   batch_size,heads,length_q,depth_v
        # queries = tf.reshape(tf.transpose(queries, [2, 0, 1, 3]), [lq, bs * hd, dk])  # lq, bs*hd, dk
        # logits_part2 = tf.matmul(queries, tf.transpose(rpr_k, [0, 2, 1]))  # lq, bs*hd, lk
        # logits_part2 = tf.reshape(tf.transpose(logits_part2, [1, 0, 2]), [bs, hd, lq, lk])
        # logits = logits_part1 + logits_part2  # bs, hd, lq, lk

        q_shape = torch.shape(q)
        bs, hd, lq, dk = q_shape[0], q_shape[1], q_shape[2], q_shape[3]
        lk = torch.shape(k)[2]
        dv = torch.shape(v)[3]

        if self.rpr is None:
            attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # b x n x lq x dv
        else:
            rpr_k, rpr_v = self.rpr['rpr_k'], self.rpr['rpr_v']
            attn1 = torch.matmul(q / self.temperature, k.transpose(2, 3))
            q = torch.reshape(torch.transpose(q, [2, 0, 1, 3]), [lq, bs * hd, dk])
            attn2 = torch.matmul(q,torch.transpose(rpr_k,[0, 2, 1]))
            attn2 = torch.reshape(torch.transpose(attn2, [1,0,2]),[bs, hd, lq, lk])
            attn = attn1 + attn2


        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        # outputs_part1 = tf.matmul(weights, values)  # bs, hd, lq, dv
        #
        # weights = tf.reshape(tf.transpose(weights, [2, 0, 1, 3]), [lq, bs * hd, lk])  # lq, bs*hd, lk
        # outputs_part2 = tf.matmul(weights, rpr_v)  # lq, bs*hd, dv
        # outputs_part2 = tf.reshape(tf.transpose(outputs_part2, [1, 0, 2]), [bs, hd, lq, dv])
        #
        # outputs = outputs_part1 + outputs_part2  # bs, hd, lq, dv
        # weights = tf.reshape(tf.transpose(weights, [1, 0, 2]), [bs, hd, lq, lk])  # bs, hd, lq, lk

        if self.rpr is None:
            output = torch.matmul(attn, v)
        else:
            output_1 = torch.matmul(attn, v)
            attn = torch.reshape(torch.transpose(attn, [2, 0, 1, 3]), [lq, bs * hd, lk])
            output_2 = torch.matmul(attn, rpr_v)
            output_2 = torch.reshape(torch.transpose(output_2, [1,0,2]), [bs, hd, lq, dv])
            output = output_1 + output_2
            attn = torch.reshape(torch.transpose(attn, [1,0,2]), [bs,hd,lq,lk])

        return output, attn
