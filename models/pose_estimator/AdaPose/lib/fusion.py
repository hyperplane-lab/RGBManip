import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None, position_embedding=None):
    d_k = query.size(-1)
    
    # scores (b,h,n,n)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    if position_embedding is not None:
        position_embedding = position_embedding.unsqueeze(1) 
        scores = scores + position_embedding

    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, fn_attention=attention, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.fn_attention = fn_attention
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None, position_embedding=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.fn_attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class ViewFusionBlock(nn.Module):
    def __init__(self, emb_dims=512, n_heads=4):
        super(ViewFusionBlock, self).__init__()
        self.fusion1 = MultiHeadedAttention(n_heads, emb_dims)
        self.fusion2 = MultiHeadedAttention(n_heads, emb_dims)

    def forward(self, *input):
        view1_feat = input[0]
        view2_feat = input[1]

        query = view1_feat.transpose(2, 1).contiguous()
        key = view2_feat.transpose(2, 1).contiguous()

        x = self.fusion1(query, key, key).transpose(2, 1).contiguous() # view2 as value, from view2 to view1
        y = self.fusion2(key, query, query).transpose(2, 1).contiguous() # view1 as value, from view1 to view2

        return x + view1_feat, y + view2_feat

class ViewFusion(nn.Module):
    def __init__(self, embed_dim = 512, num_heads = 4, depth = 4):
        super(ViewFusion, self).__init__()
        self.blocks = nn.ModuleList([
            ViewFusionBlock(emb_dims=embed_dim, n_heads=num_heads) for i in range(depth)])
    
    def forward(self, view1_feat, view2_feat):
        for i, blk in enumerate(self.blocks):
            view1_feat, view2_feat = blk(view1_feat, view2_feat)
        # x, y = self.blocks(view1_feat, view2_feat)

        return view1_feat, view2_feat

if __name__ == '__main__':

    model = ViewFusion(embed_dim=128, num_heads=4).cuda()
    key = torch.randn(2, 128, 1024).cuda()
    query = torch.randn(2, 128, 1024).cuda()

    x, y = model(query, key)

    print(x.shape, y.shape)
