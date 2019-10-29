import torch
import numpy as np
from torch import nn


def seq_len_to_mask(seq_len, max_len=None, mask_pos_to_true=True):
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


def taste_norm():
    i = torch.arange(36).resize_(2, 6, 3).to(torch.float)
    b, c, h = i.shape

    bn = nn.BatchNorm1d(c)  # (0, 1, 2, 18, 19, 20)

    in1 = nn.InstanceNorm1d(c)  # (0, 1, 2)
    in2 = nn.GroupNorm(c, c)
    assert torch.equal(in1(i).data, in2(i).data) == True

    ln1 = nn.LayerNorm(i.shape[1:])  # (0,1,2....16,17)
    ln2 = nn.GroupNorm(1, c)
    assert np.array_equal(ln1(i).to(torch.float16).detach().numpy(),
                          ln2(i).to(torch.float16).data.numpy()) == True  # (精度范围内相同）

    gn = nn.GroupNorm(3, c)  # (0,1,2,3,4,5)

    # print(i)
    print(gn(i))


def taste_transformer():
    m = nn.MultiheadAttention(embed_dim=20, num_heads=5, dropout=0.1)

    i = torch.randn(4, 5, 20)
    q = k = v = i.transpose(0, 1)
    key_padding_mask = seq_len_to_mask([5, 3, 1, 0], max_len=5)

    ao, aow = m(q, k, v, key_padding_mask=key_padding_mask)
    ao = ao.transpose(0, 1)

    print(aow)


# taste_norm()
# taste_transformer()
