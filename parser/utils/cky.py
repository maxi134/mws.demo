# import torch
# from supar.utils.fn import pad, stripe
# def decode_cky(scores, mask):
#     lens = mask[:, 0].sum(-1)
#     scores = scores.permute(1, 2, 0)
#     seq_len, seq_len,batch_size = scores.shape
#     s = scores.new_zeros(seq_len, seq_len, batch_size)
#     p_s = scores.new_zeros(seq_len, seq_len, batch_size).long()
#     p_l = scores.new_zeros(seq_len, seq_len, batch_size).long()

#     for w in range(1, seq_len):
#         n = seq_len - w
#         starts = p_s.new_tensor(range(n)).unsqueeze(0)
#         s_l, p = scores.diagonal(w).max(0)
#         p_l.diagonal(w).copy_(p)
 
#         if w == 1:
#             s.diagonal(w).copy_(s_l)
#             continue
#         # [n, w, batch_size]
#         s_s = stripe(s, n, w-1, (0, 1)) + stripe(s, n, w-1, (1, w), 0)
#         # [batch_size, n, w]
#         s_s = s_s.permute(2, 0, 1)
#         # [batch_size, n]
#         s_s, p = s_s.max(-1)
#         s.diagonal(w).copy_(s_s + s_l)
#         p_s.diagonal(w).copy_(p + starts + 1)

#     def backtrack(p_s, p_l, i, j):
#         if j == i + 1:
#             return [(i, j)]
#         split, label = p_s[i][j], p_l[i][j]
#         ltree = backtrack(p_s, p_l, i, split)
#         rtree = backtrack(p_s, p_l, split, j)
#         return [(i, j)] + ltree + rtree

#     p_s = p_s.permute(2, 0, 1).tolist()
#     p_l = p_l.permute(2, 0, 1).tolist()
#     trees = [backtrack(p_s[i], p_l[i], 0, length) for i, length in enumerate(lens.tolist())]

#     return trees

from supar.structs import ConstituencyCRF
def decode_cky(s_span, s_label, mask):
    span_preds = ConstituencyCRF(s_span, mask[:, 0].sum(-1)).argmax
    # label_preds = s_label.argmax(-1).tolist()
    return [[(i, j) for i, j in spans] for spans in span_preds]