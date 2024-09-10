# -*- coding: utf-8 -*-
# @ModuleName: semi_Markov
# @Function:
# @Author: Wxb
# @Time: 2023/4/21 16:37
import sys

import torch
import torch.autograd as autograd
from parser.utils.alg import score_function


def semi_Markov_loss_weak(span_scores, gold_spans, span_mask, mask, max_len, mode):
    batch_size, _, _ = span_scores.shape
    training = span_scores.requires_grad
    safe_negative_value = -1e9
    span_scores_mask = span_scores.masked_fill(~span_mask, safe_negative_value)
    logZ = semi_Markov_z(span_scores, mask, M=max_len)
    gold_scores = semi_Markov_z(span_scores_mask, mask, M=max_len)
    marginals = span_scores_mask  # 为了确保梯度计算正常工作
    if training and mode in ['evaluate', 'predict']:
        marginals, = autograd.grad(logZ, span_scores, retain_graph=training)
    # norm : sentence_size, word_size, char_size
    loss = (logZ - gold_scores) / mask.sum()
    # loss = torch.clamp(loss,min=1e-5,max=1e5)
    # loss = (logZ - gold_scores) / mask.sum()
    return loss, marginals


def semi_Markov_loss(span_scores, gold_spans, mask, max_len, mode):
    batch_size, _, _ = span_scores.shape
    training = span_scores.requires_grad
    gold_scores = score_function(span_scores, gold_spans, mask).sum()  # 返回值为一个浮点数，是一个标量
    logZ = semi_Markov_z(span_scores, mask, M=max_len)

    marginals = span_scores  # 为了确保梯度计算正常工作
    if training and mode in ['evaluate', 'predict']:
        marginals, = autograd.grad(logZ, span_scores, retain_graph=training)
    loss = (logZ - gold_scores) / mask.sum()
    return loss, marginals


def semi_Markov_z(span_scores, mask, M=None):  # 和正确的冲突mask掉
    batch_size, seq_len, _ = span_scores.shape
    lens = mask[:, 0].sum(dim=-1)
    logZ = span_scores.new_zeros(batch_size, seq_len).double()
    for i in range(1, seq_len):
        logZ[:, i] = torch.logsumexp(logZ[:, :i] + span_scores[:, :i, i], dim=-1)
    return logZ[torch.arange(batch_size), lens].sum()


def semi_Markov_z_span_mask(span_scores, span_mask, mask, M=None):
    '''
    弱标注
    '''
    batch_size, seq_len, _ = span_scores.shape
    lens = mask[:, 0].sum(dim=-1)
    if span_mask is not None:
        span_scores = span_scores.masked_fill(~span_mask, float('-inf'))
    logZ = span_scores.new_zeros(batch_size, seq_len).double()
    for i in range(1, seq_len):
        logZ[:, i] = torch.logsumexp(logZ[:, :i] + span_scores[:, :i, i], dim=-1)
    return logZ[torch.arange(batch_size), lens].sum()


@torch.no_grad()
def semi_Markov_y(span_scores, mask, M=10):
    """
    Chinese Word Segmentation with semi-Markov algorithm.

        Args:
            span_scores (Tensor(B, N, N)): (*, i, j) is score for span(i, j)
            mask (Tensor(B, N, N))
            M (int): default 10.

        Returns:
            segs (list[]): segmentation sequence
    """
    batch_size, seq_len, _ = span_scores.size()  # seq_len is maximum length
    # breakpoint()
    lens = mask[:, 0].sum(dim=-1)  # 添加约束之前的lens
    # lens = (mask.cumsum(dim=-1) > 0).long().sum(dim=-1)[:,0]
    chart = span_scores.new_zeros(batch_size, seq_len).double()  # 存储得分
    backtrace = span_scores.new_zeros(batch_size, seq_len, dtype=int)  # 存储索引
    for i in range(1, seq_len):
        t = max(0, i - M)
        max_score, max_index = torch.max(chart[:, t:i] + span_scores[:, t:i, i], dim=-1)
        chart[:, i], backtrace[:, i] = max_score, max_index + t
        # max_score, max_index = torch.max(chart[:, :i] + span_scores[:, :i, i], dim=-1)
        # chart[:, i], backtrace[:, i] = max_score, max_index
    backtrace = backtrace.tolist()
    segments = [traceback(each, length) for each, length in zip(backtrace, lens.tolist())]
    return segments


@torch.no_grad()
def semi_Markov_y_pos(span_scores, pos_scores, mask, M=10):
    """
    Chinese Word Segmentation with semi-Markov algorithm.

        Args:
            span_scores (Tensor(B, N, N)): (*, i, j) is score for span(i, j)
            mask (Tensor(B, N, N))
            M (int): default 10.

        Returns:
            segs (list[]): segmentation sequence
    """
    batch_size, seq_len, _ = span_scores.size()  # seq_len is maximum length
    lens = mask[:, 0].sum(dim=-1)
    # print(lens)
    chart = span_scores.new_zeros(batch_size, seq_len).double()
    backtrace = span_scores.new_zeros(batch_size, seq_len, dtype=int)

    for i in range(1, seq_len):
        t = max(0, i - M)
        max_score, max_index = torch.max(chart[:, t:i] + span_scores[:, t:i, i], dim=-1)
        chart[:, i], backtrace[:, i] = max_score, max_index + t

        # max_score, max_index = torch.max(chart[:, :i] + span_scores[:, :i, i], dim=-1)
        # chart[:, i], backtrace[:, i] = max_score, max_index

    backtrace = backtrace.tolist()
    segments = [traceback(each, length) for each, length in zip(backtrace, lens.tolist())]

    # pos_scores = pos_scores[mask]
    pred_pos = pos_scores.argmax(-1)
    # print(pred_pos.shape)
    pred_pos = [
        [
            (i, j, pred[i][j])
            for i, j in index
        ]
        for index, pred in zip(segments, pred_pos.tolist())
    ]

    return segments, pred_pos


def traceback(backtrace, length):
    res = []
    left = length
    while left:
        right = backtrace[left]
        res.append((right, left))
        left = right
    res.reverse()
    return res


def add_restrain(span_scores, gold_spans):
    batch_index, row_index, col_index = torch.where(gold_spans)
    for i in range(len(batch_index)):
        batch, start, end = batch_index[i], row_index[i], col_index[i]
        for k in range(start + 1, end):
            if start > 0:
                span_scores[batch, :start, k] = -torch.inf
            if end < len(gold_spans[batch]):
                span_scores[batch, k, end + 1:] = -torch.inf
    return span_scores
