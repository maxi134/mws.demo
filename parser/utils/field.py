# -*- coding: utf-8 -*-

from collections import Counter
from parser.utils.vocab import Vocab
from parser.utils.fn import tohalfwidth
from parser.utils.common import bos, eos, pos_label
import torch


class RawField(object):

    def __init__(self, name, fn=None):
        super(RawField, self).__init__()

        self.name = name
        self.fn = fn

    def __repr__(self):
        return f"({self.name}): {self.__class__.__name__}()"

    def preprocess(self, sequence):
        if self.fn is not None:
            sequence = self.fn(sequence)
        return sequence

    def transform(self, sequences):
        return [self.preprocess(sequence) for sequence in sequences]


class Field(RawField):

    def __init__(self, name, pad=None, unk=None, bos=None, eos=None,
                 lower=False, tohalfwidth=False, use_vocab=True, tokenize=None, fn=None):
        self.name = name
        self.pad = pad
        self.unk = unk
        self.bos = bos
        self.eos = eos

        self.pos_label = pos_label
        self.pos_label2id = {pos: index for index, pos in enumerate(self.pos_label)}
        self.pos_label_num = len(self.pos_label)

        self.lower = lower
        self.tohalfwidth = tohalfwidth
        self.use_vocab = use_vocab
        self.tokenize = tokenize
        self.fn = fn

        self.specials = [token for token in [pad, unk, bos, eos]
                         if token is not None]

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        if self.tohalfwidth:
            params.append(f"tohalfwidth={self.tohalfwidth}")
        s += f", ".join(params)
        s += f")"

        return s

    @property
    def pad_index(self):
        return self.specials.index(self.pad) if self.pad is not None else 0

    @property
    def unk_index(self):
        return self.specials.index(self.unk) if self.unk is not None else 0

    @property
    def bos_index(self):
        return self.specials.index(self.bos)

    @property
    def eos_index(self):
        return self.specials.index(self.eos)

    def preprocess(self, sequence):
        if self.fn is not None:
            sequence = self.fn(sequence)
        if self.tokenize is not None:
            sequence = self.tokenize(sequence,max_len=512)# 原来没有参数max_len
        try:
            if self.lower:
                sequence = [str.lower(token) for token in sequence]
        except TypeError as e:
            print(sequence)
        if self.tohalfwidth:
            sequence = [tohalfwidth(token) for token in sequence]

        return sequence

    def build(self, corpus, min_freq=1, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter(token
                          for sequence in sequences
                          for token in self.preprocess(sequence))
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)

        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)    # extend tokens in embed
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors
            self.embed /= torch.std(self.embed)

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        if self.use_vocab:
            sequences = [self.vocab.token2id(sequence)
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.bos_index] + sequence for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]

        return sequences


class NGramField(Field):
    def __init__(self, *args, **kwargs):
        self.n = kwargs.pop('n') if 'n' in kwargs else 1
        super(NGramField, self).__init__(*args, **kwargs)

    def build(self, corpus, min_freq=1, dict_file=None, embed=None):
        sequences = getattr(corpus, self.name)
        counter = Counter()
        sequences = [self.preprocess(sequence) for sequence in sequences]
        n_pad = self.n - 1
        for sequence in sequences:
            chars = list(sequence) + [eos] * n_pad
            bichars = ["".join(chars[i + s] for s in range(self.n))
                       for i in range(len(chars) - n_pad)]
            counter.update(bichars)
        if dict_file is not None:
            counter &= self.read_dict(dict_file)
        self.vocab = Vocab(counter, min_freq, self.specials, self.unk_index)
        if not embed:
            self.embed = None
        else:
            tokens = self.preprocess(embed.tokens)
            # if the `unk` token has existed in the pretrained,
            # then replace it with a self-defined one
            if embed.unk:
                tokens[embed.unk_index] = self.unk

            self.vocab.extend(tokens)
            self.embed = torch.zeros(len(self.vocab), embed.dim)
            self.embed[self.vocab.token2id(tokens)] = embed.vectors
            self.embed /= torch.std(self.embed)

    def read_dict(self, dict_file):
        word_list = dict()
        with open(dict_file, encoding='utf-8') as dict_in:
            for line in dict_in:
                line = line.split()
                if len(line) == 3:
                    word_list[line[0]] = 100
        return Counter(word_list)

    def __repr__(self):
        s, params = f"({self.name}): {self.__class__.__name__}(", []
        params.append(f"n={self.n}")
        if self.pad is not None:
            params.append(f"pad={self.pad}")
        if self.unk is not None:
            params.append(f"unk={self.unk}")
        if self.bos is not None:
            params.append(f"bos={self.bos}")
        if self.eos is not None:
            params.append(f"eos={self.eos}")
        if self.lower:
            params.append(f"lower={self.lower}")
        if not self.use_vocab:
            params.append(f"use_vocab={self.use_vocab}")
        if self.tohalfwidth:
            params.append(f"tohalfwidth={self.tohalfwidth}")
        s += f", ".join(params)
        s += f")"

        return s

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        n_pad = (self.n - 1)
        for sent_idx, sequence in enumerate(sequences):
            chars = list(sequence) + [eos] * n_pad
            sequences[sent_idx] = ["".join(chars[i + s] for s in range(self.n))
                                   for i in range(len(chars) - n_pad)]
        if self.use_vocab:
            sequences = [self.vocab.token2id(sequence)
                         for sequence in sequences]
        if self.bos:
            sequences = [[self.bos_index] + sequence for sequence in sequences]
        if self.eos:
            sequences = [sequence + [self.eos_index] for sequence in sequences]
        sequences = [torch.tensor(sequence) for sequence in sequences]

        return sequences


class SegmentField(Field):
    """[summary]

    Examples:
        >>> sentence = ["我", "喜欢", "这个", "游戏"]
        >>> sequence = [(0, 1), (1, 3), (3, 5), (5, 7)]
        >>> spans = field.transform([sequences])[0]  
        >>> spans
        tensor([[False,  True, False, False, False, False, False, False],
                [False, False, False,  True, False, False, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False,  True, False, False],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False,  True],
                [False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False]])
    """

    def build(self, corpus, min_freq=1):
        """do nothing

        """
        
        return

    # def transform(self, sequences):
    #     breakpoint()
    #     sequences = [self.preprocess(sequence) for sequence in sequences]
    #     spans = []
    #     for sequence in sequences:
    #         # sequence =[(0, 2), (2, 4), (4, 6), (6, 7), (7, 9), (9, 11), (11, 13), 13] 最后保留句子的最大长度
    #         seg, seq_len = sequence[:-1], sequence[-1] + 1
    #         span_chart = torch.full((seq_len, seq_len), self.pad_index).bool()
    #         if seg:
    #             for i, j in seg:
    #                 span_chart[i, j] = 1
    #         span_mask= self.add_restrain(span_chart)
    #         spans.append((span_chart,span_mask))
    #     return spans

    def transform(self, sequences):
        '''
        对baike数据集读取的标签拼接在一个矩阵中
        '''
        sequences = [self.preprocess(sequence) for sequence in sequences]
        spans = []
        for sequence in sequences:
            # sequence =[(0, 2), (2, 4), (4, 6), (6, 7), (7, 9), (9, 11), (11, 13), 13] 最后保留句子的最大长度
            if isinstance(sequence[0], list):
                sequence_ctb, sequence_msr, sequence_ppd = sequence[0], sequence[1], sequence[2]
                seg_ctb, seq_len = sequence_ctb[:-1], sequence_ctb[-1] + 1
                seg_msr, seq_len = sequence_msr[:-1], sequence_msr[-1] + 1
                seg_ppd, seq_len = sequence_ppd[:-1], sequence_ppd[-1] + 1
                span_chart_ctb = torch.full((seq_len, seq_len), self.pad_index).bool()
                if seg_ctb:
                    for i, j in seg_ctb:
                        span_chart_ctb[i, j] = 1
                span_mask_ctb = self.add_restrain(span_chart_ctb)
                # span_chart, span_mask = span_chart_ctb, span_mask_ctb

                span_chart_msr = torch.full((seq_len, seq_len), self.pad_index).bool()
                if seg_msr:
                    for i, j in seg_msr:
                        span_chart_msr[i, j] = 1
                span_mask_msr = self.add_restrain(span_chart_msr)
                # span_chart = torch.cat([span_chart, span_chart_msr], 0)
                # span_mask = torch.cat([span_mask, span_mask_msr], 0)

                span_chart_ppd = torch.full((seq_len, seq_len), self.pad_index).bool()
                if seg_ppd:
                    for i, j in seg_ppd:
                        span_chart_ppd[i, j] = 1
                span_mask_ppd = self.add_restrain(span_chart_ppd)
                #不拼接，增加维度
                span_chart = torch.stack((span_chart_ctb, span_chart_msr, span_chart_ppd), dim=0)
                span_mask = torch.stack((span_mask_ctb, span_mask_msr, span_mask_ppd), dim=0)

                # span_chart = torch.cat((span_chart_ctb, span_chart_msr, span_chart_ppd), dim=0)
                # span_mask = torch.cat((span_mask_ctb, span_mask_msr, span_mask_ppd), dim=0)

                # span_chart = torch.cat([span_chart, span_chart_ppd], 0)
                # span_mask = torch.cat([span_mask, span_mask_ppd], 0)
            else:
                seg, seq_len = sequence[:-1], sequence[-1] + 1
                span_chart = torch.full((seq_len, seq_len), self.pad_index).bool()
                if seg:
                    for i, j in seg:
                        span_chart[i, j] = 1
                span_mask = self.add_restrain(span_chart)
            spans.append((span_chart, span_mask))
        return spans


    def add_restrain(self,span_chart):
        '''
        只对产生冲突的进行了约束
        '''
        span_mask = torch.ones_like(span_chart, dtype=torch.bool)
        span_mask = torch.triu(span_mask, diagonal=0)
        eye_mask = torch.eye(span_mask.shape[0], dtype=torch.bool)
        span_mask[eye_mask] = False
        row_index, col_index = torch.where(span_chart)
        for i in range(len(row_index)):
            start, end = row_index[i], col_index[i]

            # 1. 标记所有开始于start之前且结束在start到end之间的span
            span_mask[:start,start+1:end] = False
            # 2. 标记所有开始于start和end之间并且结束在end之后的span
            span_mask[start+1:end,end+1:] = False
            # 3. 标记所有包含(start,end)但是又不等于的span
            span_mask[start,end+1:] = False
            span_mask[:start,end] = False
            # 4. 标记所有被(start,end)包含的span
            span_mask[start:end+1,start:end+1] = False
            span_mask[start,end] = True
        return span_mask

            


class PosField(Field):
    """[summary]

    Examples:
        >>> sentence = ["我", "喜欢", "这个", "游戏"]
        # >>> sequence = ['pos1', 'pos2', 'pos3', 'pos4']
        >>> sequence = [(0, 1, 'pos1'), (1, 3, 'pos2'), (3, 5, 'pos3'), (5, 7, 'pos4')]
        >>> pos = field.transform([sequences])[0]
        >>> pos
        tensor([id1, id2, id3, id4])
    """

    def build(self, corpus, min_freq=1):
        """do nothing

        """

        return

    def transform(self, sequences):
        sequences = [self.preprocess(sequence) for sequence in sequences]
        spans = []
        for sequence in sequences:
            seq_len = sequence[-1][1] + 1
            span_chart = torch.full((seq_len, seq_len), self.pad_index)
            for i, j, pos in sequence:
                span_chart[i, j] = self.pos_label2id[pos]
            spans.append(span_chart)

        return spans


class BertField(Field):
    def transform(self, sequences):
        subwords, lens = [], []
        sequences = [list(sequence)
                     for sequence in sequences]

        for sequence in sequences:
            # TODO bert 
            sequence = self.preprocess(sequence)  # 将字符转换为对应的id
            sequence = [piece if piece else self.preprocess(self.pad)
                        for piece in sequence]
            subwords.append(sequence)
        subwords = [torch.tensor(pieces) for pieces in subwords]
        mask = [torch.ones(len(pieces)).gt(0) for pieces in subwords]

        return list(zip(subwords, mask))
