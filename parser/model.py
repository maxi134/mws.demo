# -*- coding: utf-8 -*-
import sys

from parser.modules import CHAR_LSTM, MLP, BertEmbedding, Biaffine, ElementWiseBiaffine, BiLSTM
from parser.modules.dropout import IndependentDropout, SharedDropout
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args
        self.pretrained = False
        # the embedding layer
        self.char_embed = nn.Embedding(num_embeddings=args.n_chars,
                                       embedding_dim=args.n_embed)
        n_lstm_input = args.n_embed  # 100
        if args.feat == 'bert': # 使用bert进行编码
            args.bert_model = "/pythonProject/bert-base-chinese"
            self.feat_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_embed,
                                            requires_grad=True)
            n_lstm_input += args.n_embed  # 200
        if self.args.feat in {'bigram', 'trigram'}:
            self.bigram_embed = nn.Embedding(num_embeddings=args.n_bigrams,
                                             embedding_dim=args.n_embed)
            n_lstm_input += args.n_embed  # 200
        if self.args.feat == 'trigram':
            self.trigram_embed = nn.Embedding(num_embeddings=args.n_trigrams,
                                              embedding_dim=args.n_embed)
            n_lstm_input += args.n_embed

        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_lstm_input,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the bert layer
        self.bert = BertEmbedding(model='D:/pythonProject/bert-base-chinese',
                                  n_layers=args.n_bert_layers,
                                  n_out=800,
                                  requires_grad=True)
        self.bert_dropout = SharedDropout(p=args.lstm_dropout)
        

        # the MLP layers
        self.mlp_span_l_ctb9 = MLP(n_in=args.n_lstm_hidden * 2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_span_r_ctb9 = MLP(n_in=args.n_lstm_hidden * 2,
                              n_out=args.n_mlp_span,  # 500
                              dropout=args.mlp_dropout)
        self.mlp_span_l_msr = MLP(n_in=args.n_lstm_hidden * 2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_span_r_msr = MLP(n_in=args.n_lstm_hidden * 2,
                              n_out=args.n_mlp_span,  # 500
                              dropout=args.mlp_dropout)
        self.mlp_span_l_ppd = MLP(n_in=args.n_lstm_hidden * 2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_span_r_ppd = MLP(n_in=args.n_lstm_hidden * 2,
                              n_out=args.n_mlp_span,  # 500
                              dropout=args.mlp_dropout)
        if args.joint:
            self.mlp_label_l = MLP(n_in=args.n_lstm_hidden * 2,
                                   n_out=args.n_mlp_label,
                                   dropout=args.mlp_dropout
                                   )
            self.mlp_label_r = MLP(n_in=args.n_lstm_hidden * 2,
                                   n_out=args.n_mlp_label,  # 100
                                   dropout=args.mlp_dropout
                                   )
        #========================================================================================================
        # the Biaffine layers
        self.span_attn_ctb9 = Biaffine(n_in=args.n_mlp_span,  # 500
                                  bias_x=True,
                                  bias_y=False)
        self.span_attn_msr = Biaffine(n_in=args.n_mlp_span,  # 500
                                      bias_x=True,
                                  bias_y=False)
        self.span_attn_ppd = Biaffine(n_in=args.n_mlp_span,  # 500
                                  bias_x=True,
                                  bias_y=False)
        if args.joint:
            self.label_attn = Biaffine(n_in=args.n_mlp_label,  # 100
                                       n_out=args.n_pos_labels,
                                       bias_x=True,
                                       bias_y=True)

        if args.link == 'mlp':
            # a representation that a fencepost is a split point
            self.mlp_span_s = MLP(n_in=args.n_lstm_hidden * 2,
                                  n_out=args.n_mlp_span,
                                  dropout=args.mlp_dropout)

            # scores for split points
            self.score_split = nn.Linear(args.n_mlp_span, 1)

        elif args.link == 'att':
            self.split_attn = ElementWiseBiaffine(n_in=args.n_lstm_hidden,
                                                  bias_x=True,
                                                  bias_y=False)

        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed_dict=None):
        embed = embed_dict['embed'] if isinstance(
            embed_dict, dict) and 'embed' in embed_dict else None
        if embed is not None:
            self.pretrained = True
            self.char_pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.char_embed.weight)
            if self.args.feat == 'bigram':
                embed = embed_dict['bi_embed']
                self.bi_pretrained = nn.Embedding.from_pretrained(embed)
                nn.init.zeros_(self.bigram_embed.weight)
            elif self.args.feat == 'trigram':
                bi_embed = embed_dict['bi_embed']
                tri_embed = embed_dict['tri_embed']
                self.bi_pretrained = nn.Embedding.from_pretrained(bi_embed)
                self.tri_pretrained = nn.Embedding.from_pretrained(tri_embed)
                nn.init.zeros_(self.bigram_embed.weight)
                nn.init.zeros_(self.trigram_embed.weight)
        return self

    def forward(self, feed_dict, dataset_name, link=None):
        chars = feed_dict["chars"]
        batch_size, seq_len = chars.shape
        # get the mask and lengths of given batch
        mask = chars.ne(self.pad_index)
        lens = mask.sum(dim=1).cpu()
        ext_chars = chars
        # set the indices larger than num_embeddings to unk_index
        if self.pretrained:
            ext_mask = chars.ge(self.char_embed.num_embeddings)
            ext_chars = chars.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        char_embed = self.char_embed(ext_chars)
        if self.pretrained:
            char_embed += self.char_pretrained(chars)

        if self.args.feat == 'bert':
            feats = feed_dict["feats"] # feats中包含了subwords,mask，Bert可以对其进行编码
            # # embedding 在Bert编码中不需要
            # feat_embed = self.feat_embed(*feats)
            # char_embed, feat_embed = self.embed_dropout(char_embed, feat_embed)
            # embed = torch.cat((char_embed, feat_embed), dim=-1)
        elif self.args.feat == 'bigram':
            bigram = feed_dict["bigram"]
            ext_bigram = bigram
            if self.pretrained:
                ext_mask = bigram.ge(self.bigram_embed.num_embeddings)
                ext_bigram = bigram.masked_fill(ext_mask, self.unk_index)
            bigram_embed = self.bigram_embed(ext_bigram)
            if self.pretrained:
                bigram_embed += self.bi_pretrained(bigram)
            char_embed, bigram_embed = self.embed_dropout(
                char_embed, bigram_embed)
            embed = torch.cat((char_embed, bigram_embed), dim=-1)
        elif self.args.feat == 'trigram':
            bigram = feed_dict["bigram"]
            trigram = feed_dict["trigram"]
            ext_bigram = bigram
            ext_trigram = trigram
            if self.pretrained:
                ext_mask = bigram.ge(self.bigram_embed.num_embeddings)
                ext_bigram = bigram.masked_fill(ext_mask, self.unk_index)
                ext_mask = trigram.ge(self.trigram_embed.num_embeddings)
                ext_trigram = trigram.masked_fill(ext_mask, self.unk_index)
            bigram_embed = self.bigram_embed(ext_bigram)
            trigram_embed = self.trigram_embed(ext_trigram)
            if self.pretrained:
                bigram_embed += self.bi_pretrained(bigram)
                trigram_embed += self.tri_pretrained(trigram)
            char_embed, bigram_embed, trigram_embed = self.embed_dropout(
                char_embed, bigram_embed, trigram_embed)
            embed = torch.cat(
                (char_embed, bigram_embed, trigram_embed), dim=-1)
        else:
            embed = self.embed_dropout(char_embed)[0]
        
        s_label = None
        if self.args.feat == 'bert':
            x = self.bert(*feats)
            x = x[:,1:,:]
            x = self.bert_dropout(x)
        else:
            x = pack_padded_sequence(embed, lens, True, False)
            x, _ = self.lstm(x)
            x, _ = pad_packed_sequence(x, True, total_length=seq_len)
            x = self.lstm_dropout(x)
            x_f, x_b = x.chunk(2, dim=-1)
            x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)

        if dataset_name == 'ctb':
            span_l = self.mlp_span_l_ctb9(x)
            span_r = self.mlp_span_r_ctb9(x)
            s_span = self.span_attn_ctb9(span_l, span_r)
            return s_span, s_label
        elif dataset_name == 'msr':
            span_l = self.mlp_span_l_msr(x)
            span_r = self.mlp_span_r_msr(x)
            s_span = self.span_attn_msr(span_l, span_r)
            return s_span, s_label
        elif dataset_name == 'ppd':
            span_l = self.mlp_span_l_ppd(x)
            span_r = self.mlp_span_r_ppd(x)
            s_span = self.span_attn_ppd(span_l, span_r)
            return s_span, s_label

            # if self.args.joint:
            #     label_l = self.mlp_label_l(x)
            #     label_r = self.mlp_label_r(x)
            #     # [batch_size, seq_len, seq_len, n_rels]
            #     s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)
            #     # (B, N, N), (B, N, 1)
            #
            # if link == 'mlp':
            #     # (B, L-1)
            #     span_s = self.mlp_span_s(x)
            #     s_link = self.score_split(span_s)
            # elif link == 'att':
            #     # s_link = self.split_attn(x_f[:, :-1], x_b[:, 1:]).unsqueeze(dim=-1)
            #     pass
            # else:
            #     s_link = None

        else:
            span_l_ctb = self.mlp_span_l_ctb9(x)
            span_r_ctb = self.mlp_span_r_ctb9(x)
            s_span_ctb = self.span_attn_ctb9(span_l_ctb, span_r_ctb)
            span_l_msr = self.mlp_span_l_msr(x)
            span_r_msr = self.mlp_span_r_msr(x)
            s_span_msr = self.span_attn_msr(span_l_msr, span_r_msr)
            span_l_ppd = self.mlp_span_l_ppd(x)
            span_r_ppd = self.mlp_span_r_ppd(x)
            s_span_ppd = self.span_attn_ppd(span_l_ppd, span_r_ppd)
            return s_span_ctb, s_span_msr, s_span_ppd, s_label


    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if self.pretrained:
            pretrained = {'embed': state_dict.pop('char_pretrained.weight')}
            if hasattr(self, 'bi_pretrained'):
                pretrained.update(
                    {'bi_embed': state_dict.pop('bi_pretrained.weight')})
            if hasattr(self, 'tri_pretrained'):
                pretrained.update(
                    {'tri_embed': state_dict.pop('tri_pretrained.weight')})
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)
