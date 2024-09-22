# -*- coding: utf-8 -*-

import os
import sys
from tqdm import tqdm
from parser.utils import Embedding
from parser.utils.alg import neg_log_likelihood, directed_acyclic_graph, crf
from parser.utils.common import pad, unk, bos, eos, pos_label
from parser.utils.corpus import CoNLL, Corpus
from parser.utils.field import BertField, Field, NGramField, SegmentField, PosField
from parser.utils.fn import get_spans, tensor2scalar
from parser.utils.metric import SegF1Metric
from parser.utils.semi_Markov import semi_Markov_loss, semi_Markov_loss_weak, semi_Markov_y, semi_Markov_y_pos
# maxi
from parser.utils.cky import decode_cky
from parser.utils.find_conflict import ConflictFinder
from parser.utils.merge_span import SpanMerger

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
import random
import numpy as np
import time
import functools


class CMD(object):

    def __call__(self, args):
        self.args = args
        if not os.path.exists(args.file):
            os.mkdir(args.file)
        if not os.path.exists(args.fields) or args.preprocess:
            print("Preprocess the data")

            self.CHAR = Field('chars', pad=pad, unk=unk,
                              bos=bos, eos=eos, lower=True)
            # TODO span as label, modify chartfield to spanfield
            self.SEG = SegmentField('segs')
            self.POS = PosField('pos') if args.joint else None
            if args.feat == 'bert':
                bert_model_path = "D:/pythonProject/bert-base-chinese"
                tokenizer = BertTokenizer.from_pretrained(bert_model_path)
                self.FEAT = BertField('bert',
                                      pad='[PAD]',
                                      bos='[CLS]',
                                      eos='[SEP]',
                                      tokenize=tokenizer.encode)  # maxi:原本是没有参数的
                self.fields = CoNLL(CHAR=(self.CHAR, self.FEAT),
                                    SEG=self.SEG, POS=self.POS)
            elif args.feat == 'bigram':
                self.BIGRAM = NGramField(
                    'bichar', n=2, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                # self.fields = CoNLL(CHAR=(self.CHAR, self.BIGRAM),
                #                     SEG=self.SEG)     # 原本的
                self.fields = CoNLL(CHAR=(self.CHAR, self.BIGRAM),
                                    SEG=self.SEG, POS=self.POS)
            elif args.feat == 'trigram':
                self.BIGRAM = NGramField(
                    'bichar', n=2, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                self.TRIGRAM = NGramField(
                    'trichar', n=3, pad=pad, unk=unk, bos=bos, eos=eos, lower=True)
                self.fields = CoNLL(CHAR=(self.CHAR,
                                          self.BIGRAM,
                                          self.TRIGRAM),
                                    SEG=self.SEG,
                                    POS=self.POS)
            else:
                self.fields = CoNLL(CHAR=self.CHAR,
                                    SEG=self.SEG,
                                    POS=self.POS)
            # 应该传入的是什么数据集
            embed_path = "/data3/maxi/Word_Sengment/mws/ChineseWordSegment/data/embed.ws"
            train = Corpus.load(embed_path, self.fields, args.joint)  # get: field.name, value
            embed = Embedding.load(
                'data/tencent.char.200.txt',
                args.unk) if args.embed else None
            self.CHAR.build(train, args.min_freq, embed)
            if hasattr(self, 'FEAT'):
                self.FEAT.build(train)
            if hasattr(self, 'BIGRAM'):
                embed = Embedding.load(
                    'data/tencent.bi.200.txt',
                    args.unk) if args.embed else None
                self.BIGRAM.build(train, args.min_freq,
                                  embed=embed,
                                  dict_file=args.dict_file)
            if hasattr(self, 'TRIGRAM'):
                embed = Embedding.load(
                    'data/tencent.tri.200.txt',
                    args.unk) if args.embed else None
                self.TRIGRAM.build(train, args.min_freq,
                                   embed=embed,
                                   dict_file=args.dict_file)
            # TODO
            self.SEG.build(train)
            if self.POS:
                self.POS.build(train)  # 新加的
            torch.save(self.fields, args.fields)
        else:
            self.fields = torch.load(args.fields)
            if args.feat == 'bert':
                self.CHAR, self.FEAT = self.fields.CHAR
            elif args.feat == 'bigram':
                self.CHAR, self.BIGRAM = self.fields.CHAR
            elif args.feat == 'trigram':
                self.CHAR, self.BIGRAM, self.TRIGRAM = self.fields.CHAR
            else:
                self.CHAR = self.fields.CHAR
            # TODO
            # self.SEG = self.fields.SEG   # 原本的
            self.SEG, self.POS = self.fields.SEG, self.fields.POS

        self.interval = [0] * 11
        self.right = [0] * 11
        self.all = [0] * 11
        self.metrics = [SegF1Metric() for _ in range(10)]

        # TODO loss funciton 
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        # self.criterion = nn.CrossEntropyLoss(reduction='sum')

        args.update({
            'n_chars': self.CHAR.vocab.n_init,
            'pad_index': self.CHAR.pad_index,
            'unk_index': self.CHAR.unk_index,
            'n_pos_labels': len(pos_label) if args.joint else 0
        })

        # TODO
        vocab = f"{self.CHAR}\n"
        if hasattr(self, 'FEAT'):
            args.update({
                'n_feats': self.FEAT.vocab.n_init,
            })
            vocab += f"{self.FEAT}\n"
        if hasattr(self, 'BIGRAM'):
            args.update({
                'n_bigrams': self.BIGRAM.vocab.n_init,
            })
            vocab += f"{self.BIGRAM}\n"
        if hasattr(self, 'TRIGRAM'):
            args.update({
                'n_trigrams': self.TRIGRAM.vocab.n_init,
            })
            vocab += f"{self.TRIGRAM}\n"

        print(f"Override the default configs\n{args}")
        print(vocab[:-1])

    # def train(self, ctb_train, msr_train, ppd_train, baike_ctb_train, baike_msr_train, baike_ppd_train):  # 应该在这里同时传入三个数据集的train
    #     pos=None
    #     self.model.train()
    #     torch.set_grad_enabled(True)
    #
    #     ctb_batchs = len(ctb_train)
    #     msr_batchs = len(msr_train)
    #     ppd_batchs = len(ppd_train)
    #     baike_ctb_batchs = len(baike_ctb_train)
    #
    #     all_batchs = ctb_batchs + msr_batchs + ppd_batchs + baike_ctb_batchs
    #     shuffles_indices = list(range(all_batchs))  # 可以通过在索引列表中多次添加相同数据集的索引，来实现多次加载一个数据集的目的
    #     random.shuffle(shuffles_indices)
    #
    #     ctb_loader = iter(ctb_train)
    #     msr_loader = iter(msr_train)
    #     ppd_loader = iter(ppd_train)
    #     baike_ctb_loader = iter(baike_ctb_train)
    #     baike_msr_loader = iter(baike_msr_train)
    #     baike_ppd_loader = iter(baike_ppd_train)
    #
    #     for index in shuffles_indices:
    #     # for index in tqdm(shuffles_indices, desc="Training", unit="batch"):
    #         if index < ctb_batchs:
    #             dataset_name = 'ctb'
    #             data = next(ctb_loader)
    #
    #         elif index < ctb_batchs + msr_batchs:
    #             dataset_name = 'msr'
    #             data = next(msr_loader)
    #
    #         elif index < ctb_batchs + msr_batchs + ppd_batchs:
    #             dataset_name = 'ppd'
    #             data = next(ppd_loader)
    #         else:
    #             dataset_name = 'baike'
    #             data_ctb = next(baike_ctb_loader)
    #             data_msr = next(baike_msr_loader)
    #             data_ppd = next(baike_ppd_loader)
    #
    #         if self.args.feat == 'bert':
    #             if self.args.joint:
    #                 chars, feats, segs, pos = data
    #             else:
    #                 if dataset_name in ['ctb', 'msr', 'ppd']:
    #                     chars, feats, segs_mask = data # segs[0]是gold_span,segs[1]是对应的span_mask TODO 修改保留多个gola_label
    #                     segs, span_mask = segs_mask
    #                     feed_dict = {"chars": chars, "feats": feats}
    #                 else:
    #                     chars, feats, segs_mask_ctb = data_ctb
    #                     segs_ctb, span_mask_ctb = segs_mask_ctb
    #                     _, _, segs_mask_msr = data_msr
    #                     segs_msr, span_mask_msr = segs_mask_msr
    #                     _, _, segs_mask_ppd = data_ppd
    #                     segs_ppd, span_mask_ppd = segs_mask_ppd
    #                     feed_dict = {"chars": chars, "feats": feats}
    #
    #
    #         elif self.args.feat == 'bigram':
    #             if self.args.joint:
    #                 chars, bigram, segs, pos = data
    #             else:
    #                 chars, bigram, segs = data
    #             feed_dict = {"chars": chars, "bigram": bigram}
    #         elif self.args.feat == 'trigram':
    #             if self.args.joint:
    #                 chars, bigram, trigram, segs, pos = data
    #             else:
    #                 chars, bigram, trigram, segs = data
    #             feed_dict = {"chars": chars,
    #                          "bigram": bigram, "trigram": trigram}
    #         else:
    #             if self.args.joint:
    #                 chars, segs, pos = data
    #             else:
    #                 chars, segs = data
    #             feed_dict = {"chars": chars}
    #
    #         self.optimizer.zero_grad()
    #         batch_size, seq_len = chars.shape
    #         lens = chars.ne(self.args.pad_index).sum(1) - 1
    #         mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
    #         mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)
    #
    #         if dataset_name  in ['ctb', 'msr', 'ppd']:
    #             s_span, s_label = self.model(feed_dict, dataset_name)
    #             loss, span_marginal = self.get_loss(s_span, segs, s_label, pos, mask, max_len=self.args.m_train)
    #         else:
    #             s_span_ctb, s_span_msr, s_span_ppd, s_label = self.model(feed_dict, dataset_name)
    #             loss_ctb, _ = self.get_loss_weak(s_span_ctb, segs_ctb, span_mask_ctb, s_label, pos, mask, max_len=self.args.m_train)
    #             loss_msr, _ = self.get_loss_weak(s_span_msr, segs_msr, span_mask_msr, s_label, pos, mask, max_len=self.args.m_train)
    #             loss_ppd, _ = self.get_loss_weak(s_span_ppd, segs_ppd, span_mask_ppd, s_label, pos, mask, max_len=self.args.m_train)
    #             loss = (loss_ctb + loss_msr + loss_ppd) // 3
    #             loss *= self.args.loss_weight
    #
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(self.model.parameters(),
    #                                  self.args.clip)
    #         self.optimizer.step()
    #         self.scheduler.step()
    #         torch.cuda.empty_cache()

    def train(self, ctb_train, msr_train, ppd_train, baike_train):  # 应该在这里同时传入三个数据集的train
        pos = None
        self.model.train()
        torch.set_grad_enabled(True)

        ctb_batchs = len(ctb_train)
        msr_batchs = len(msr_train)
        ppd_batchs = len(ppd_train)
        baike_batchs = len(baike_train)

        all_batchs = ctb_batchs + msr_batchs + ppd_batchs + baike_batchs
        shuffles_indices = list(range(all_batchs))  # 可以通过在索引列表中多次添加相同数据集的索引，来实现多次加载一个数据集的目的
        random.shuffle(shuffles_indices)

        ctb_loader = iter(ctb_train)
        msr_loader = iter(msr_train)
        ppd_loader = iter(ppd_train)
        baike_loader = iter(baike_train)

        for index in shuffles_indices:
            # for index in tqdm(shuffles_indices, desc="Training", unit="batch"):
            if index < ctb_batchs:
                dataset_name = 'ctb'
                data = next(ctb_loader)

            elif index < ctb_batchs + msr_batchs:
                dataset_name = 'msr'
                data = next(msr_loader)

            elif index < ctb_batchs + msr_batchs + ppd_batchs:
                dataset_name = 'ppd'
                data = next(ppd_loader)
            else:
                dataset_name = 'baike'
                data = next(baike_loader)

            if self.args.feat == 'bert':
                if self.args.joint:
                    chars, feats, segs, pos = data
                else:
                    chars, feats, segs_mask = data  # segs[0]是gold_span,segs[1]是对应的span_mask TODO 修改保留多个gola_label
                    segs, span_mask = segs_mask
                    if dataset_name not in ['ctb', 'msr', 'ppd']:
                        ctb_segs, msr_segs, ppd_segs = torch.unbind(segs, dim=1)
                        ctb_span_mask, msr_span_mask, ppd_span_mask = torch.unbind(span_mask, dim=1)
                        # ctb_segs, msr_segs, ppd_segs = torch.chunk(segs, 3, dim=1)
                        # ctb_span_mask, msr_span_mask, ppd_span_mask =  torch.chunk(span_mask, 3, dim=1)
                feed_dict = {"chars": chars, "feats": feats}

            elif self.args.feat == 'bigram':
                if self.args.joint:
                    chars, bigram, segs, pos = data
                else:
                    # chars, bigram, segs = data  原本的代码
                    chars, bigram, segs_mask = data
                    segs, span_mask = segs_mask
                    if dataset_name not in ['ctb', 'msr', 'ppd']:
                        ctb_segs, msr_segs, ppd_segs = torch.unbind(segs, dim=1)
                        ctb_span_mask, msr_span_mask, ppd_span_mask = torch.unbind(span_mask, dim=1)
                feed_dict = {"chars": chars, "bigram": bigram}

            elif self.args.feat == 'trigram':
                if self.args.joint:
                    chars, bigram, trigram, segs, pos = data
                else:
                    chars, bigram, trigram, segs = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                if self.args.joint:
                    chars, segs, pos = data
                else:
                    chars, segs = data
                feed_dict = {"chars": chars}

            self.optimizer.zero_grad()
            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)

            if dataset_name in ['ctb', 'msr', 'ppd']:
                s_span, s_label = self.model(feed_dict, dataset_name)
                loss, span_marginal = self.get_loss(s_span, segs, s_label, pos, mask, max_len=self.args.m_train)
                # print(f"all_loss={loss}")
            else:
                s_span_ctb, s_span_msr, s_span_ppd, s_label = self.model(feed_dict, dataset_name)
                loss_ctb, _ = self.get_loss_weak(s_span_ctb, ctb_segs, ctb_span_mask, s_label, pos, mask,
                                                 max_len=self.args.m_train)
                loss_msr, _ = self.get_loss_weak(s_span_msr, msr_segs, msr_span_mask, s_label, pos, mask,
                                                 max_len=self.args.m_train)
                loss_ppd, _ = self.get_loss_weak(s_span_ppd, ppd_segs, ppd_span_mask, s_label, pos, mask,
                                                 max_len=self.args.m_train)
                # loss_ctb, _ = self.get_loss(s_span_ctb, ctb_segs, s_label, pos, mask, max_len=self.args.m_train)
                # loss_msr, _ = self.get_loss(s_span_msr, msr_segs, s_label, pos, mask, max_len=self.args.m_train)
                # loss_ppd, _ = self.get_loss(s_span_ppd, ppd_segs, s_label, pos, mask, max_len=self.args.m_train)
                threshold = 1.0
                loss = (loss_ctb + loss_msr + loss_ppd) / 3
                if loss > 1:
                    loss *= 0.01
                # print(f"loss_ctb={loss_ctb},loss_msr={loss_msr},loss_ppd={loss_ppd},and loss_mean={loss}")
                # loss *= self.args.loss_weight

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.args.clip)
            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.empty_cache()

    # 训练时使用 三个数据集一起
    def evaluate_train(self, loader, dataset_name):
        if self.args.mode == 'evaluate':
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        self.model.eval()

        all_segs_pred, all_segs_gold = [], []
        total_loss, metric_span, metric_pos = 0, SegF1Metric(), SegF1Metric()
        for data in loader:
            # for data in tqdm(loader,desc="Evaluation_SWS",leave=False):
            pos = None
            if self.args.feat == 'bert':
                if self.args.joint:
                    chars, feats, segs, pos = data
                else:
                    chars, feats, segs = data
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                if self.args.joint:
                    chars, bigram, segs, pos = data
                else:
                    chars, bigram, segs = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                if self.args.joint:
                    chars, bigram, trigram, segs, pos = data
                else:
                    chars, bigram, trigram, segs = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                if self.args.joint:
                    chars, segs, pos = data
                else:
                    chars, segs = data
                feed_dict = {"chars": chars}

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)

            s_span, s_label = self.model(feed_dict, dataset_name)

            loss, marginals = self.get_loss(s_span, segs, s_label, pos, mask, max_len=self.args.m_test)
            total_loss += loss.item()
            pred_segs, pred_pos = self.decode(s_span, s_label, mask, self.args.m_test)

            gold_segs = [list(zip(*tensor2scalar(torch.nonzero(gold, as_tuple=True))))
                         for gold in segs]

            all_segs_pred.extend(pred_segs)
            all_segs_gold.extend(gold_segs)

            metric_span(pred_segs, gold_segs)
        total_loss /= len(loader)
        return total_loss, metric_span, metric_pos, all_segs_pred, all_segs_gold

    @torch.no_grad()
    def evaluate(self, loader, dataset_name=None):
        if self.args.mode == 'evaluate':
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        self.model.eval()
        total_loss, metric_span, metric_pos = 0, SegF1Metric(), SegF1Metric()
        total_loss_ctb, total_loss_msr, total_loss_ppd = 0, 0, 0
        all_segs = []
        for data in loader:
            pos = None
            if self.args.feat == 'bert':
                if self.args.joint:
                    chars, feats, segs, pos = data
                else:
                    chars, feats, segs_mask = data
                    segs, span_mask = segs_mask
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                if self.args.joint:
                    chars, bigram, segs, pos = data
                else:
                    # chars, bigram, segs = data
                    chars, bigram, segs_mask = data
                    segs, span_mask = segs_mask
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                if self.args.joint:
                    chars, bigram, trigram, segs, pos = data
                else:
                    chars, bigram, trigram, segs = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                if self.args.joint:
                    chars, segs, pos = data
                else:
                    chars, segs = data
                feed_dict = {"chars": chars}

            batch_size, seq_len = chars.shape
            flag = [False] * batch_size
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
            mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)

            s_span_ctb, s_label = self.model(feed_dict, dataset_name='ctb')
            s_span_msr, s_label = self.model(feed_dict, dataset_name='msr')
            s_span_ppd, s_label = self.model(feed_dict, dataset_name='ppd')

            loss_ctb, span_marginals_ctb = self.get_loss(s_span_ctb, segs, s_label, pos, mask, max_len=self.args.m_test)
            loss_msr, span_marginals_msr = self.get_loss(s_span_msr, segs, s_label, pos, mask, max_len=self.args.m_test)
            loss_ppd, span_marginals_ppd = self.get_loss(s_span_ppd, segs, s_label, pos, mask, max_len=self.args.m_test)

            total_loss_ctb += loss_ctb.item()
            total_loss_msr += loss_msr.item()
            total_loss_ppd += loss_ppd.item()
            total_loss += total_loss_ctb + total_loss_msr + total_loss_ppd

            pred_segs_ctb, pred_pos = self.decode(s_span_ctb, s_label, mask, self.args.m_test)
            pred_segs_msr, pred_pos = self.decode(s_span_msr, s_label, mask, self.args.m_test)
            pred_segs_ppd, pred_pos = self.decode(s_span_ppd, s_label, mask, self.args.m_test)

            gold_segs = [list(zip(*tensor2scalar(torch.nonzero(gold, as_tuple=True))))
                         for gold in segs]

            # ========================================================================
            # # 统计不同的解码层得到的解码结果中，有多少发生了冲突,并根据冲突找到CKY的正确解码结果
            # # 很奇怪，去掉这部分之后性能会变差 因为会将冲突的部分修改为负无穷，导致预测不到，这部分应该修改一下
            span_merge = []
            all_seg = []
            for i in range(batch_size):
                # 找到存在的冲突
                conflict_finder = ConflictFinder(pred_segs_ctb[i], pred_segs_msr[i], pred_segs_ppd[i])
                span, span_1, span_2, span_3 = conflict_finder.find_conflicts()  # span_3中存放了所有存在冲突的标签
                span_merge.append([span, [span_1, span_2, span_3]])
            # =================================================================================
            # baseline
            #     tmp = span_1+span_2+span_3
            #     all_seg.append(tmp)
            # metric_span(all_seg, gold_segs)
            # all_segs.extend(all_seg)
            # =================================================================================

            # span_marginals = (span_marginals_ctb + span_marginals_msr + span_marginals_ppd) / 3  # 取平均值
            # 将取平均设置值为取最大，看看效果
            span_marginals = torch.max(torch.max(span_marginals_ctb, span_marginals_msr), span_marginals_ppd)
            span_marginals_restrain = self.add_restraint(span_marginals, span_merge)
            pred_segs = self.decode_CKY(s_span=span_marginals_restrain, s_label=None, mask=mask)

            # =================================================================================
            # 根据三个解码结果进行解码，最终效果最好
            segs_cky = []
            for i in range(batch_size):
                tmp_cky = self.pruning(pred_segs_ctb[i], pred_segs_msr[i], pred_segs_ppd[i], pred_segs[i])
                segs_cky.append(tmp_cky)

            # =================================================================================
            # 将分割结果分为三类进行合并，效果差0.1
            # for i in range(batch_size):
            #     # 找到根据我的写法得到的CKY解码结果
            #     span_merger = SpanMerger(pred_segs_ctb[i], pred_segs_msr[i], pred_segs_ppd[i])
            #     tmp_cky = span_merger.merge_span(span_merge[i], pred_segs[i])
            #     segs_cky.append(tmp_cky)
            # =================================================================================
            all_segs.extend(segs_cky)
            metric_span(segs_cky, gold_segs)
            torch.cuda.empty_cache()
        total_loss /= (len(loader) * 3)
        # # TODO metric 总的loss，F1值
        return total_loss, metric_span, metric_pos, all_segs

    @torch.no_grad()
    def predict(self, loader):
        if self.args.mode == 'predict':
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        self.model.eval()
        pos = None
        result_cky = []
        # for data in loader:
        for data in tqdm(loader, desc="Prediction", leave=False):
            if self.args.feat == 'bert':
                if self.args.joint:
                    chars, feats, segs, pos = data
                else:
                    chars, feats, segs_mask = data  # segs[0]是gold_span,segs[1]是对应的span_mask
                    segs, span_mask = segs_mask
                feed_dict = {"chars": chars, "feats": feats}

            elif self.args.feat == 'bigram':
                chars, bigram, segs_mask = data
                segs, span_mask = segs_mask
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars = data
                feed_dict = {"chars": chars}

            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)  # len(lens)=12
            mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)  # torch.Size([12, 217, 217]) 对mask矩阵进一步的约束
            s_span_ctb, s_label = self.model(feed_dict, dataset_name='ctb')
            s_span_msr, s_label = self.model(feed_dict, dataset_name='msr')
            s_span_ppd, s_label = self.model(feed_dict, dataset_name='ppd')
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            # 添加约束应该是在得分矩阵，而不是mask，mask只有一个作用就是限制句子的长度，尝试只约束CKY的解码，这里添加的约束有错误
            # s_span_ctb = update_span_score_with_segs(segs,s_span_ctb)
            # s_span_msr = update_span_score_with_segs(segs,s_span_msr)
            # s_span_ppd = update_span_score_with_segs(segs,s_span_ppd)
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
            pred_segs_ctb, _ = self.decode(s_span_ctb, s_label, mask, self.args.m_test)  #[[(0, 2), (2, 4), (4, 5), (5, 7), (7, 12)]]
            pred_segs_msr, _ = self.decode(s_span_msr, s_label, mask, self.args.m_test)
            pred_segs_ppd, _ = self.decode(s_span_ppd, s_label, mask, self.args.m_test)
            # 与测试不需要gold
            # gold_segs = [list(zip(*tensor2scalar(torch.nonzero(gold, as_tuple=True))))
            #              for gold in segs]
            # #!!!!!!!!!!!!!!!!!!!!!!!!!!预测词典数据集时使用
            # s_span = (s_span_ctb+s_span_msr+s_span_ppd)/3
            # pred_segs = self.decode_CKY(s_span=s_span,s_label=None,mask=mask)
            # #!!!!!!!!!!!!!!!!!!!!!!!!!!
            span_merge = []
            for i in range(batch_size):
                conflict_finder = ConflictFinder(pred_segs_ctb[i], pred_segs_msr[i], pred_segs_ppd[i])
                span, span_1, span_2, span_3 = conflict_finder.find_conflicts()
                # 根据span对概率矩阵添加限制，使用概率矩阵和CKY解码
                span_merge.append([span, [span_1, span_2, span_3]])
            loss_ctb, span_marginals_ctb = \
                self.get_loss(s_span_ctb, segs, s_label, pos, mask, max_len=self.args.m_test)
            loss_msr, span_marginals_msr = \
                self.get_loss(s_span_msr, segs, s_label, pos, mask, max_len=self.args.m_test)
            loss_ppd, span_marginals_ppd = \
                self.get_loss(s_span_ppd, segs, s_label, pos, mask, max_len=self.args.m_test)
            torch.set_printoptions(threshold=10_000)
            span_marginals = (span_marginals_ctb + span_marginals_msr + span_marginals_ppd) / 3
            span_marginals_restrain = self.add_restraint(span_marginals, span_merge)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # span_marginals_restrain = update_span_score_with_segs(segs,span_marginals_restrain)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            pred_segs = self.decode_CKY(s_span=span_marginals_restrain, s_label=None, mask=mask)

            # TODO 这个简直方法是错误的
            segs_cky = []
            for i in range(batch_size):
                tmp_cky = self.pruning(pred_segs_ctb[i], pred_segs_msr[i], pred_segs_ppd[i], pred_segs[i])
                segs_cky.append(tmp_cky)


            # result_cky.extend(segs_cky)  # 原本的输出结果
            # 在输出的结果中添加边缘概率
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            segs_cky_marginal = []
            for batch_index, segs in enumerate(segs_cky):
                tmp = []
                for start, end in segs:
                    marginal_prob = span_marginals_restrain[batch_index, start, end].item()
                    marginal_prob = round(marginal_prob, 2)
                    tmp.append((start, end, marginal_prob))
                segs_cky_marginal.append(tmp)
            result_cky.extend(segs_cky_marginal)

            marginal_pred_segs_ctb = []
            for batch_index, segs in enumerate(pred_segs_ctb):
                tmp = []
                for start, end in segs:
                    marginal_prob = span_marginals_ctb[batch_index, start, end].item()
                    marginal_prob = round(marginal_prob, 2)
                    tmp.append((start, end, marginal_prob))
                marginal_pred_segs_ctb = tmp
            pred_segs_ctb = marginal_pred_segs_ctb

            marginal_pred_segs_msr = []
            for batch_index, segs in enumerate(pred_segs_msr):
                tmp = []
                for start, end in segs:
                    marginal_prob = span_marginals_msr[batch_index, start, end].item()
                    marginal_prob = round(marginal_prob, 2)
                    tmp.append((start, end, marginal_prob))
                marginal_pred_segs_msr = tmp
            pred_segs_msr = marginal_pred_segs_msr
            marginal_pred_segs_ppd = []
            for batch_index, segs in enumerate(pred_segs_ppd):
                tmp = []
                for start, end in segs:
                    marginal_prob = span_marginals_ppd[batch_index, start, end].item()
                    marginal_prob = round(marginal_prob, 2)
                    tmp.append((start, end, marginal_prob))
                marginal_pred_segs_ppd = tmp
            pred_segs_ppd = marginal_pred_segs_ppd
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            result_demo = []
            result_demo.append([result_cky[0], pred_segs_ctb, pred_segs_msr, pred_segs_ppd]) # 只支持单个的句子
        return result_demo

    @torch.no_grad()
    def predict_cws(self, loader):
        if self.args.mode == 'predict':
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        self.model.eval()
        all_segs, all_marginal = [], []
        # for data in loader:
        for data in tqdm(loader, desc="Prediction", leave=False):
            if self.args.feat == 'bert':
                # chars, feats, segs = data
                # feed_dict = {"chars": chars, "feats": feats}
                if self.args.joint:
                    chars, feats, segs, pos = data
                else:
                    chars, feats, segs_mask = data  # segs[0]是gold_span,segs[1]是对应的span_mask TODO 修改保留多个gola_label
                    segs, span_mask = segs_mask
                feed_dict = {"chars": chars, "feats": feats}
            elif self.args.feat == 'bigram':
                chars, bigram, segs = data
                # chars, bigram = data
                feed_dict = {"chars": chars, "bigram": bigram}
            elif self.args.feat == 'trigram':
                chars, bigram, trigram = data
                feed_dict = {"chars": chars,
                             "bigram": bigram, "trigram": trigram}
            else:
                chars = data
                feed_dict = {"chars": chars}

            pos = None
            batch_size, seq_len = chars.shape
            lens = chars.ne(self.args.pad_index).sum(1) - 1
            mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)  # len(lens)=12
            mask = mask & mask.new_ones(seq_len - 1, seq_len - 1).triu_(1)  # torch.Size([12, 217, 217])

            # ======================选择分词器=========================================
            # s_span, s_label = self.model(feed_dict,dataset_name='ctb')
            # s_span, s_label = self.model(feed_dict, dataset_name='msr')
            s_span, s_label = self.model(feed_dict, dataset_name='ppd')
            # ======================选择分词器=========================================

            # ==========对s_span添加约束，将与自然标注产生交叉的修改为-inf==========================
            s_span = update_span_score_with_segs(segs, s_span)

            # ==========对s_span添加约束，将与自然标注产生交叉的修改为-inf==========================
            pred_segs, _ = self.decode(s_span, s_label, mask, self.args.m_test)
            span_loss, span_marginal = self.get_loss(s_span, segs, s_label, pos, mask, max_len=self.args.m_test)
            all_marginal.extend(span_marginal)  # 无用的代码

            # all_segs.extend(pred_segs)  # 不带边缘概率的输出

            # 连带边缘概率一起输出 TODO !!!
            marginal_pred_segs = []
            for batch_index, segs in enumerate(pred_segs):
                tmp = []
                for start, end in segs:
                    marginal_prob = span_marginal[batch_index, start, end].item()
                    marginal_prob = round(marginal_prob, 2)
                    tmp.append((start, end, marginal_prob))
                marginal_pred_segs.append(tmp)
            all_segs.extend(marginal_pred_segs)
            # TODO !!!
        return all_segs

    def get_loss_weak(self, s_span, segs, span_mask, s_label, pos, mask, max_len):
        span_loss, span_marginals = semi_Markov_loss_weak(s_span, segs, span_mask, mask, max_len, self.args.mode)
        if self.args.joint:
            s_label, pos = s_label[mask], pos[mask]
            s_label, pos = s_label[segs[mask]], pos[segs[mask]]
            alpha = 0.7  # [0.1, 0.3, 0.5, 0.7]
            pos_loss = self.criterion(s_label, pos)
            loss = (1 - alpha) * span_loss + alpha * pos_loss

            return loss, span_marginals
        else:
            return span_loss, span_marginals

    def get_loss(self, s_span, segs, s_label, pos, mask, max_len):
        span_loss, span_marginals = semi_Markov_loss(s_span, segs, mask, max_len, self.args.mode)
        if self.args.joint:
            s_label, pos = s_label[mask], pos[mask]
            s_label, pos = s_label[segs[mask]], pos[segs[mask]]
            alpha = 0.7  # [0.1, 0.3, 0.5, 0.7]
            pos_loss = self.criterion(s_label, pos)
            loss = (1 - alpha) * span_loss + alpha * pos_loss

            return loss, span_marginals
        else:
            return span_loss, span_marginals

    @staticmethod
    def decode(s_span, s_label, mask, max_len):
        pred_pos = None
        if s_label is None:
            pred_spans = semi_Markov_y(s_span, mask, M=max_len)
        else:
            pred_spans, pred_pos = semi_Markov_y_pos(s_span, s_label, mask, M=max_len)
        return pred_spans, pred_pos

    @staticmethod  # 不考虑联合建模怎么写
    def decode_CKY(s_span, s_label, mask):
        pred_spans = decode_cky(s_span, s_label, mask)
        return pred_spans

    def collect(self, pred_segs, gold_segs, span_marginals,
                pred_poses=None, gold_poses=None, pos_marginals=None):
        from collections import Counter

        span_res, pos_res = [], []  # [prob, bool]
        iteration = zip(pred_segs, gold_segs, span_marginals,
                        pred_poses, gold_poses, pos_marginals) if pred_poses \
            else zip(pred_segs, gold_segs, span_marginals)
        for temp in iteration:
            if not pred_poses:
                pred_seg, gold_seg, span_m = temp
            else:
                pred_seg, gold_seg, span_m, pred_pos, gold_pos, pos_m = temp
                tp_pos = list((Counter(pred_pos) & Counter(gold_pos)).elements())
                for i, j, pos in tp_pos:
                    if pos_m[i][j][pos] > 0.0001:
                        pos_res.append([pos_m[i][j][pos].item(), True])
                not_right = list((Counter(pred_pos) - Counter(gold_pos)).elements())
                for i, j, pos in not_right:
                    if pos_m[i][j][pos] > 0.0001:
                        pos_res.append([pos_m[i][j][pos].item(), False])

            tp_span = list((Counter(pred_seg) & Counter(gold_seg)).elements())
            not_right = list((Counter(pred_seg) - Counter(gold_seg)).elements())
            for i, j in tp_span:
                if span_m[i][j] > 0.0001:
                    span_res.append([span_m[i][j].item(), True])
            for i, j in not_right:
                if span_m[i][j] > 0.0001:
                    span_res.append([span_m[i][j].item(), False])

        return span_res, pos_res

    @staticmethod
    def AdaptiveBinning(self, infer_results, show_reliability_diagram=True, name='span'):
        """
        This function implement adaptive binning. It returns AECE, AMCE and some other useful values.

        Arguements:
        infer_results (list of list): a list where each element "res" is a two-element list denoting the infer result of a single sample. res[0] is the confidence score r and res[1] is the correctness score c. Since c is either 1 or 0, here res[1] is True if the prediction is correctd and False otherwise.
        show_reliability_diagram (boolean): a boolean value to denote wheather to plot a Reliability Diagram.
        name (str): name of the figure.

        Return Values:
        AECE (float): expected calibration error based on adaptive binning.
        AMCE (float): maximum calibration error based on adaptive binning.
        cofidence (list): average confidence in each bin.
        accuracy (list): average accuracy in each bin.
        cof_min (list): minimum of confidence in each bin.
        cof_max (list): maximum of confidence in each bin.

        """
        import matplotlib.pyplot as plt
        # Initialize.
        infer_results.sort(key=lambda x: x[0], reverse=True)
        n_total_sample = len(infer_results)

        assert infer_results[0][0] <= 1 and infer_results[1][0] >= 0, 'Confidence score should be in [0,1]'

        z = 1.645
        num = [0 for i in range(n_total_sample)]
        final_num = [0 for i in range(n_total_sample)]
        correct = [0 for i in range(n_total_sample)]
        confidence = [0 for i in range(n_total_sample)]
        cof_min = [1 for i in range(n_total_sample)]
        cof_max = [0 for i in range(n_total_sample)]
        accuracy = [0 for i in range(n_total_sample)]

        ind = 0
        target_number_samples = float('inf')
        min_bin, rate = 20, 0.5

        # Traverse all samples for an initial binning.
        for i, confidence_correctness in enumerate(infer_results):
            confidence_score = confidence_correctness[0]
            correctness = confidence_correctness[1]
            # Merge the last bin if too small.
            if num[ind] > target_number_samples:
                if (n_total_sample - i) > min_bin and cof_min[ind] - infer_results[-1][0] > 0.05:
                    ind += 1
                    target_number_samples = float('inf')
            num[ind] += 1
            confidence[ind] += confidence_score

            assert correctness in [True, False], 'Expect boolean value for correctness!'
            if correctness:
                correct[ind] += 1

            cof_min[ind] = min(cof_min[ind], confidence_score)
            cof_max[ind] = max(cof_max[ind], confidence_score)
            # Get target number of samples in the bin.
            if cof_max[ind] == cof_min[ind]:
                target_number_samples = float('inf')
            else:
                target_number_samples = (z / (cof_max[ind] - cof_min[ind])) ** 2 * rate

        n_bins = ind + 1

        # Get final binning.
        if target_number_samples - num[ind] > 0:
            needed = target_number_samples - num[ind]
            extract = [0 for i in range(n_bins - 1)]
            final_num[n_bins - 1] = num[n_bins - 1]
            for i in range(n_bins - 1):
                extract[i] = int(needed * num[ind] / n_total_sample)
                final_num[i] = num[i] - extract[i]
                final_num[n_bins - 1] += extract[i]
            print('神奇操作')
        else:
            final_num = num
        final_num = final_num[:n_bins]

        # Re-intialize.
        num = [0 for i in range(n_bins)]
        correct = [0 for i in range(n_bins)]
        confidence = [0 for i in range(n_bins)]
        cof_min = [1 for i in range(n_bins)]
        cof_max = [0 for i in range(n_bins)]
        accuracy = [0 for i in range(n_bins)]
        gap = [0 for i in range(n_bins)]
        neg_gap = [0 for i in range(n_bins)]
        # Bar location and width.
        x_location = [0 for i in range(n_bins)]
        width = [0 for i in range(n_bins)]
        percetnt = [0 for i in range(n_bins)]

        # Calculate confidence and accuracy in each bin.
        ind = 0
        for i, confidence_correctness in enumerate(infer_results):

            confidence_score = confidence_correctness[0]
            correctness = confidence_correctness[1]
            num[ind] += 1
            confidence[ind] += confidence_score

            if correctness:
                correct[ind] += 1
            cof_min[ind] = min(cof_min[ind], confidence_score)
            cof_max[ind] = max(cof_max[ind], confidence_score)

            if num[ind] == final_num[ind]:
                confidence[ind] = confidence[ind] / num[ind] if num[ind] > 0 else 0
                accuracy[ind] = correct[ind] / num[ind] if num[ind] > 0 else 0
                left = cof_min[ind]
                right = cof_max[ind]
                x_location[ind] = (left + right) / 2
                width[ind] = (right - left) * 0.9

                percetnt[ind] = num[ind] / n_total_sample

                if confidence[ind] - accuracy[ind] > 0:
                    gap[ind] = confidence[ind] - accuracy[ind]
                else:
                    neg_gap[ind] = confidence[ind] - accuracy[ind]
                ind += 1
        print('acc=', sum(correct) / n_total_sample)
        # Get AECE and AMCE based on the binning.
        AMCE = 0
        AECE = 0
        for i in range(n_bins):
            AECE += abs((accuracy[i] - confidence[i])) * final_num[i] / n_total_sample
            AMCE = max(AMCE, abs((accuracy[i] - confidence[i])))

        # Plot the Reliability Diagram if needed.
        if show_reliability_diagram:
            f1, ax = plt.subplots()
            plt.bar(x_location, accuracy, width)
            plt.bar(x_location, gap, width, bottom=accuracy)
            plt.bar(x_location, neg_gap, width, bottom=accuracy)
            print(correct)
            print(num)
            # print(accuracy)
            # print(gap)
            # print(neg_gap)
            # plt.plot(x_location, percetnt, marker="s", mfc="white",
            #          ms=6, color='#FF00FF', linestyle='--')
            # plt.legend(['Percent', 'Accuracy', 'Positive gap', 'Negative gap'], fontsize=18, loc=2)
            plt.legend(['Accuracy', 'Positive gap', 'Negative gap'], fontsize=14, loc=2)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel('Marginal Probability', fontsize=15)
            plt.ylabel('Accuracy', fontsize=15)
            plt.title(f'{self.args.fdata.split(r"/")[2]} dev set ({name})', fontsize=15)
            if self.args.joint:
                fig_name = f'../result/joint/joint_{name}_{self.args.fdata.split(r"/")[2]}_{min_bin}_{rate}_dev_n.png'
            else:
                fig_name = f'../result/{name}_{self.args.fdata.split(r"/")[2]}_{min_bin}_{rate}_dev_n.png'
            plt.savefig(fig_name)
            plt.show()
            print(f'figure save in {fig_name}')
            # print(final_num)

        return AECE, AMCE, cof_min, cof_max, confidence, accuracy

    @staticmethod
    def adjust_segs_matrix(self, chars, segs):
        batch_size, n = chars.shape
        _, x, _ = segs.shape
        new_segs = torch.zeros((batch_size, n - 1, n - 1), dtype=torch.bool)
        new_segs[:, :x, :x] = segs
        return new_segs

    # @ classmethod
    def add_restraint(self, span_marginals, span):
        """
        根据解码得到的span对边缘概率矩阵添加约束
        eg:
            span(i,j)  对于i<k<j  令(a,k) (k,b) = -torch.inf
        """
        for batch in range(len(span_marginals)):
            for item in span[batch][0]:
                i = item[0]
                j = item[1]
                for k in range(i + 1, j):
                    if i > 0:
                        for a in range(i):
                            span_marginals[batch, a, k] = -torch.inf
                            # span_marginals[batch,a,k] = -torch.inf
                    if j < len(span_marginals[0]):
                        for b in range(j + 1, len(span_marginals[0])):
                            span_marginals[batch, k, b] = -torch.inf
                            # span_marginals[batch,k,b] = -torch.inf
        return span_marginals

    @staticmethod
    def pruning(pred_ctb, pred_msr, pred_ppd, pred_cky):
        # 不能是简单的删除，应该
        parent_dic = CMD.find_parent(pred_cky)
        res = []
        all_pred = list(set(pred_ctb + pred_msr + pred_ppd))
        for item in pred_cky:
            if item in all_pred:
                res.append(item)

        candidates = [item for item in pred_ctb if item not in res] # 候选词在词典中找不到父节点
        for candidate in candidates:
            candidate_parents = [item for item in pred_cky
                                 if item[0] <= candidate[0] and item[1] >= candidate[1]]
            candidate_parent = min(candidate_parents, key=lambda x:x[1]-x[0])  # 找到了候选词的跟节点
            wait_del = [item for item in pred_ctb
                        if item[0] >= candidate_parent[0] and item[1] <= candidate_parent[1]]
            for item in wait_del:
                if item in res:
                    res.remove(item)
        candidates = [item for item in pred_msr if item not in res]  # 候选词在词典中找不到父节点
        for candidate in candidates:
            candidate_parents = [item for item in pred_cky
                                 if item[0] <= candidate[0] and item[1] >= candidate[1]]
            candidate_parent = min(candidate_parents, key=lambda x: x[1] - x[0])  # 找到了候选词的跟节点
            wait_del = [item for item in pred_msr
                        if item[0] >= candidate_parent[0] and item[1] <= candidate_parent[1]]
            for item in wait_del:
                if item in res:
                    res.remove(item)
        candidates = [item for item in pred_ppd if item not in res]  # 候选词在词典中找不到父节点
        for candidate in candidates:
            candidate_parents = [item for item in pred_cky
                                 if item[0] <= candidate[0] and item[1] >= candidate[1]]
            candidate_parent = min(candidate_parents, key=lambda x: x[1] - x[0])  # 找到了候选词的跟节点
            wait_del = [item for item in pred_ppd
                        if item[0] >= candidate_parent[0] and item[1] <= candidate_parent[1]]
            for item in wait_del:
                if item in res:
                    res.remove(item)
        # for item in candidate:
        #     item_parent = parent_dic[f"{item}"]
        #     for item_ctb in pred_ctb:
        #         if parent_dic[item_ctb] == item_parent and item_ctb in res:
        #             res.delete(item_ctb)
        return res

    @staticmethod
    def find_parent(spans):
        dic = dict()
        for span in spans:
            candidates = [item for item in spans if
                          (item[0] == span[0] and item[1] > span[1]) or (item[1] == span[1] and item[0] < span[0])]
            if candidates:
                parent = min(candidates, key=lambda n: n[1] - n[0])
            else:
                parent = None

            dic[f"{span}"] = parent
        return dic


def update_span_score_with_segs(segs, span_score):
    batch_index, row_index, col_index = torch.where(segs)
    for i in range(len(batch_index)):
        batch, start, end = batch_index[i], row_index[i], col_index[i]
        for k in range(start + 1, end):
            if start > 0:
                span_score[batch, :start, k] = -torch.inf
            if end < len(segs[batch]):
                span_score[batch, k, end + 1:] = -torch.inf
    return span_score
