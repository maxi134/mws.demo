# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timedelta
from parser.model import Model  # 导入模型的位置
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.utils.metric import Metric
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import time
import matplotlib.pylab as plt
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup


class Train(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--ctb_train', default='data/ctb9/ctb9.train.ws',
                               help='path to train file')
        subparser.add_argument('--ppd_train', default='data/ppd/ppd.train.ws',
                               help='path to train file')
        subparser.add_argument('--msr_train', default='data/msr/msr.train.ws',
                               help='path to train file')
        subparser.add_argument('--baike_train', default='data/ctb9/ctb9.train.ws',
                               help='path to train file')
        subparser.add_argument('--dev',default='data/mws/mws.dev.ws')
        subparser.add_argument('--embed', action='store_true',
                               help='whether to use pretrained embeddings')
        subparser.add_argument('--unk', default=None,
                               help='unk token in pretrained embeddings')
        subparser.add_argument('--dict-file', default=None,
                               help='path for dictionary')
        subparser.add_argument('--loss_weight', default=0.05, type=float,
                               help='weight for loss weak data')
        return subparser
    
    def __call__(self, args):
        start_time = time.time()
        super(Train, self).__call__(args) # 修改embed预训练的数据集

        print("在CTB9数据集中:")
        ctb_train = self.preprocess(args, args.ctb_train)
        print("在MSR数据集中:")
        msr_train = self.preprocess(args, args.msr_train)
        print("在PPD数据集中:")
        ppd_train = self.preprocess(args, args.ppd_train)
        # TODO
        print("在百科数据集中:")
        baike_train = self.preprocess_baike(args, args.baike_train)
        print("在dev数据集中:")
        dev = self.preprocess(args,args.dev)
        print(f"load data spend time {time.time()-start_time} seconds")

        print("Create the model")
        embed = {'embed': self.CHAR.embed}
        if hasattr(self, 'BIGRAM'):
            embed.update({
                'bi_embed': self.BIGRAM.embed,
            })
        if hasattr(self, 'TRIGRAM'):
            embed.update({
                'tri_embed': self.TRIGRAM.embed,
            })
        self.model = Model(args).load_pretrained(embed) 
        print(f"{self.model}\n")

        self.model = self.model.to(args.device)
        # 微调Bert修改优化器
        if args.feat == 'bigram':
            self.optimizer = Adam(self.model.parameters(),
                                    args.lr,
                                    (args.mu, args.nu),
                                    args.epsilon)
            decay_steps = args.decay_epochs * (len(ctb_train)+len(msr_train)+len(ppd_train)+len(baike_train))
            self.scheduler = ExponentialLR(self.optimizer,
                                            args.decay**(1/decay_steps))
        elif args.feat == 'bert':
            decay_steps = args.decay_epochs * (len(ctb_train)+len(msr_train)+len(ppd_train)+len(baike_train))
            self.optimizer = AdamW(
                [{'params': c.parameters(), 'lr': args.lr * (1 if n == 'bert' else args.lr_rate)}
                 for n, c in self.model.named_children()],
                args.lr)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(decay_steps*args.warmup), decay_steps)
        
        total_time = timedelta()
        best_e = 1
        best_metric_span = Metric()

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            # TODO
            self.train(ctb_train, msr_train, ppd_train, baike_train)
            print(f"Epoch {epoch} / {args.epochs}:")
            loss, metric_span, _, _ = self.evaluate(dev, dataset_name=None)
            print(f"span--{'dev:':6} Loss: {loss:.4f} {metric_span}")

            t = datetime.now() - start
            if metric_span > best_metric_span:
                best_metric_span = metric_span
                best_e = epoch
                if hasattr(self.model, 'module'):
                    self.model.module.save(args.model)
                else:
                    self.model.save(args.model)
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")
            total_time += t
            if epoch - best_e >= args.patience:
                break

        self.model = Model.load(args.model)

        loss,metric_span,pos,all_segs = self.evaluate(dev,dataset_name=None)

        # ctb_loss, ctb_metric_span, _, _, _ = self.evaluate_train(ctb_dev,dataset_name='ctb')
        # msr_loss, msr_metric_span, _, _, _ = self.evaluate_train(msr_dev,dataset_name='msr')
        # ppd_loss, ppd_metric_span, _, _, _= self.evaluate_train(ppd_dev,dataset_name='ppd')

        print("the result of the model:")
        print(f"the span score of dev at epoch {best_e} is {metric_span.score:.2%}")

        print(f"average time of each epoch is {total_time / epoch}s")
        print(f"{total_time}s elapsed")

    def preprocess(self,args,data):
        if data == args.dev:
            train = Corpus.load(data, self.fields, args.joint)
            train = TextDataset(train, self.fields, args.buckets)
            train.loader = batchify(train, 500, True)
        else:
            train = Corpus.load(data, self.fields, args.joint)
            train = TextDataset(train, self.fields, args.buckets)
            train.loader = batchify(train, args.batch_size, True)
        print(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print("=" * 100)
        return train.loader
    
    def preprocess_baike(self,args,data):
        train = Corpus.load_baike(data, self.fields, args.joint)
        train = TextDataset(train, self.fields, args.buckets)
        train.loader = batchify(train, args.batch_size, True)
        print(f"{'train:':6} {len(train):5} sentences, "
              f"{len(train.loader):3} batches, "
              f"{len(train.buckets)} buckets")
        print("=" * 100)
        return train.loader