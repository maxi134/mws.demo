# -*- coding: utf-8 -*-

from datetime import datetime
from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
import torch
from parser.utils.metric import SegF1Metric

class Evaluate(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--fdata', default='../data/ctb5/ctb5.test.ws',
                               help='path to dataset')
        subparser.add_argument('--batch_size', default=500, type=int,
                               help='batch_size')

        return subparser

    def __call__(self, args):
        super(Evaluate, self).__call__(args)

        print("Load the dataset")
        # TODO:如果选择多粒度分词的数据集时要修改标签转换的函数
        corpus = Corpus.load(args.fdata, self.fields, args.joint)
        dataset = TextDataset(corpus, self.fields, args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)
        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches, "
              f"{len(dataset.buckets)} buckets")

        print("Load the model")
        self.model = Model.load(args.model)
        print(f"{self.model}\n")

        print("Evaluate the dataset", args.fdata)
        start = datetime.now()
        # data = corpus.read_sentence(args.fdata)
        loss, metric_span, metric_pos, all_segs = self.evaluate(dataset.loader,dataset_name=None)
        total_time = datetime.now() - start

        print(f"Loss: {loss:.4f} {metric_span}")
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sentences/s")
