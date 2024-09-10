# -*- coding: utf-8 -*-

from datetime import datetime
from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
import torch

class Evaluate(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Evaluate the specified model and dataset.'
        )
        subparser.add_argument('--fdata', default='data/ctb51/test.conll',
                               help='path to dataset')

        return subparser

    def __call__(self, args):
        super(Evaluate, self).__call__(args)

        print("Load the dataset")
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
        loss, metric_span, metric_pos = self.evaluate(dataset.loader,dataset_name=None)
        total_time = datetime.now() - start
        # ==============================================================================
        # indices = torch.tensor([i
        #                         for bucket in dataset.buckets.values()
        #                         for i in bucket]).argsort()

        print(f"Loss: {loss:.4f} {metric_span}")
        # print(f"Loss: {loss:.4f} {metric_pos}")
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sentences/s")
