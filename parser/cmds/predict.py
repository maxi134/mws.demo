# -*- coding: utf-8 -*-

from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np

from parser import Model
from parser.cmds.cmd import CMD
from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify


# from parser.utils.find_conflict import ConflictFinder

class Predict(CMD):

    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Use a trained model to make predictions.'
        )
        subparser.add_argument('--fdata', default='../data/ctb5/ctb5.test.ws',
                               help='path to dataset')
        subparser.add_argument('--fpred', default='../predict/mws/test',
                               help='path to predicted result')

        return subparser

    def __call__(self, args):
        super(Predict, self).__call__(args)

        print("Load the dataset")

        corpus = Corpus.load(args.fdata, self.fields, args.joint)
        dataset = TextDataset(corpus,
                              # self.fields[:-1],
                              # self.fields[:-2],    # joint
                              self.fields,
                              args.buckets)
        # set the data loader
        dataset.loader = batchify(dataset, args.batch_size)

        data = corpus.read_sentence(args.fdata)  # get sentences

        print(f"{len(dataset)} sentences, "
              f"{len(dataset.loader)} batches"
              f"{len(dataset.buckets)} buckets")

        print("Load the model")
        self.model = Model.load(args.model)
        print(f"{self.model}\n")

        print("Make predictions on the dataset")
        start = datetime.now()

        # pred_labels = self.predict_cws(dataset.loader)
        pred_labels = self.predict(dataset.loader)

        indices = torch.tensor([i
                                for bucket in dataset.buckets.values()
                                for i in bucket]).argsort()

        labels = [pred_labels[i] for i in indices]
        corpus.labels = [pred_labels[i] for i in indices]

        print(f"Save the predicted result to {args.fpred}")
        with open(args.fpred, 'w') as f:
            for i in range(len(data)):
                f.writelines(data[i])
                f.writelines('\n')
                f.writelines(str(labels[i]) + '\n')

        total_time = datetime.now() - start
        print(f"{total_time}s elapsed, "
              f"{len(dataset) / total_time.total_seconds():.2f} Sentences/s")
