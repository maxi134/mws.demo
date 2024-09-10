# -*- coding: utf-8 -*-
# @ModuleName: analyze
# @Function:
# @Author: Wxb
# @Time: 2023/4/27 10:20

from parser.utils.corpus import Corpus
from parser.utils.data import TextDataset, batchify
from parser.cmds.cmd import CMD
from parser import Model
from datetime import datetime


class Analyze(CMD):
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(
            name, help='Train a model.'
        )
        subparser.add_argument('--ftrain', default='../data/ctb5/ctb5.train.ws',
                               help='path to test file')
        subparser.add_argument('--fdev', default='../data/ctb5/ctb5.dev.ws',
                               help='path to dev file')
        subparser.add_argument('--ftest', default='../data/ctb5/ctb5.test.ws',
                               help='path to test file')
        subparser.add_argument('--embed', action='store_true',
                               help='whether to use pretrained embeddings')
        subparser.add_argument('--unk', default=None,
                               help='unk token in pretrained embeddings')
        subparser.add_argument('--dict-file', default=None,
                               help='path for dictionary')
        return subparser

    def __call__(self, args):
        super(Analyze, self).__call__(args)

        train_sentences_count, train_words_count, train_chars_count = self.read_data(args.ftrain)
        dev_sentences_count, dev_words_count, dev_chars_count = self.read_data(args.fdev)
        test_sentences_count, test_words_count, test_chars_count = self.read_data(args.ftest)

        print(f"在train_data中:有{train_sentences_count}个句子，{train_words_count}个词，{train_chars_count}个字符")
        print(f"在dev_data中:有{dev_sentences_count}个句子，{dev_words_count}个词，{dev_chars_count}个字符")
        print(f"在test_data中:有{test_sentences_count}个句子，{test_words_count}个词，{test_chars_count}个字符")

    def read_data(self,filename):
        sentences_count, words_count, chars_count = 0, 0, 0
        with open(filename,'r') as f:
            data = f.readlines()
            for line in data:
                tmp = line.split()
                if len(tmp) > 0:
                    chars_count += 1
                    if tmp[1] == 's' or tmp[1] == 'b':
                        words_count += 1
                else:
                    sentences_count += 1
        return sentences_count, words_count, chars_count

