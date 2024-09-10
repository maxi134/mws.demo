# -*- coding: utf-8 -*-

import argparse
import os
from parser.cmds import Evaluate, Predict, Train, Analyze
from parser.config import Config
import torch
from parser.model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create the Biaffine Parser model.'
    )
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subcommands = {
        'evaluate': Evaluate(),
        'predict': Predict(),
        'train': Train(),
        'analyze': Analyze()
    }
    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)
        subparser.add_argument('--conf', '-c', default='config.ini',
                               help='path to config file')
        subparser.add_argument('--file', '-f', default='exp/ctb9.char',
                               help='path to saved files')
        subparser.add_argument('--preprocess', '-p', action='store_true',
                               help='whether to preprocess the data first')
        subparser.add_argument('--device', '-d', default='1',
                               help='ID of GPU to use')
        subparser.add_argument('--seed', '-s', default=5, type=int,
                               help='seed for generating random numbers')
        subparser.add_argument('--threads', '-t', default=10, type=int,
                               help='max num of threads')
        subparser.add_argument('--feat', default='bert',
                               choices=[None, 'bert', 'bigram', 'trigram'],
                               help='choices of additional features')
        # 在这里修改batch_size试试
        subparser.add_argument('--batch-size', default=5000, type=int,
                               help='batch size')
        subparser.add_argument('--buckets', default=1, type=int,
                               help='max num of buckets to use')
        subparser.add_argument('--m_train', default=10, type=int,
                               help='max length of words in train set')
        subparser.add_argument('--m_test', default=10, type=int,
                               help='max length of words in test set')
        subparser.add_argument('--marg', action='store_true',
                               help='whether to use marginal probs')
        subparser.add_argument('--joint', '-j', action='store_true',
                               help='whether to joint the part-of-speech modeling')
        subparser.add_argument('--link', default=None,
                               choices=[None, 'mlp', 'att'],  
                               help='choices of link method')
        subparser.add_argument('--bert_model', default="/data3/maxi/bert-base-chinese",
                               help='the path of model')
        subparser.add_argument('--n_embed', default=100,
                               help='embedding')
        subparser.add_argument('--embed_path', default="../data/embed.ws",
                               help='the path of embedding file')

                    
    args = parser.parse_args()
    # breakpoint()
    print(f"Set the max num of threads to {args.threads}")
    print(f"Set the seed for generating random numbers to {args.seed}")
    print(f"Set the device with ID {args.device} visible")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.fields = os.path.join(args.file, 'fields')
    args.model = os.path.join(args.file, 'model')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = Config(args.conf).update(vars(args))

    print(f"Run the subcommand in mode {args.mode}")
    cmd = subcommands[args.mode]
    cmd(args)
