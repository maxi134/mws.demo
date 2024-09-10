# -*- coding: utf-8 -*-

from collections import namedtuple
from collections.abc import Iterable

from parser.utils.fn import tag2seg, tag2seg_pos, tag2seg_mws
from parser.utils.field import Field
import sys

CoNLL = namedtuple(typename='CoNLL',
                   field_names=['CHAR', 'SEG', 'POS'],
                   defaults=[None] * 3)




class Sentence(object):

    def __init__(self, fields, values):
        for field, value in zip(fields, values):
            if isinstance(field, Iterable):  # field.name 与 char or seg 对应
                for j in range(len(field)):
                    setattr(self, field[j].name, value)
            else:
                setattr(self, field.name, value)
        self.fields = fields

    @property
    def values(self):
        for field in self.fields:
            if isinstance(field, Iterable):
                yield getattr(self, field[0].name)
            else:
                yield getattr(self, field.name)

    def __len__(self):
        return len(next(iter(self.values)))

    def __repr__(self):
        if hasattr(self, "labels"):  # span-based预测评估时使用
            temp = list(self.values)
            # temp[-1] = self.labels   # 当时未联合建模的原版
            # return '\n'.join('\t'.join(map(str, line))
            #                  for line in zip(*temp)) + '\n'
            temp[1] = self.labels
            return '\n'.join('\t'.join(map(str, line))    # 新版本
                             for line in zip(*temp[:-1])) + '\n'
        else:
            return '\n'.join('\t'.join(map(str, line))
                             for line in zip(*self.values)) + '\n'


class Corpus(object):

    def __init__(self, fields, sentences):
        super(Corpus, self).__init__()

        self.fields = fields
        self.sentences = sentences  # all sentences

    def __len__(self):
        return len(self.sentences)

    def __repr__(self):
        return '\n'.join(str(sentence) for sentence in self)

    def __getitem__(self, index):
        return self.sentences[index]

    def __getattr__(self, name):
        if not hasattr(self.sentences[0], name):
            raise AttributeError
        for sentence in self.sentences:
            yield getattr(sentence, name)  # 生成器
        # res = []   # only for test
        # for sentence in self.sentences[:5]:
        #     res.append(getattr(sentence, name))
        # return res

    def __setattr__(self, name, value):
        if name in ['fields', 'sentences']:
            self.__dict__[name] = value
        else:
            for i, sentence in enumerate(self.sentences):
                setattr(sentence, name, value[i])
    @classmethod
    def read_sentence(cls,path):
        sentences = []
        sentence = []
        with open(path,'r') as f:
            lines = [line.strip() for line in f]
        sentence = ' '.join(line.split('\t')[0] for line in lines)
        sentences.append(sentence)
        # for item in lines:
        #     if item:
        #         sentence.append(item.split('\t')[0])
        #     else:
        #         tmp = ''.join(sentence)
        #         sentences.append(tmp)
        #         sentence = []
        return sentences

    @classmethod  # 无需实例化class可直接调用，cls = self
    def load(cls, path, fields, joint):
        start, sentences = 0, []
        fields = [field if field is not None else Field(str(i))
                  for i, field in enumerate(fields)]
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        # 选择多粒度分词的数据集时需要修改标签转换
        # fn = tag2seg_pos if joint else tag2seg
        fn = tag2seg_pos if joint else tag2seg_mws
        values = list(zip(*[l.split('\t') for l in lines]))
        values[-1], pos = fn(values[-1])
        values.append(pos)
        sentences.append(Sentence(fields, values))
        # 这个逻辑无法处理只有一个句子的情况
        # for i, line in enumerate(lines):
        #     if not line:
        #         # [chars: ["我", "爱", ...], tags: [B, M, E, ...]], segment: [(0,1), (1,3)...]
        #         values = list(zip(*[l.split('\t') for l in lines[start:i]]))  # char and tags  ('b', 'e', 'b', 'e', 'b', 'e', 's', 'b', 'e', 'b', 'e', 'b', 'e')
        #         try:
        #             values[-1], pos = fn(values[-1])  # tags to segment (and pos)  [(0, 2), (2, 4), (4, 6), (6, 7), (7, 9), (9, 11), (11, 13), 13]
        #         except UnboundLocalError as e:
        #             print(values)
        #         values.append(pos)
        #         sentences.append(Sentence(fields, values))  # field.name, char and segment
        #         start = i + 1
        return cls(fields, sentences)

    @classmethod
    def load_baike(cls, path, fields, joint):
        start, sentences = 0, []

        fields = [field if field is not None else Field(str(i))
                  for i, field in enumerate(fields)]
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        fn = tag2seg_pos if joint else tag2seg_mws
        for i, line in enumerate(lines):
            values = []
            if not line:
                sent = lines[start:i]
                values.append(tuple([l.split('\t')[0] for l in sent]))
                values.append([tuple([l.split('\t')[1][0] for l in sent]),
                               tuple([l.split('\t')[1][1] for l in sent]),
                               tuple([l.split('\t')[1][2] for l in sent])])
                values[1][0], pos = fn(values[1][0])
                values[1][1], pos = fn(values[1][1])
                values[1][2], pos = fn(values[1][2])
                values.append(pos)
                sentences.append(Sentence(fields, values))
                start = i+1
        return cls(fields, sentences)

    def save(self, path):
        with open(path, 'w') as f:
            f.write(f"{self}\n")
