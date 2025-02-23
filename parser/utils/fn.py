# -*- coding: utf-8 -*-

import unicodedata

from nltk.tree import Tree
from parser.utils.common import pos_label


def ispunct(token):
    return all(unicodedata.category(char).startswith('P')
               for char in token)


def isfullwidth(token):
    return all(unicodedata.east_asian_width(char) in ['W', 'F', 'A']
               for char in token)


def islatin(token):
    return all('LATIN' in unicodedata.name(char)
               for char in token)


def isdigit(token):
    return all('DIGIT' in unicodedata.name(char)
               for char in token)


def tohalfwidth(token):
    # 确保所有字符串在底层有相同的表示
    return unicodedata.normalize('NFKC', token)


def stripe(x, n, w, offset=(0, 0), dim=1):
    r"""Returns a diagonal stripe of the tensor.

    Parameters:
        x (Tensor): the input tensor with 2 or more dims.
        n (int): the length of the stripe.
        w (int): the width of the stripe.
        offset (tuple): the offset of the first two dims.
        dim (int): 0 if returns a horizontal stripe; 1 else.

    Example::
    >>> x = torch.arange(25).view(5, 5)
    >>> x
    tensor([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]])
    >>> stripe(x, 2, 3, (1, 1))
    tensor([[ 6,  7,  8],
            [12, 13, 14]])
    >>> stripe(x, 2, 3, dim=0)
    tensor([[ 0,  5, 10],
            [ 6, 11, 16]])
    """
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[0, 0].numel()
    stride[0] = (seq_len + 1) * numel
    stride[1] = (1 if dim == 1 else seq_len) * numel
    return x.as_strided(size=(n, w, *x.shape[2:]),
                        stride=stride,
                        storage_offset=(offset[0]*seq_len+offset[1])*numel)


def isprojective(sequence):
    sequence = [0] + list(sequence)
    arcs = [(h, d) for d, h in enumerate(sequence[1:], 1) if h >= 0]
    for i, (hi, di) in enumerate(arcs):
        for hj, dj in arcs[i+1:]:
            (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
            if (li <= hj <= ri and hi == dj) or (lj <= hi <= rj and hj == di):
                return False
            if (li < lj < ri or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                return False
    return True


def get_spans(labels):
    spans = []
    for i, label in enumerate(labels):
        if label in ('b', 's', 'B', 'S'):
            spans.append((i, i+1))
        else:
            spans[-1] = (spans[-1][0], i+1)
    return set(spans)


def pad(tensors, padding_value=0):
    size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                             for i in range(len(tensors[0].size()))]
    out_tensor = tensors[0].data.new(*size).fill_(padding_value)
    for i, tensor in enumerate(tensors):
        out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
    return out_tensor


def binarize(tree):
    tree = tree.copy(True)
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Tree):
            nodes.extend([child for child in node])
            if len(node) > 1:
                for i, child in enumerate(node):
                    if not isinstance(child[0], Tree):
                        node[i] = Tree(f"{node.label()}|<>", [child])
    tree.chomsky_normal_form('left', 0, 0)
    tree.collapse_unary()

    return tree


def decompose(tree):
    tree = tree.copy(True)
    pos = set(list(zip(*tree.pos()))[1])
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Tree):
            nodes.extend([child for child in node])
            for i, child in enumerate(node):
                if isinstance(child, Tree) and len(child) == 1 and isinstance(child[0], str):
                    node[i] = Tree(child.label(), [Tree("CHAR", [char])
                                                   for char in child[0]])

    return tree, pos


def compose(tree):
    tree = tree.copy(True)
    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Tree):
            nodes.extend([child for child in node])
            for i, child in enumerate(node):
                if isinstance(child, Tree) and child.label() in pos_label:
                    node[i] = Tree(child.label(), ["".join(child.leaves())])

    return tree


def factorize(tree, delete_labels=None, equal_labels=None):
    def track(tree, i):
        label = tree.label()
        if delete_labels is not None and label in delete_labels:
            label = None
        if equal_labels is not None:
            label = equal_labels.get(label, label)
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return (i+1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = [(i, j, label)] + spans
        return j, spans
    return track(tree, 0)[1]

def all_length_one(lst):
    return all(len(item) == 1 for item in lst)

def tag2seg_mws(tags):
    """transform a mws tag sequence to a segmentation sequence.
    Args:
        tags (list): ('ss','sb','be','es')
    Returns:
        segs (list): [(0, 1), (1, 2), (2, 4), (0,1), (1,3), (3,4)]
    """
    output_list = []
    segs = []
    pos = None
    tags = list(tags)
    sentence_length = len(tags)  # 为了方便弱标注数据的使用，记录句子的长度
    if all_length_one(tags):
        for i, tag in enumerate(tags):
            if tag == 'b':
                start = i  # Update start to the current index when a 'b' tag is found.
            elif tag == 'e' and start is not None:
                # Once an 'e' tag is found and start is not None, append the segment (start, i+1).
                segs.append((start, i + 1))
                start = None  # Reset start to None for the next segment.
            elif tag == 's':
                # 's' tags represent a segment on their own.
                segs.append((i, i + 1))
    else:
    # 遍历输入的元组中的元素
        for i in range(5):
            lst = []
            for j in range(len(tags)):
                if tags[j]:
                    lst.append(tags[j][0])
                    tags[j] = tags[j][1:]
                else:
                    lst.append('')
            output_list.append(lst)
        for item in output_list:
            segs.extend(tag2seg2(item))
        segs = sorted(segs,key=lambda x:(x[0], x[1]))
    segs.append(sentence_length)
    return segs, pos

def tag2seg2(tags):
    """
    多粒度分词数据集使用
    """
    start = 0
    end = 0
    segs = []
    for tag in tags:
        flag = False
        end += 1
        if tag == 'b' or tag == 's':
            start = end -1
        if tag == 'e' or tag == 's':
            flag = True
            segs.append((start, end))
            start = end
    if start != end and flag == True:
        segs.append((start, end))
    return segs

# def tag2seg(tags):
#     """transform a tag sequence to a segmentation sequence.
#     Args:
#         tags (list): ['s', 's', 'b', 'e', 's', 'b', 'e', 'b', 'e', 'b', 'e']

#     Returns:
#         segs (list): [(0, 1), (1, 2), (2, 4), (4, 5), (5, 7), (7, 9), (9, 11)]
#     """
#     start = 0
#     end = 0
#     segs = []
#     pos = None
#     for tag in tags:
#         end += 1
#         if tag == 'e' or tag == 's':
#             segs.append((start, end))
#             start = end
#     if start != end:
#         segs.append((start, end))
#     return segs, pos


def tag2seg(tags):
    """Transform a tag sequence to a segmentation sequence, ignoring 'x'
    
    Args:
        tags (list): A list of tags including 'b', 'e', 'i', 's', and 'x', where 'x' represents uncertain parts.
        tag1 (list):['s', 's', 'b', 'e', 's', 'b', 'e', 'b', 'e', 'b', 'e']
        tag2 (list):['x', 'x', 's', 'x', 'b', 'i', 'e', 'x', 's']
        
    Returns:
        segs (list): A list of tuples indicating the segments determined by 'b' and 'e' tags, and individual segments for 's'.
        seg1 (list):[(0, 1), (1, 2), (2, 4), (4, 5), (5, 7), (7, 9), (9, 11)]
        seg2 (list):[(2, 3), (4, 7), (8, 9)]
    """
    segs = []
    start = None  # Initialize start as None to indicate that we haven't found a 'b' tag yet.
    pos = None
    for i, tag in enumerate(tags):
        if tag == 'b':
            start = i  # Update start to the current index when a 'b' tag is found.
        elif tag == 'e' and start is not None:
            # Once an 'e' tag is found and start is not None, append the segment (start, i+1).
            segs.append((start, i + 1))
            start = None  # Reset start to None for the next segment.
        elif tag == 's':
            # 's' tags represent a segment on their own.
            segs.append((i, i + 1))
    return segs, pos

def tag2seg_pos(tags):
    """transform a tag sequence to a segmentation sequence.

    Args:
        tags (list): ['sPOS', 'sPOS', 'b', 'ePOS', 's', 'b', 'ePOS', 'b', 'ePOS', 'b', 'ePOS']

    Returns:
        segs (list): [(0, 1), (1, 2), (2, 4), (4, 5), (5, 7), (7, 9), (9, 11)]
    """
    start = 0
    end = 0
    segs, pos = [], []

    for tag in tags:
        end += 1
        if tag[0] == 'e' or tag[0] == 's':
            # pos.append(tag[1:])    # pos
            pos.append((start, end, tag[1:]))
            segs.append((start, end))
            start = end

    if start != end:
        segs.append((start, end))

    return segs, pos


def seg2tag(segs):
    """transform a segmentation sequence to a tag sequence.

    Args:
        segs (list): [(0, 1), (1, 2), (2, 4), (4, 5), (5, 7), (7, 9), (9, 11)]

    Returns:
        tags (list): ['s', 's', 'b', 'e', 's', 'b', 'e', 'b', 'e', 'b', 'e']
    """

    # TODO

    pass


def tensor2scalar(indexes):
    """[summary]

    Args:
        indexes ([type]): tuple(Tensor)
    """

    return [t.tolist() for t in indexes]


def build(tree, sequence):
    label = tree.label()
    leaves = [subtree for subtree in tree.subtrees()
              if not isinstance(subtree[0], Tree)]

    def recover(label, children):
        if label.endswith('|<>'):
            if label[:-3] in pos_label:
                label = label[:-3]
                tree = Tree(label, children)
                return [Tree(label, [Tree("CHAR", [char])
                                     for char in tree.leaves()])]
            else:
                return children
        else:
            sublabels = [l for l in label.split('+')]
            sublabel = sublabels[-1]
            tree = Tree(sublabel, children)
            if sublabel in pos_label:
                tree = Tree(sublabel, [Tree("CHAR", [char])
                                       for char in tree.leaves()])
            for sublabel in reversed(sublabels[:-1]):
                tree = Tree(sublabel, [tree])
            return [tree]

    def track(node):
        i, j, label = next(node)
        if j == i+1:
            return recover(label, [leaves[i]])
        else:
            return recover(label, track(node) + track(node))

    tree = Tree(label, track(iter(sequence)))

    return tree
