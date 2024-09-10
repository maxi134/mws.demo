
class SpanMerger:
    def __init__(self, pred_segs_ctb, pred_segs_msr, pred_segs_ppd):
        """
        根据三个解码层，以及CKY的解码结果，获取最终CKY的解码结果
        args:
            pred_segs_ctb:
            pred_segs_msr:
            pred_segs_ppd:
            pred_segs:根据添加了约束的边缘概率矩阵进行解码得到的结果，其中包含较多的无用结点
            span_merge:由find_conflict得到
                        对三个解码层的结果进行合并，len(span_meger)=1
                        span_merge[0]用于对边缘概率矩阵添加约束
                        span_merge[1]包含三个元素，分别表示：相同的预测，多粒度预测，冲突预测
        return：
            CKY解码的最终结果，多粒度分词的结果
        """
        self.pred_segs_ctb = pred_segs_ctb
        self.pred_segs_msr = pred_segs_msr
        self.pred_segs_ppd = pred_segs_ppd
        self.pred_segs_all = sorted(list(set(pred_segs_ctb + pred_segs_msr + pred_segs_ppd)), key=lambda x: x[0])

    def merge_span(self, span_merge, pred_segs):
        """
        span_merge:由find_conflict得到，长度为2
        """
        res = []
        span_res = span_merge[1] # 用于边缘概率矩阵，span拆分为三类

        # Merge spans from span_res
        for seg in span_res[0]:
            res.append(seg)
        for seg in span_res[1]:
            res.append(seg)

        # Find spans from pred_segs that match conditions in CKY spans
        for item in span_res[2]:
            res.append(item)
        # res.append(seg for seg in span_res[2])


        # 找到三种解码方式对于冲突的部分是怎么划分的

        conflict_ctb, conflict_msr, conflict_ppd, conflict_cky = [], [], [], []
        for item in span_res[2]:
            tmp = [seg for seg in self.pred_segs_ctb if self.is_within_bounds(seg,item)]
            conflict_ctb.append(tmp)
        for item in span_res[2]:
            tmp = [seg for seg in self.pred_segs_msr if self.is_within_bounds(seg,item)]
            conflict_msr.append(tmp)
        for item in span_res[2]:
            tmp = [seg for seg in self.pred_segs_ppd if self.is_within_bounds(seg,item)]
            conflict_ppd.append(tmp)
        for item in span_res[2]:
            tmp = [seg for seg in pred_segs if self.is_within_bounds(seg,item)]
            conflict_cky.append(tmp)
        for i in range(len(span_res[2])):
            item = span_res[2][i]
            try:
                tmp_ctb = [seg for seg in conflict_ctb[i] if self.is_list1_subset_of_list2(conflict_ctb[i],conflict_cky[i])]
            except IndexError as e:
                print(self.pred_segs_ctb)
                print(self.pred_segs_msr)
                print(self.pred_segs_ppd)
                print(pred_segs)
                print(span_merge)
            tmp_msr = [seg for seg in conflict_msr[i] if self.is_list1_subset_of_list2(conflict_msr[i],conflict_cky[i])]
            tmp_ppd = [seg for seg in conflict_ppd[i] if self.is_list1_subset_of_list2(conflict_ppd[i],conflict_cky[i])]
            tmp = tmp_ctb + tmp_msr + tmp_ppd
            if item not in tmp:
                tmp.append(item)
            tmp = list(set(tmp))
            res.extend(tmp)
        # tmp = [seg for seg in segs if seg in self.pred_segs_all]
        # res.extend(tmp)
        res = list(set(res))
        # Sort the result based on the starting index
        res = sorted(res, key=lambda x: x[0])

        return res

    @staticmethod
    def is_within_bounds(seg, item):
        i, j = item[0], item[1]
        return i <= seg[0] <= j and i <= seg[1] <= j
    
    @staticmethod
    def is_list1_subset_of_list2(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        return set1.issubset(set2)

# # 示例用法
# pred_segs_ctb = [(0, 2),(2, 4),(4, 6),(6, 9),(9,10)]
# pred_segs_msr = [(0, 5),(5, 6),(6, 8),(8,10)]
# pred_segs_ppd = [(0, 2),(2, 4),(4, 6),(6, 7),(7, 9),(9,10)]
# span_merge = [[],[[(0, 6), (6, 9), (9, 10)],[(6, 9),(6, 7),(7, 9)],[(0, 6)]]]
# pred_segs = [(0,10),(0, 6),(6,10),(0, 2),(2, 6),(6, 9),(9,10),(0, 1),(1, 2),(2, 4),(4, 6),(2, 3),(3, 4),(4, 5),(5, 6),(6, 7),(7, 9),(7, 8),(8, 9)]

# pred_segs_ctb = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 9), (9, 11), (11, 13), (13, 15), (15, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 30), (30, 32), (32, 33), (33, 35), (35, 36)]
# pred_segs_msr = [(0, 5), (5, 6), (6, 8), (8, 9), (9, 14), (14, 15), (15, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 30), (30, 32), (32, 33), (33, 35), (35, 36)]
# pred_segs_ppd = [(0, 2), (2, 6), (6, 8), (8, 9), (9, 11), (11, 15), (15, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 30), (30, 32), (32, 33), (33, 35), (35, 36)]
# pred_segs = [(0, 1), (0, 2), (0, 6), (0, 8), (0, 30), (0, 32), (0, 33), (0, 35), (0, 36), (1, 2), (2, 3), (2, 4), (2, 6), (3, 4), (4, 5), (4, 6), (5, 6), (6, 7), (6, 8), (7, 8), (8, 9), (8, 29), (8, 30), (9, 10), (9, 11), (9, 15), (9, 29), (10, 11), (11, 12), (11, 13), (11, 15), (12, 13), (13, 14), (13, 15), (14, 15), (15, 16), (15, 20), (15, 21), (15, 23), (15, 25), (15, 27), (15, 29), (16, 17), (16, 18), (16, 20), (17, 18), (18, 19), (18, 20), (19, 20), (20, 21), (21, 22), (21, 23), (22, 23), (23, 24), (23, 25), (24, 25), (25, 26), (25, 27), (26, 27), (27, 28), (27, 29), (28, 29), (29, 30), (30, 31), (30, 32), (31, 32), (32, 33), (33, 34), (33, 35), (34, 35), (35, 36)]
# span_merge = [[(0, 6), (6, 8), (8, 9), (9, 15), (15, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 30), (30, 32), (32, 33), (33, 35), (35, 36)], [[(6, 8), (8, 9), (15, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 30), (30, 32), (32, 33), (33, 35), (35, 36)], [], [(0, 6), (9, 15)]]]


# span_merger = SpanMerger(pred_segs_ctb, pred_segs_msr, pred_segs_ppd)
# result = span_merger.merge_span(span_merge, pred_segs)
# print(result)
