class ConflictFinder:
    def __init__(self, list1, list2, list3):
        """ find conflict in three span

        给定对于同一个句子的三种解码方式，根据这三个解码结果，找到一个综合结果，最大的变化是将冲突发生的结点，将冲突的部分进行合并即可
    
        Args:
            list1 = [(0, 2), (2, 4), (4, 6), (6, 9), (9, 10), (10, 13), (13, 15), (15, 17), (17, 19)]
            list2 = [(0, 2), (2, 5), (5, 6), (6, 9), (9, 10), (10, 13), (13, 15), (15, 17), (17, 19)]
            list3 = [(0, 2), (2, 6), (6, 7), (7, 9), (9, 10), (10, 13), (13, 15), (15, 17), (17, 18), (18, 19)]
    
        Return:
            span = [(0, 2), (2, 6), (6, 7), (7, 9), (9, 10), (10, 13), (13, 15), (15, 17), (17, 19)], 用于对概率矩阵添加约束
            span_1 = [(0, 2), (6, 9), (9, 10), (10, 13), (13, 15), (15, 17), (17, 19)], # 表示公共的部分
            span_2 = [], # 表示多粒度的部分
            span_3 = [(2, 6)] # 表示存在冲突的首尾
      """
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3

    def find_conflicts(self):
        """
        有两个return,需要修改两次
        span:用于对边缘概率矩阵添加约束
        span_1:表示三个解码结果相同的部分，这部分直接作为最后的解码
        span_2:表示粒度不同的地方，都作为最后的结果
        span_3:表示产生了冲突,应该选择CKY解码得到的结果,将合并的部分视作一个span,span的分支根据CKY的解码结果做选择
        """
        res, res_diff, tmp = [], [], []
        tmp1, tmp2, tmp3 = [], [], []
        i, j, k = 0, 0, 0
        count_diff = 0

        # Remove repeated intervals in all three lists
        list1, list2, list3, span_1 = self.delete_repet(self.list1, self.list2, self.list3)
        if len(list1) == 0 and len(list2) == 0 and len(list3) == 0:
            span_2 = []
            span_3 = []
            span = span_1 + span_2 + span_3
            return span, span_1, span_2, span_3
        span_2, span_3 = [], []
        while i < len(list1) and j < len(list2) and k < len(list3):
            interval1, interval2, interval3 = list1[i], list2[j], list3[k]

            # Find overlapping intervals
            if interval1[0] == interval2[0] == interval3[0]:
                tmp1.append(interval1)
                tmp2.append(interval2)
                tmp3.append(interval3)

                while interval1[1] != interval2[1] or interval1[1] != interval3[1]:
                    if interval1[1] < interval2[1]:
                        i += 1
                        interval1 = list1[i]
                        tmp1.append(interval1)
                    elif interval1[1] < interval3[1]:
                        i += 1
                        interval1 = list1[i]
                        tmp1.append(interval1)
                    elif interval2[1] < interval1[1]:
                        j += 1
                        interval2 = list2[j]
                        tmp2.append(interval2)
                    elif interval2[1] < interval3[1]:
                        j += 1
                        interval2 = list2[j]
                        tmp2.append(interval2)
                    elif interval3[1] < interval1[1]:
                        k += 1
                        interval3 = list3[k]
                        tmp3.append(interval3)
                    elif interval3[1] < interval2[1]:
                        k += 1
                        interval3 = list3[k]
                        tmp3.append(interval3)

                if interval1[1] == interval2[1] == interval3[1]:
                    if len(tmp1) != 1 and len(tmp2) != 1 and tmp1 != tmp2:
                        tmp1, tmp2 = self.delete_repet_2(tmp1, tmp2)
                        if tmp1 != 1 and tmp2 != 1 and tmp1 != tmp2:
                            res.append([tmp1,tmp2,tmp3])
                    elif len(tmp2) != 1 and len(tmp3) != 1 and tmp2 != tmp3:
                        tmp2, tmp3 = self.delete_repet_2(tmp2,tmp3)
                        if tmp2 != 1 and tmp3!= 1 and tmp2 != tmp3:
                            res.append([tmp1,tmp2,tmp3])
                    elif len(tmp1) != 1 and len(tmp3) !=1 and tmp1 != tmp3:
                        tmp1, tmp3 = self.delete_repet_2(tmp1,tmp3)
                        if len(tmp1) != 1 and len(tmp3) != 1 and tmp1 != tmp3:
                            res.append([tmp1,tmp2,tmp3])
                    else:
                        span_2.append([tmp1, tmp2, tmp3])
                    tmp1, tmp2, tmp3 = [], [], []

                    # if (len(tmp1) != 1 and len(tmp2) != 1 and tmp1 != tmp2) or \
                    # (len(tmp2) != 1 and len(tmp3) != 1 and tmp2 != tmp3) or \
                    # (len(tmp1) != 1 and len(tmp3) != 1 and tmp1 != tmp3):
                    #     res.append([tmp1, tmp2, tmp3])
                    #     count_diff += 1
                    # else:
                    #     span_2.append([tmp1, tmp2, tmp3])
                    # tmp1, tmp2, tmp3 = [], [], []
            i += 1
            j += 1
            k += 1
        span_2 = list(set(item for sublist in span_2 for lst in sublist for item in lst))
        span_2 = sorted(span_2, key=lambda x: x[0])
        span_3 = res
        span_tmp = []
        for item in span_3:
            span_tmp.append((item[0][0][0], item[-1][-1][-1]))
        span_3 = span_tmp
        span = sorted(span_1+span_2+span_3, key=lambda x: x[0])
        return span, span_1, span_2, span_3
    
    def delete_repet_2(self,list1,list2):
        tmp1, tmp2 = [], []
        for item in list1:
            if item not in list2:
                tmp1.append(item)
        for item in list2:
            if item not in list1:
                tmp2.append(item)
        return tmp1, tmp2

    def delete_repet(self, list1, list2, list3):
        common = []
        tmp1, tmp2, tmp3 = [], [], []
        for item in list1:
            if item not in list2 or item not in list3:
                tmp1.append(item)
            else:
                if item not in common:
                    common.append(item)
        for item in list2:
            if item not in list1 or item not in list3:
                tmp2.append(item)
            else:
                if item not in common:
                    common.append(item)
        for item in list3:
            if item not in list1 or item not in list2:
                tmp3.append(item)
            else:
                if item not in common:
                    common.append(item)
        return tmp1, tmp2, tmp3, common


# # 示例用法


# list1 = [(0, 2), (2, 4), (4, 5), (5, 6), (6, 9), (9, 11), (11, 14), (14, 15), (15, 17), (17, 19), (19, 22), (22, 25), (25, 28), (28, 29), (29, 31), (31, 33), (33, 36), (36, 39), (39, 42), (42, 43), (43, 45), (45, 47), (47, 48), (48, 49), (49, 51), (51, 53), (53, 55), (55, 57), (57, 59), (59, 60)]
# list2 = [(0, 9), (9, 11), (11, 14), (14, 15), (15, 22), (22, 23), (23, 25), (25, 28), (28, 29), (29, 36), (36, 37), (37, 39), (39, 42), (42, 43), (43, 45), (45, 47), (47, 48), (48, 49), (49, 51), (51, 53), (53, 55), (55, 57), (57, 59), (59, 60)]
# list3 = [(0, 4), (4, 6), (6, 9), (9, 11), (11, 14), (14, 15), (15, 19), (19, 22), (22, 23), (23, 25), (25, 28), (28, 29), (29, 33), (33, 36), (36, 37), (37, 39), (39, 42), (42, 43), (43, 45), (45, 47), (47, 48), (48, 49), (49, 51), (51, 53), (53, 55), (55, 57), (57, 59), (59, 60)]

# conflict_finder = ConflictFinder(list1, list2, list3)
# span, span_1, span_2, span_3 = conflict_finder.find_conflicts()
# # print(span)
# # print(span_1)
# # print(span_2)
# print(span_3)




# # # 示例用法


# # list1 = [(0, 2), (2, 4), (4, 5), (5, 6), (6, 9), (9, 11), (11, 14), (14, 15), (15, 17), (17, 19), (19, 22), (22, 25), (25, 28), (28, 29), (29, 31), (31, 33), (33, 36), (36, 39), (39, 42), (42, 43), (43, 45), (45, 47), (47, 48), (48, 49), (49, 51), (51, 53), (53, 55), (55, 57), (57, 59), (59, 60)]
# # list2 = [(0, 9), (9, 11), (11, 14), (14, 15), (15, 22), (22, 23), (23, 25), (25, 28), (28, 29), (29, 36), (36, 37), (37, 39), (39, 42), (42, 43), (43, 45), (45, 47), (47, 48), (48, 49), (49, 51), (51, 53), (53, 55), (55, 57), (57, 59), (59, 60)]
# # list3 = [(0, 4), (4, 6), (6, 9), (9, 11), (11, 14), (14, 15), (15, 19), (19, 22), (22, 23), (23, 25), (25, 28), (28, 29), (29, 33), (33, 36), (36, 37), (37, 39), (39, 42), (42, 43), (43, 45), (45, 47), (47, 48), (48, 49), (49, 51), (51, 53), (53, 55), (55, 57), (57, 59), (59, 60)]

# # conflict_finder = conflictmuliFinder(list1, list2, list3)
# # con,not_con = conflict_finder.find_conflict_muli()
# # print(con)
# # print(not_con)