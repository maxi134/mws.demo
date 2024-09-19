#
# import json
#
# class Node:
#     def __init__(self, value):
#         self.value = value
#         self.children = []
#
#     def __repr__(self):
#         return f"Node({self.value})"
#
# class Build_Tree:
#     def __init__(self, data):
#         self.sentence = data[0]
#         self.mws_res = data[1]
#         self.mws_marginal = data[2]
#         self.ctb_res = data[3]
#         self.msr_res = data[4]
#         self.pku_res = data[5]
#         self.root = (self.mws_res[0][0], self.mws_res[-1][-1])
#
#     def find_parent(self):
#         parent_dict = {}
#         for interval in self.mws_res:
#             candidates = [item for item in self.mws_res if item[0] <= interval[0] and item[1] >= interval[1] and item != interval]
#             if candidates:
#                 parent_dict[interval] = min(candidates, key=lambda n: n[1]-n[0])
#             else:
#                 parent_dict[interval] = self.root
#         parent_index = {}
#         for child, parent in parent_dict.items():
#             try:
#                 parent_index[self.mws_res.index(child)] = self.mws_res.index(parent)
#             except ValueError:
#                 parent_index[self.mws_res.index(child)] = -1
#         return parent_index
#
#     def build_tree_from_relationships(self):
#         parent_index = self.find_parent()
#         nodes = {-1: Node(-1)}
#         for interval in parent_index.keys():
#             if interval != -1:
#                 nodes[interval] = Node(interval)
#         for child, parent in parent_index.items():
#             if child != -1:
#                 if parent == -1:
#                     nodes[-1].children.append(nodes[child])
#                 else:
#                     nodes[parent].children.append(nodes[child])
#         return nodes[-1]
#
#     def build_json_tree(self, root):
#         if root is None:
#             return {}
#         if root.value == -1:
#             name = self.sentence
#             value = None
#         else:
#             span = self.mws_res[root.value]
#             name = self.sentence[span[0]:span[1]]
#             value = self.mws_marginal[root.value]
#         node_json = {
#             "name": str(name),
#             "value": value,
#             "children": []
#         }
#         for child in root.children:
#             node_json["children"].append(self.build_json_tree(child))
#         return node_json
#
#     def generate_json_tree(self, data):
#         """
#         接受数据并生成JSON字符串表示的树结构。
#
#         :param data: 包含句子及其分词结果的数据列表
#         :return: JSON格式的字符串
#         """
#         self.__init__(data)
#         root = self.build_tree_from_relationships()
#         data = self.build_json_tree(root)
#         return json.dumps(data, ensure_ascii=False, indent=4)
#
# if __name__ == '__main__':
#     data = ['基于片段的多粒度分词',
#             [(0, 2), (2, 4), (4, 5), (5, 6), (5, 10), (6, 8), (8, 9), (9, 10)],
#             [0.99, 1.0, 1.0, 0.36, 0.33, 0.31, 0.36, 0.4],
#             [(0, 2), (2, 4), (4, 5), (5, 10)],
#             [(0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 9), (9, 10)],
#             [(0, 2), (2, 4), (4, 5), (5, 10)]]
#     new_data = ['基于片段的多粒度分词',
#                 [(0, 2), (2, 4), (4, 5), (5, 6), (5, 10), (6, 8), (8, 9), (9, 10)],
#                 [0.99, 1.0, 1.0, 0.36, 0.33, 0.31, 0.36, 0.4],
#                 [(0, 2), (2, 4), (4, 5), (5, 10)],
#                 [0.98, 1.0, 1.0, 0.58],
#                 [(0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 9), (9, 10)],
#                 [1.0, 1.0, 1.0, 0.9, 0.81, 0.88, 0.91],
#                 [(0, 2), (2, 4), (4, 5), (5, 10)],
#                 [1.0, 1.0, 1.0, 0.4]]
#
#     tree = Build_Tree(data)
#     json_string = tree.generate_json_tree(data)
#     print(json_string)



import json

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def __repr__(self):
        return f"Node({self.value})"

class Build_Tree:
    def __init__(self, data):
        self.sentence = data[0]
        self.mws_res = data[1]
        self.mws_marginal = data[2]
        self.ctb_res = data[3]
        self.ctb_marginal = data[4]
        self.msr_res = data[5]
        self.msr_marginal = data[6]
        self.pku_res = data[7]
        self.pku_marginal = data[8]
        self.root = (self.mws_res[0][0], self.mws_res[-1][-1])

    def obtain_candidate(self):
        candidate_dict = {}
        for i in range(len(self.ctb_res)):
            if self.ctb_res[i] not in self.mws_res:
                candidate_dict[f"{self.sentence[self.ctb_res[i][0]:self.ctb_res[i][1]]}"] = self.ctb_marginal[i]
        for i in range(len(self.msr_res)):
            if self.msr_res[i] not in self.mws_res:
                candidate_dict[f"{self.sentence[self.msr_res[i][0]:self.msr_res[i][1]]}"] = self.msr_marginal[i]
        for i in range(len(self.pku_res)):
            if self.pku_res[i] not in self.mws_res:
                candidate_dict[f"{self.sentence[self.pku_res[i][0]:self.pku_res[i][1]]}"] = self.pku_marginal[i]
        return candidate_dict

    def find_parent(self):
        parent_dict = {}
        for interval in self.mws_res:
            candidates = [item for item in self.mws_res if item[0] <= interval[0] and item[1] >= interval[1] and item != interval]
            if candidates:
                parent_dict[interval] = min(candidates, key=lambda n: n[1]-n[0])
            else:
                parent_dict[interval] = self.root
        parent_index = {}
        for child, parent in parent_dict.items():
            try:
                parent_index[self.mws_res.index(child)] = self.mws_res.index(parent)
            except ValueError:
                parent_index[self.mws_res.index(child)] = -1
        return parent_index

    def build_tree_from_relationships(self):
        parent_index = self.find_parent()
        nodes = {-1: Node(-1)}
        for interval in parent_index.keys():
            if interval != -1:
                nodes[interval] = Node(interval)
        for child, parent in parent_index.items():
            if child != -1:
                if parent == -1:
                    nodes[-1].children.append(nodes[child])
                else:
                    nodes[parent].children.append(nodes[child])
        return nodes[-1]

    def build_json_tree(self, root):
        if root is None:
            return {}
        if root.value == -1:
            name = self.sentence
            value = None
        else:
            span = self.mws_res[root.value]
            name = self.sentence[span[0]:span[1]]
            value = self.mws_marginal[root.value]
        node_json = {
            "name": str(name),
            "value": value,
            "children": []
        }
        for child in root.children:
            node_json["children"].append(self.build_json_tree(child))
        return node_json

    def create_segment_dict(self, list1, list2):
        """
        将片段信息和对应的值组合成一个字典格式。

        :param list1: 一个元组列表，每个元组包含两个整数，表示片段的开始和结束位置。
        :param list2: 一个浮点数列表，每个浮点数对应一个片段的值。
        :return: 一个字典，包含所有片段的信息。
        """
        # 创建一个空列表来存储每个片段的字典
        segment_dicts = {}

        # 遍历两个列表，创建字典并添加到列表中
        for segment, value in zip(list1, list2):
            segment_dicts[f"{self.sentence[segment[0]:segment[-1]]}"] = value


        return segment_dicts


    def generate_json_tree(self, data):
        """
        接受数据并生成JSON字符串表示的树结构。这里的data不影响最终结果，只要data[0]和data[1]不变就可以

        :param data: 包含句子及其分词结果的数据列表
        :return: JSON格式的字符串
        """
        self.__init__(data)
        root = self.build_tree_from_relationships()
        data = self.build_json_tree(root)
        ctb_dict = self.create_segment_dict(self.ctb_res, self.ctb_marginal)
        msr_dict = self.create_segment_dict(self.msr_res, self.msr_marginal)
        pku_dict = self.create_segment_dict(self.pku_res, self.pku_marginal)
        mws_sws_dict = {}
        mws_sws_dict['mws_res'] = data
        mws_sws_dict['ctb_res'] = ctb_dict
        mws_sws_dict['msr_res'] = msr_dict
        mws_sws_dict['pku_res'] = pku_dict
        candidate_dict = self.obtain_candidate()
        mws_sws_dict['candiate_words'] = candidate_dict
        return json.dumps(mws_sws_dict, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    data = ['基于片段的多粒度分词',
            [(0, 2), (2, 4), (4, 5), (5, 6), (5, 10), (6, 8), (8, 9), (9, 10)],
            [0.99, 1.0, 1.0, 0.36, 0.33, 0.31, 0.36, 0.4],
            [(0, 2), (2, 4), (4, 5), (5, 10)],
            [(0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 9), (9, 10)],
            [(0, 2), (2, 4), (4, 5), (5, 10)]]
    new_data = ['基于片段的多粒度分词',
                [(0, 2), (2, 4), (4, 5), (5, 6), (5, 10), (6, 8), (8, 9), (9, 10)],
                [0.99, 1.0, 1.0, 0.36, 0.33, 0.31, 0.36, 0.4],
                [(0, 2), (2, 4), (4, 5), (5, 10)],
                [0.98, 1.0, 1.0, 0.58],
                [(0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 9), (9, 10)],
                [1.0, 1.0, 1.0, 0.9, 0.81, 0.88, 0.91],
                [(0, 2), (2, 4), (4, 5), (5, 10)],
                [1.0, 1.0, 1.0, 0.4]]

    tree = Build_Tree(new_data)
    json_string = tree.generate_json_tree(new_data)
    print(json_string)