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
# class TreeNodeModifier:
#     def __init__(self, data):
#         self.data = data
#
#     def split_leaf_node(self, node):
#         if 'children' in node and not node['children']:
#             # 当前节点是叶子节点，拆分名称
#             names = list(node['name'])
#             node['children'] = [
#                 {"name": name, "value": None, "children": []}
#                 for name in names
#             ]
#         else:
#             for child in node.get('children', []):
#                 self.split_leaf_node(child)
#
#     def modify_non_leaf_nodes(self, node, is_root=False):
#         if 'children' in node and node['children']:
#             # 当前节点是非叶子节点
#             if is_root:
#                 node['name'] = "S"
#             else:
#                 node['name'] = "W"
#             node['value'] = None
#             for child in node['children']:
#                 self.modify_non_leaf_nodes(child, False)
#         else:
#             # 当前节点是叶子节点，不需要修改
#             pass
#
#     # def modify_leaf_node_annotation(self, node, is_root=False):
#     #     ctb_lst = [self.sen]
#     #     if 'children' in node and node['children']:
#     #         # 当前节点使非叶子节点
#     #         if is_root:
#     #             node['name'] == "S"
#     #         else:
#     #             tmp = ''
#
#
#     def insert_c_nodes(self, node):
#         if 'children' in node:
#             new_children = []
#             for child in node['children']:
#                 if 'children' in child and not child['children']:
#                     # 当前节点是叶子节点，需要在其上一层新增一个"C"节点
#                     new_child = {
#                         "name": "C",
#                         "value": None,
#                         "children": [child]
#                     }
#                     new_children.append(new_child)
#                 else:
#                     # 当前节点不是叶子节点，递归处理其子节点
#                     self.insert_c_nodes(child)
#                     new_children.append(child)
#             node['children'] = new_children
#
#     def process_tree(self):
#         self.split_leaf_node(self.data)
#         breakpoint()
#         self.modify_non_leaf_nodes(self.data, True)
#         # self.insert_c_nodes(self.data)
#         return self.data
#
# class Build_Tree:
#     def __init__(self, data):
#         self.sentence = data[0]
#         self.mws_res = data[1]
#         self.mws_marginal = data[2]
#         tmp_ctb = zip(data[3],data[4])
#         tmp_ctb = sorted(tmp_ctb,key=lambda x: x[0][0])
#         self.ctb_res, self.ctb_marginal = zip(*tmp_ctb)
#         tmp_msr = zip(data[5], data[6])
#         tmp_msr = sorted(tmp_msr, key=lambda x: x[0][0])
#         self.msr_res, self.msr_marginal = zip(*tmp_msr)
#         tmp_pku = zip(data[7], data[8])
#         tmp_pku = sorted(tmp_pku, key=lambda x: x[0][0])
#         self.pku_res, self.pku_marginal = zip(*tmp_pku)
#         self.root = (self.mws_res[0][0], self.mws_res[-1][-1])
#
#     def obtain_candidate(self):
#         candidate_dict = {}
#         for i in range(len(self.ctb_res)):
#             if self.ctb_res[i] not in self.mws_res:
#                 candidate_dict[f"{self.sentence[self.ctb_res[i][0]:self.ctb_res[i][1]]}"] = self.ctb_marginal[i]
#         for i in range(len(self.msr_res)):
#             if self.msr_res[i] not in self.mws_res:
#                 candidate_dict[f"{self.sentence[self.msr_res[i][0]:self.msr_res[i][1]]}"] = self.msr_marginal[i]
#         for i in range(len(self.pku_res)):
#             if self.pku_res[i] not in self.mws_res:
#                 candidate_dict[f"{self.sentence[self.pku_res[i][0]:self.pku_res[i][1]]}"] = self.pku_marginal[i]
#         return candidate_dict
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
#     def build_json_tree_anna(self, root):
#         if root is None:
#             return {}
#         if root.value == -1:
#             name = self.sentence
#             value = None
#         else:
#             span = self.mws_res[root.value]
#             tmp_name = ""
#             if span in self.ctb_res:
#                 tmp_name += "ctb"
#             if span in self.msr_res:
#                 tmp_name += "msr"
#             if span in self.pku_res:
#                 tmp_name += "pku"
#             name = tmp_name
#             value = self.mws_marginal[root.value]
#         node_json = {
#             "name": str(name),
#             "value": value,
#             "children": []
#         }
#         for child in root.children:
#             node_json["children"].append(self.build_json_tree_anna(child))
#         return node_json
#
#     def create_segment_dict(self, list1, list2):
#         """
#         将片段信息和对应的值组合成一个字典格式。
#
#         :param list1: 一个元组列表，每个元组包含两个整数，表示片段的开始和结束位置。
#         :param list2: 一个浮点数列表，每个浮点数对应一个片段的值。
#         :return: 一个字典，包含所有片段的信息。
#         """
#         # 创建一个空列表来存储每个片段的字典
#         segment_dicts = {}
#
#         # 遍历两个列表，创建字典并添加到列表中
#         for segment, value in zip(list1, list2):
#             segment_dicts[f"{self.sentence[segment[0]:segment[-1]]}"] = value
#
#
#         return segment_dicts
#
#
#     def generate_json_tree(self, data):
#         """
#         接受数据并生成JSON字符串表示的树结构。这里的data不影响最终结果，只要data[0]和data[1]不变就可以
#
#         :param data: 包含句子及其分词结果的数据列表
#         :return: JSON格式的字符串
#         """
#         self.__init__(data)
#         # TODO 已经是字典形式了，所以可以考虑在这里修改
#         root = self.build_tree_from_relationships()
#         # data = self.build_json_tree(root)
#         data_tree = self.build_json_tree_anna(root)   # 可以在节点中显示每个词的规范来源
#         ctb_dict = self.create_segment_dict(self.ctb_res, self.ctb_marginal)
#         msr_dict = self.create_segment_dict(self.msr_res, self.msr_marginal)
#         pku_dict = self.create_segment_dict(self.pku_res, self.pku_marginal)
#         mws_sws_dict = {}
#         processor = TreeNodeModifier(data_tree)
#         processed_data = processor.process_tree()
#         mws_sws_dict['mws_res'] = processed_data
#         mws_sws_dict['ctb_res'] = ctb_dict
#         mws_sws_dict['msr_res'] = msr_dict
#         mws_sws_dict['pku_res'] = pku_dict
#         candidate_dict = self.obtain_candidate()
#         mws_sws_dict['candidate_words'] = candidate_dict
#         return json.dumps(mws_sws_dict, ensure_ascii=False, indent=4)
#
#
#
# if __name__ == '__main__':
#     data = ['基于片段的多粒度分词',
#             [(0, 2), (2, 4), (4, 5), (5, 6), (5, 10), (6, 8), (8, 9), (9, 10)],
#             [0.99, 1.0, 1.0, 0.36, 0.33, 0.31, 0.36, 0.4],
#             [(0, 2), (2, 4), (4, 5), (5, 10)],
#             [(0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 9), (9, 10)],
#             [(0, 2), (2, 4), (4, 5), (5, 10)]]
#     new_data = ['高敖曹加封侍中、开府，进爵武城县侯',
#                 [(0, 3), (3, 4), (3, 5), (4, 5), (5, 7), (7, 8), (8, 10), (10, 11), (11, 13), (13, 16), (13, 17), (16, 17)],
#                 [0.94, 0.63, 0.34, 0.66, 1.0, 1.0, 0.99, 0.99, 0.62, 0.36, 0.43, 0.41],
#                 [(0, 3), (3, 4), (4, 5), (5, 7), (7, 8), (8, 10), (10, 11), (11, 13), (13, 15), (15, 17)],
#                 [0.9, 0.66, 0.71, 1.0, 1.0, 0.99, 0.98, 0.66, 0.6, 0.46],
#                 [(0, 3), (3, 5), (5, 7), (7, 8), (8, 10), (10, 11), (11, 13), (13, 16), (16, 17)],
#                 [0.95, 0.5, 1.0, 1.0, 1.0, 1.0, 0.64, 0.82, 0.82],
#                 [(0, 3), (3, 4), (4, 5), (5, 7), (7, 8), (8, 10), (10, 11), (11, 13), (13, 17)],
#                 [0.96, 0.73, 0.76, 1.0, 1.0, 0.99, 0.99, 0.57, 0.92]]
#
#
#     tree = Build_Tree(new_data)
#     json_string = tree.generate_json_tree(new_data)
#     print(json_string)
import re
import json

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def __repr__(self):
        return f"Node({self.value})"

class TreeNodeModifier:
    def __init__(self, data):
        self.data = data

    def split_leaf_node(self, node):
        if 'children' in node and not node['children']:
            # 当前节点是叶子节点，拆分名称
            names = list(node['name'].split('/')[1])
            node['children'] = [
                {"name": name, "value": None, "children": []}
                for name in names
            ]
        else:
            for child in node.get('children', []):
                self.split_leaf_node(child)

    def modify_non_leaf_nodes(self, node, is_root=False):
        if 'children' in node and node['children']:
            # 当前节点是非叶子节点
            if is_root:
                node['name'] = "S"
            else:
                text = str(node['children'])
                pattern = re.compile(r'[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]')
                children_char = pattern.findall(text)
                children_char = list(set(children_char))
                if len(children_char) < len(node['name'].split('/')[1]):
                    # 当前节点的孩子节点中缺少一些单个字符
                    for item in node['name'].split('/')[1]:
                        if item not in children_char:
                            node['children'].append({'name': item, 'value': None, 'children': []})

                name_ = node['name'].split('/')[0]
                node['name'] = name_
                node['value'] = None
            for child in node['children']:
                self.modify_non_leaf_nodes(child, False)
        else:
            # 当前节点是叶子节点，不需要修改
            pass

    # def modify_leaf_node_annotation(self, node, is_root=False):
    #     ctb_lst = [self.sen]
    #     if 'children' in node and node['children']:
    #         # 当前节点使非叶子节点
    #         if is_root:
    #             node['name'] == "S"
    #         else:
    #             tmp = ''


    def insert_c_nodes(self, node):
        if 'children' in node:
            new_children = []
            for child in node['children']:
                if 'children' in child and not child['children']:
                    # 当前节点是叶子节点，需要在其上一层新增一个"C"节点
                    new_child = {
                        "name": "C",
                        "value": None,
                        "children": [child]
                    }
                    new_children.append(new_child)
                else:
                    # 当前节点不是叶子节点，递归处理其子节点
                    self.insert_c_nodes(child)
                    new_children.append(child)
            node['children'] = new_children

    def process_tree(self):
        self.split_leaf_node(self.data)
        self.modify_non_leaf_nodes(self.data, True)
        # self.insert_c_nodes(self.data)
        return self.data

class Build_Tree:
    def __init__(self, data):
        self.sentence = data[0]
        self.mws_res = data[1]
        self.mws_marginal = data[2]
        tmp_ctb = zip(data[3],data[4])
        tmp_ctb = sorted(tmp_ctb,key=lambda x: x[0][0])
        self.ctb_res, self.ctb_marginal = zip(*tmp_ctb)
        tmp_msr = zip(data[5], data[6])
        tmp_msr = sorted(tmp_msr, key=lambda x: x[0][0])
        self.msr_res, self.msr_marginal = zip(*tmp_msr)
        tmp_pku = zip(data[7], data[8])
        tmp_pku = sorted(tmp_pku, key=lambda x: x[0][0])
        self.pku_res, self.pku_marginal = zip(*tmp_pku)
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

    def build_json_tree_anna(self, root):
        if root is None:
            return {}
        if root.value == -1:
            name = self.sentence
            value = None
        else:
            span = self.mws_res[root.value]
            tmp_name = []
            if span in self.ctb_res:
                index = self.ctb_res.index(span)
                tmp_name.append(('C', self.ctb_marginal[index]))
            if span in self.msr_res:
                index = self.msr_res.index(span)
                tmp_name.append(('M', self.msr_marginal[index]))
            if span in self.pku_res:
                index = self.pku_res.index(span)
                tmp_name.append(('P', self.pku_marginal[index]))
            sorted_name = sorted(tmp_name, key=lambda x: x[1], reverse=True)
            name_lst = [item[0] for item in sorted_name]
            name = '+'.join(name_lst)
            name = name + '/' + self.sentence[span[0]:span[1]]
            value = self.mws_marginal[root.value]
        node_json = {
            "name": str(name),
            "value": value,
            "children": []
        }
        for child in root.children:
            node_json["children"].append(self.build_json_tree_anna(child))
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

    def print_tree(self, node, level=0):
        """
        打印树形结构。

        :param node: 当前节点
        :param level: 当前节点所在的层级，默认为0表示根节点
        """
        if node is not None:
            # 在当前节点之前打印缩进
            print(' ' * 4 * level + '|--', node)
            # 递归地打印子节点
            for child in node.children:
                self.print_tree(child, level + 1)

    def generate_json_tree(self, data):
        """
        接受数据并生成JSON字符串表示的树结构。这里的data不影响最终结果，只要data[0]和data[1]不变就可以

        :param data: 包含句子及其分词结果的数据列表
        :return: JSON格式的字符串
        """
        self.__init__(data)
        # TODO 已经是字典形式了，所以可以考虑在这里修改
        root = self.build_tree_from_relationships()
        # data = self.build_json_tree(root)
        # self.print_tree(root,0)  # 树的构建没有问题

        data_tree = self.build_json_tree_anna(root)   # 可以在节点中显示每个词的规范来源
        ctb_dict = self.create_segment_dict(self.ctb_res, self.ctb_marginal)
        msr_dict = self.create_segment_dict(self.msr_res, self.msr_marginal)
        pku_dict = self.create_segment_dict(self.pku_res, self.pku_marginal)
        mws_sws_dict = {}
        processor = TreeNodeModifier(data_tree)
        processed_data = processor.process_tree()
        mws_sws_dict['mws_res'] = processed_data
        mws_sws_dict['ctb_res'] = ctb_dict
        mws_sws_dict['msr_res'] = msr_dict
        mws_sws_dict['pku_res'] = pku_dict
        candidate_dict = self.obtain_candidate()
        mws_sws_dict['candidate_words'] = candidate_dict
        return json.dumps(mws_sws_dict, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    # data = ['基于片段的多粒度分词',
    #         [(0, 2), (2, 4), (4, 5), (5, 6), (5, 10), (6, 8), (8, 9), (9, 10)],
    #         [0.99, 1.0, 1.0, 0.36, 0.33, 0.31, 0.36, 0.4],
    #         [(0, 2), (2, 4), (4, 5), (5, 10)],
    #         [(0, 2), (2, 4), (4, 5), (5, 6), (6, 8), (8, 9), (9, 10)],
    #         [(0, 2), (2, 4), (4, 5), (5, 10)]]
    # new_data = ['高敖曹加封侍中、开府，进爵武城县侯',
    #             [(0, 3), (3, 4), (3, 5), (4, 5), (5, 7), (7, 8), (8, 10), (10, 11), (11, 13), (13, 16), (13, 17), (16, 17)],
    #             [0.94, 0.63, 0.34, 0.66, 1.0, 1.0, 0.99, 0.99, 0.62, 0.36, 0.43, 0.41],
    #             [(0, 3), (3, 4), (4, 5), (5, 7), (7, 8), (8, 10), (10, 11), (11, 13), (13, 15), (15, 17)],
    #             [0.9, 0.66, 0.71, 1.0, 1.0, 0.99, 0.98, 0.66, 0.6, 0.46],
    #             [(0, 3), (3, 5), (5, 7), (7, 8), (8, 10), (10, 11), (11, 13), (13, 16), (16, 17)],
    #             [0.95, 0.5, 1.0, 1.0, 1.0, 1.0, 0.64, 0.82, 0.82],
    #             [(0, 3), (3, 4), (4, 5), (5, 7), (7, 8), (8, 10), (10, 11), (11, 13), (13, 17)],
    #             [0.96, 0.73, 0.76, 1.0, 1.0, 0.99, 0.99, 0.57, 0.92]]
    data = ['自然科学家们指出：',
            [(0, 2), (0, 4), (0, 5), (4, 5), (5, 6), (6, 8), (8, 9)],
            [0.33, 0.33, 0.34, 0.33, 0.67, 1.0, 1.0],
            [(0, 2), (2, 6), (6, 8), (8, 9)],
            [0.99, 0.99, 1.0, 1.0],
            [(0, 4), (4, 5), (5, 6), (6, 8), (8, 9)],
            [0.98, 0.98, 1.0, 1.0, 1.0],
            [(0, 5), (5, 6), (6, 8), (8, 9)],
            [0.99, 1.0, 1.0, 1.0]]


    tree = Build_Tree(data)
    json_string = tree.generate_json_tree(data)
    print(json_string)