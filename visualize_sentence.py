import matplotlib.pyplot as plt

from collections import deque

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []

    def __repr__(self):
        return f"Node({self.value})"


class Build_Tree:
    def __init__(self,intervals):
        self.intervals = intervals
        self.root = (self.intervals[0][0], self.intervals[-1][-1])
    def find_parent(self):

        parent_dict = {}
        for interval in self.intervals:
            candiates = [item for item in self.intervals if item[0] <= interval[0] and item[1] >= interval[1] and item != interval]
            if candiates:
                parent_dict[interval] = min(candiates, key=lambda n: n[1]-n[0])
            else:
                parent_dict[interval] = self.root
        return parent_dict

    def build_tree_from_relationships(self):
        """
        根据节点之间的关系字典构建树结构。

        :param parent_dict: 字典，记录每个节点的父节点关系
        :param root_interval: 根节点的区间
        :return: 根节点对象
        """
        parent_dict = self.find_parent()
        root_value = (self.intervals[0][0], self.intervals[-1][-1])
        # 初始化一个字典来存储节点
        nodes = {}

        # 创建根节点
        root = Node(root_value)
        nodes[root_value] = root

        # 创建其他节点
        for interval in parent_dict.keys():
            if interval != root_value:
                nodes[interval] = Node(interval)
        # 构建树结构
        for interval, parent in parent_dict.items():
            if interval != root_value:
                if parent is None:
                    # 如果没有找到父节点，则使用根节点作为当前节点的父节点
                    root.children.append(nodes[interval])
                else:
                    # 否则将当前节点添加到其父节点的 children 列表中
                    nodes[parent].children.append(nodes[interval])

        return root

    def generate_node_and_links(self):
        """
        从给定的父子关系字典中生成节点和链接数据，适用于ECharts图表。

        参数:
        parent_dict (dict): 字典形式表示的父子节点关系，键是子节点，值是父节点。

        返回:
        tuple: 包含两个元素的元组，第一个元素是节点信息列表，第二个元素是链接信息列表。
        """

        # 提取所有节点
        parent_dict = self.find_parent()
        all_nodes = set()
        for key, value in parent_dict.items():
            all_nodes.add(key)
        # 转换为列表
        nodes_list = list(all_nodes)

        # 创建 data 列表
        nodes = [{"name": str(item)} for item in nodes_list]

        # 创建 links 列表
        links = []
        for (child_x, child_y), (parent_x, parent_y) in parent_dict.items():
            links.append({
                "source": f"({child_x}, {child_y})",
                "target": f"({parent_x}, {parent_y})"
            })

        return nodes, links


    def print_tree(self, root, indent=0):
        print('  ' * indent + str(root.value))
        for child in root.children:
            self.print_tree(child, indent + 3)




if __name__ == '__main__':
    intervals = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4), (4, 6)]
    tree = Build_Tree(intervals)
    root = tree.build_tree_from_relationships()
    # tree.print_tree(root, indent=0)
    nodes, links = tree.generate_node_and_links()
    breakpoint()