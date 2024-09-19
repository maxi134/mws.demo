import json

def split_leaf_node(node):
    if 'children' in node and not node['children']:
        # 当前节点是叶子节点，拆分名称
        names = list(node['name'])
        node['children'] = [
            {"name": name, "value": None, "children": []}
            for name in names
        ]
    else:
        for child in node.get('children', []):
            split_leaf_node(child)

def modify_non_leaf_nodes(node, is_root=False):
    if 'children' in node and node['children']:
        # 当前节点是非叶子节点
        if is_root:
            node['name'] = "S"
        else:
            node['name'] = "W"
        node['value'] = None
        for child in node['children']:
            modify_non_leaf_nodes(child, False)
    else:
        # 当前节点是叶子节点，不需要修改
        pass

def insert_c_nodes(node):
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
                insert_c_nodes(child)
                new_children.append(child)
        node['children'] = new_children

data = {
    "name": "基于片段的多粒度分词",
    "value": None,
    "children": [
        {
            "name": "基于",
            "value": 0.99,
            "children": []
        },
        {
            "name": "片段",
            "value": 1.0,
            "children": []
        },
        {
            "name": "的",
            "value": 1.0,
            "children": []
        },
        {
            "name": "多粒度分词",
            "value": 0.33,
            "children": [
                {
                    "name": "多",
                    "value": 0.36,
                    "children": []
                },
                {
                    "name": "粒度",
                    "value": 0.31,
                    "children": []
                },
                {
                    "name": "分",
                    "value": 0.36,
                    "children": []
                },
                {
                    "name": "词",
                    "value": 0.4,
                    "children": []
                }
            ]
        }
    ]
}

split_leaf_node(data)
modify_non_leaf_nodes(data, True)
insert_c_nodes(data)

data = json.dumps(data, ensure_ascii=False, indent=4)
print(data)