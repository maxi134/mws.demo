from flask import Flask, request, jsonify
from mws import Mws  # 导入自定义的 Mws
from mws_no_baike import Mws_no_baike
from flask_cors import CORS  # 导入 CORS
from init_sentence import Node, Build_Tree
app = Flask(__name__)
CORS(app)  # 启用 CORS
# 初始化 Mws 实例
mws = Mws()
mws_no_baike = Mws_no_baike()

@app.route('/segment', methods=['POST'])
def segment():
    # 获取请求的 JSON 数据
    data = request.get_json()

    # 从 JSON 数据中获取句子
    sentence = data.get("sentence")
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    # 使用 Mws 库预测结果
    result = mws.predict(sentence)
    tree = Build_Tree(result)
    json_string = tree.generate_json_tree(result)  # 只返回多粒度分词的结果，可以用于画动态树结构
    # 将结果作为 JSON 返回
    # return jsonify({"sentence": sentence, "prediction": result})
    return json_string

@app.route('/segment_no_baike', methods=['POST'])
def segment_no_baike():
    # 获取请求的 JSON 数据
    data = request.get_json()

    # 从 JSON 数据中获取句子
    sentence = data.get("sentence")
    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400

    result = mws_no_baike.predict(sentence)
    tree = Build_Tree(result)
    json_string = tree.generate_json_tree(result)

    return json_string


if __name__ == '__main__':
    app.run(debug=True)
