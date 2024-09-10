

import subprocess
import os
import ast

class Mws:
    def __init__(self, model_dir="exp/2024-08-21.split_mean_marginal_fine.xx_0.2-0.5", pred_dir="predict/demo", device=0, cws_feat="bert"):
        """
        初始化预测器类。
        :param model_dir: 模型文件所在目录
        :param pred_dir: 预测结果输出目录
        :param device: 使用的设备编号（默认0表示GPU 0号设备，-1表示CPU）
        :param cws_feat: 特征文件路径
        """
        self.model_dir = model_dir
        self.pred_dir = pred_dir
        self.device = device
        self.cws_feat = cws_feat

    @ staticmethod
    def write_(sentence):
        tmp_file = "data/tmp_file"
        with open(tmp_file, 'w', encoding='utf-8') as f:
            for item in sentence:
                f.writelines(item + '\t' + 's' + '\n')
        return tmp_file

    @ staticmethod
    def visualize_multigranularity_segmentation(pred_file):

        """
        根据多粒度分词结果对文本进行可视化，使用累加括号表示不同粒度。
        """
        with open(pred_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sentence, segments = lines[0].strip().replace(" ", ""), ast.literal_eval(lines[1].strip())
        mws_result, ctb_result, msr_result, pku_result = segments
        marginal = [item[2] for item in mws_result[0]]
        mws_result = [item[:-1] for item in mws_result[0]]

        # 按照索引从小到大排序，确保较短的粒度在外层，较长的粒度在内层
        sorted_indices = sorted(range(len(mws_result)), key=lambda i: (mws_result[i][0], mws_result[i][1]))
        mws_result = [mws_result[i] for i in sorted_indices]
        marginal = [marginal[i] for i in sorted_indices]

        bracket_arr = [""] * (int(mws_result[-1][-1]) + 1)
        for start, end in mws_result:
            bracket_arr[start] += '{'
            bracket_arr[end] += '}'
        for i in range(len(sentence)):
            bracket_arr[i] += sentence[i]
        mws_res = "".join(bracket_arr)
        ctb_res = [sentence[start:end] for start, end in ctb_result[0]]
        msr_res = [sentence[start:end] for start, end in msr_result[0]]
        pku_res = [sentence[start:end] for start, end in pku_result[0]]
        return [mws_res, ctb_res, msr_res, pku_res, marginal]



    def predict(self, fdata):
        """
        执行预测功能，fdata 可以是文件路径或者直接传入字符串。
        :param fdata: 输入数据，可以是文件路径或直接的字符串数据
        :return: 预测结果文件路径
        """
        # 判断 fdata 是文件路径还是直接的字符串
        if os.path.exists(fdata):
            input_data = fdata  # 直接使用文件路径
        else:
            # 创建临时文件并将字符串数据写入其中
            input_data = self.write_(fdata)

        command = [
            "python", "-u", "run.py", "predict",
            f"-d={self.device}",
            f"-f={self.model_dir}",
            f"--feat={self.cws_feat}",
            f"--fdata={input_data}",
            f"--fpred={self.pred_dir}"
        ]
        # process = subprocess.Popen(command, stderr=subprocess.STDOUT, text=True)
        log_file = 'log/predict.log'
        try:
            with open(log_file, 'w') as logfile:
                process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, text=True)
                process.wait()  # 等待进程完成
            return self.visualize_multigranularity_segmentation(self.pred_dir)
        except subprocess.CalledProcessError as e:
            print(f"预测失败，错误信息已保存至: {log_file}")
            raise

        # 如果创建了临时文件，删除该文件
        if not os.path.exists(fdata):
            os.remove(input_data)

        # return self.pred_dir

if __name__ == '__main__':
    mws = Mws()
    data = mws.predict("多粒度分词")
    print(data)