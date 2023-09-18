from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import pandas as pd
import json
import glob
import os


def load_csv_file(path):
    """
    :param path: csv file path
    the file must be fomated by
    first row  : predict  data
    second row : label    data
    """
    result = pd.read_csv(path, header=None, sep=',')
    result = result.values[:, :-1]
    pred_idx = result[0]
    truth_idx = result[1]

    return pred_idx, truth_idx


class ConfusionMatrix(object):

    def __init__(self, y_true, y_pred, classes, exclude_class, decimal=4):
        '''
        :param y_true:
        :param y_pred:
        :param decimal: nums of decimal
        for example:
        y_true = ['2', '0', '0', '1', '2', '0']
        y_pred = ['2', '0', '0', '1', '2', '1']

        y_true = ['cat', 'dog', 'cat']
        y_pred = ['cat', 'dog', 'dog']
        '''
        self.decimal = decimal  # 结果保留小数点后几位
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = unique_labels(self.y_true, self.y_pred)  # 得到labels
        self.label_num = len(self.labels)  # labels数量
        self.c = np.array(confusion_matrix(y_true, y_pred, labels=self.labels), np.float32)  # 计算混淆矩阵
        self.classes = classes
        self.exclude_class = exclude_class

    # ==========================================================================================#
    #                               一级指标：混淆矩阵                                          #
    # ==========================================================================================#

    def plot_confusion_matrix(self, output_path, mode='series'):
        """
        plot confusion matrix
        :return:
        """
        sns.set()
        f, ax = plt.subplots(figsize=(6, 4))  # 18 12
        # cmap="YlGnBu",
        # fmt: 表格里显示数据的类型
        # fmt = '.0%'  # 显示百分比
        # fmt = 'f'
        # 显示完整数字 = fmt = 'g'
        # fmt = '.3'
        # 显示小数的位数 = fmt = '.3f' = fmt = '.3g'
        yticklabels = xticklabels = self.classes
        sns.heatmap(self.c, annot=True, cmap="Blues", xticklabels=xticklabels, yticklabels=yticklabels,
                    linewidths=0.05, fmt='.0f')  # .invert_yaxis()# 反轴显示
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('y_pred')
        ax.set_ylabel('y_true')
        # flights_long = sns.load_dataset("flights")
        # flights = flights_long.pivot("month", "year", "passengers")
        # 设置坐标字体方向
        label_y = ax.get_yticklabels()
        plt.setp(label_y, rotation=360, horizontalalignment='right')
        label_x = ax.get_xticklabels()
        plt.setp(label_x, rotation=45, horizontalalignment='right')

        plt.savefig('%s/confusion_matrix_%s.jpg' % (output_path, mode))

    def level_1(self):
        """
        print confusion matrix
        """
        print(self.c)

    # ==========================================================================================#
    #   二级指标：准确率（Accuracy）, 精确率（Precision），召回率（Recall）, 特异度（Specificity）#
    # ==========================================================================================#
    def accuracy(self, output_path, model_name):
        """
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        :return:
        """
        acc_file = open('%s/acc.txt' % output_path, 'a')
        acc, num = 0, 0

        for i in range(len(self.y_true)):
            true_label = self.classes[int(self.y_true[i])]
            if true_label in self.exclude_class:
                continue
            num += 1
            if self.y_true[i] == self.y_pred[i]:
                acc += 1
        mean_acc = round(acc / num, self.decimal)
        # print(num, acc, mean_acc)
        acc_file.write('%s, accuracy:%.4f\n' % (mode, mean_acc))
        print('%s, accuracy:%.4f' % (model_name, mean_acc))

    def level_2(self, output_path, model_name):
        '''
        accuracy、Precision、Recall、Specificity
        '''

        self.accuracy(output_path, model_name)
        acc_file = open('%s/acc.txt' % output_path, 'a')

        acc = {}
        class_nums = {}
        for label in self.classes:
            acc[label] = 0
            class_nums[label] = 0
        for i in range(len(self.y_true)):
            true_label = self.classes[int(self.y_true[i])]
            class_nums[true_label] += 1
            if self.y_true[i] == self.y_pred[i]:
                acc[true_label] += 1

        for i in range(self.label_num):
            true_label = self.classes[i]
            if true_label in self.exclude_class:
                continue
            if class_nums[true_label] != 0:
                mean_acc = round(acc[true_label] / class_nums[true_label], self.decimal)
            else:
                mean_acc = 0.0
            print("%s, accuracy: %.4f (%d/%d)" % (true_label, mean_acc, acc[true_label], class_nums[true_label]))
            acc_file.write(
                "%s, accuracy: %.4f (%d/%d)" % (true_label, mean_acc, acc[true_label], class_nums[true_label]) + '\n')

    def _bi_matrix(self, i):
        bi_arr = np.zeros((2, 2), np.float32)
        # tp
        bi_arr[0, 0] = self.c[i, i]
        # fn
        bi_arr[0, 1] = np.sum(self.c[i, :]) - self.c[i, i]
        # fp
        bi_arr[1, 0] = np.sum(self.c[:, i]) - self.c[i, i]
        # tn
        bi_arr[1, 1] = np.sum(self.c[:, :]) - bi_arr[0, 0] - bi_arr[0, 1] - bi_arr[1, 0]
        return bi_arr

    def recall(self, config=True):
        """
        recall = TP / (TP + FN)
        :param config: print infomation or not
        :return:
        """
        Recall = {}
        for i in range(self.label_num):
            bi_arr = self._bi_matrix(i)
            tp = bi_arr[0, 0]
            fn = bi_arr[0, 1]
            if tp == 0:
                Recall[self.classes[i]] = 0
                if config == True:
                    print("class:", self.classes[i], "    Recall:", 0)
            else:
                Recall[self.classes[i]] = round(tp / (tp + fn), self.decimal)
                if config == True:
                    print("class:", self.classes[i], "    Recall:", round(tp / (tp + fn), self.decimal))
        return Recall

    def precision(self, config=True):
        """
        precision = TP / (TP + FP)
        :param config:
        :return:
        """

        Precision = {}
        for i in range(self.label_num):
            bi_arr = self._bi_matrix(i)
            tp = bi_arr[0, 0]
            fp = bi_arr[1, 0]
            if tp == 0:
                Precision[self.classes[i]] = 0
                if config == True:
                    print("class:", self.classes[i], "    Precision:", 0)
            else:
                Precision[self.classes[i]] = round(tp / (tp + fp), self.decimal)
                if config == True:
                    print("class:", self.classes[i], "    Precision:", round(tp / (tp + fp), self.decimal))
        return Precision

    def F1_score(self, mode, config=False):
        """
        F1 score = 2PR / (P + R)
        :return:
        """
        f1_file = open('%s/f1_score.txt' % output_path, 'a')

        total_f1 = 0
        p = self.precision(config=config)
        r = self.recall(config=config)
        for i in range(self.label_num):
            cp = p[self.classes[i]]
            cr = r[self.classes[i]]
            if cp == 0 or cr == 0:
                F1_score = 0
            else:
                F1_score = 2. * cp * cr / (cp + cr)
            total_f1 += F1_score
            # f1_file.write("class: %s    F1_score: %s" % (self.classes[i], round(F1_score, self.decimal)) + '\n')
            print("class:", self.classes[i], "    F1_score:", round(F1_score, self.decimal))
        # f1_file.write("Mode: %s, Mean F1_score: %s" % (mode, str((round(total_f1/len(self.classes), self.decimal)))) + '\n')
        print("Mode: %s, Mean F1_score: %s" % (mode, round(total_f1 / 5, self.decimal)))


if __name__ == "__main__":

    breast_classes = ["3", "4A", "4B", "4C", "5"]

    valid_classes, exclude_class = [], []
    root_path = r'../results/multi-label/20210628-052540-tf_efficientnet_b0_ns-512'
    result_path = root_path + '/results-birads.csv'
    mode = result_path.split('/')[-1].split('.')[0].split('-')[-1]
    output_path = '/'.join(result_path.split('/')[:-1])
    dataset = root_path.split('/')[-2]

    y_pred, y_true = load_csv_file(result_path)
    c = ConfusionMatrix(y_true, y_pred, breast_classes, exclude_class)
    c.F1_score(mode, config=True)
    c.level_2(output_path, model_name=mode)  # 同时输出准确率，精确率，召回率，特异度
    c.plot_confusion_matrix(output_path=output_path, mode=mode)  # 混淆矩阵热力图可视化
