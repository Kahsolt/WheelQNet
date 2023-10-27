import numpy as np
import pandas as pd


def eval(test_data_file:str) -> np.ndarray:
  """
  输入test_data 的路径,返回一个numpy数组为选手模型输出的预测结果，其shape为[N],N为测试集个数，其元素为1或者0的预测值。
  对于QTensor，可以使用to_numpy()进行转换成numpy
  """

  pass


if __name__ == "__main__":
  test_data_file = 'data/test_without_label.csv'
  print(eval(test_data_file))
