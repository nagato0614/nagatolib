#!/usr/bin/env python
import numpy as np
from tensorflow.keras.datasets import mnist

def main():
    # MNISTデータセットをロードします。
    # x_train, x_test はそれぞれ (num_samples, 28, 28) の配列、
    # y_train, y_test はそれぞれ (num_samples,) のラベルです。
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 画像データは(28,28)の2次元なので、1枚ずつ1行に出力するために
    # 1次元にフラット化 (28*28=784次元) します。
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat  = x_test.reshape(x_test.shape[0], -1)
    
    # 画像データをCSVで保存します。
    # 1行ごとに1画像（各ピクセルの値がカンマ区切り）となります。
    np.savetxt("train_data.csv", x_train_flat, delimiter=",", fmt="%d")
    np.savetxt("test_data.csv",  x_test_flat,  delimiter=",", fmt="%d")
    
    # ラベルは1行に1つの答えとしてCSVに保存します。
    np.savetxt("train_label.csv", y_train, delimiter=",", fmt="%d")
    np.savetxt("test_label.csv",  y_test,  delimiter=",", fmt="%d")
    
    print("CSVファイルの作成が完了しました。")

if __name__ == "__main__":
    main()