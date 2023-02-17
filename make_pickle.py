
# -*- coding: utf-8 -*

import glob
from sys import maxsize
import cv2
import matplotlib.pyplot as plt
import pickle


out_dir = "データセットのパスを入力" # 画像データのあるディレクトリ
save_file = "保存先のパスを入力/pose_emission.pickle" # 保存先

result = []
maxsize = 10000
fs = glob.glob(out_dir+"/*")
for i,labeldir in enumerate(fs):
    print("i=",i)
    print("labeldir=",labeldir)
    fs0 = glob.glob(labeldir+"/*")
    for j,f in enumerate(fs0):
        if j>=maxsize:break
        result.append([i,cv2.imread(f)])

pickle.dump(result, open(save_file, "wb"))
print("done!")