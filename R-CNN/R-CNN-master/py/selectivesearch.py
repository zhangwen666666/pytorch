# -*- coding: utf-8 -*-

"""
@author: zj
@file:   selectivesearch.py
@time:   2020-02-25
"""

import sys
import cv2

#得到一个Segmentation对象gs
def get_selective_search():
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    return gs


#使用选择性搜索算法生成候选区
def config(gs, img, strategy='q'):
    gs.setBaseImage(img)

    if (strategy == 's'):
        gs.switchToSingleStrategy() #将选择性搜索算法切换到单策略模式,使用单个策略来生成候选区域
    elif (strategy == 'f'):
        gs.switchToSelectiveSearchFast()    #将选择性搜索算法切换到快速模式，用更快的策略来生成候选区
    elif (strategy == 'q'):
        gs.switchToSelectiveSearchQuality() #将选择性搜索算法切换到高质量模式，使用更复杂的策略来生成候选区域
    else:
        print(__doc__)
        sys.exit(1)


def get_rects(gs):
    rects = gs.process()    #调用 seg.process() 后，算法将开始执行，并在图像中查找和合并相似的像素
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects


if __name__ == '__main__':
    """
    选择性搜索算法操作
    """
    gs = get_selective_search()

    img = cv2.imread('./data/lena.jpg', cv2.IMREAD_COLOR)
    config(gs, img, strategy='q')

    rects = get_rects(gs)
    print(rects)
