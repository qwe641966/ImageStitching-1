# Image Stitching
## Requirement
操作系统：Windows 10/Linux或其他类Unix操作系统
Python 3.6以上
OpenCV 3.4以下（因为OpenCV3.4以上取消了sift特征检测器，如需要则必须从源码编译）
## 程序说明
从视频文件使用cv2的VideoCapture来获取视频帧，对视频帧灰度化之后，提取其中的SIFT特征点。
获取两个视频帧的SIFT特征点的匹配（使用KNN算法），并根据匹配计算得到单应性矩阵H，通过仿射变换将右视频帧映射到左视频帧的坐标系内。
通过简单复制将两个视频帧同步到同一个坐标系内。
## TODO
1.集成拉普拉斯金字塔进行图像融合
2.优化图像拼接的效率
