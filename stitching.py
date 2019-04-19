# -*- coding:utf-8 -*-
import cv2
import numpy as np
import blender
import time


def compute_matches(features0, features1, matcher, knn=5, lowe=0.7):
    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1

    print(u'寻找相似特征点')

    matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)

    print("使用lowe测试过滤匹配点")

    positive = []
    for match0, match1 in matches:
        if match0.distance < lowe * match1.distance:
            positive.append(match0)

    src_pts = np.array(
        [keypoints0[good_match.queryIdx].pt for good_match in positive], dtype=np.float32)
    src_pts = src_pts.reshape((-1, 1, 2))
    dst_pts = np.array(
        [keypoints1[good_match.trainIdx].pt for good_match in positive], dtype=np.float32)
    dst_pts = dst_pts.reshape((-1, 1, 2))

    return src_pts, dst_pts, len(positive)


def Laplacian_blending(img1, img2):
    levels = 3
    # generating Gaussian pyramids for both images
    gpImg1 = [img1.astype('float32')]
    gpImg2 = [img2.astype('float32')]
    for i in range(levels):
        img1 = cv2.pyrDown(img1)  # Downsampling using Gaussian filter
        gpImg1.append(img1.astype('float32'))
        img2 = cv2.pyrDown(img2)
        gpImg2.append(img2.astype('float32'))

    # Generating Laplacin pyramids for both images
    lpImg1 = [gpImg1[levels]]
    lpImg2 = [gpImg2[levels]]

    for i in range(levels, 0, -1):
        # Upsampling and subtracting from upper level Gaussian pyramid image to get Laplacin pyramid image
        tmp = cv2.pyrUp(gpImg1[i]).astype('float32')
        tmp = cv2.resize(tmp, (gpImg1[i - 1].shape[1], gpImg1[i - 1].shape[0]))
        lpImg1.append(np.subtract(gpImg1[i - 1], tmp))

        tmp = cv2.pyrUp(gpImg2[i]).astype('float32')
        tmp = cv2.resize(tmp, (gpImg2[i - 1].shape[1], gpImg2[i - 1].shape[0]))
        lpImg2.append(np.subtract(gpImg2[i - 1], tmp))

    laplacianList = []
    for lImg1, lImg2 in zip(lpImg1, lpImg2):
        rows, cols = lImg1.shape
        # Merging first and second half of first and second images respectively at each level in pyramid
        mask1 = np.zeros(lImg1.shape)
        mask2 = np.zeros(lImg2.shape)
        mask1[:, 0:cols / 2] = 1
        mask2[:, cols / 2:] = 1

        tmp1 = np.multiply(lImg1, mask1.astype('float32'))
        tmp2 = np.multiply(lImg2, mask2.astype('float32'))
        tmp = np.add(tmp1, tmp2)

        laplacianList.append(tmp)

    img_out = laplacianList[0]
    for i in range(1, levels + 1):
        # Upsampling the image and merging with higher resolution level image
        img_out = cv2.pyrUp(img_out)
        img_out = cv2.resize(
            img_out, (laplacianList[i].shape[1], laplacianList[i].shape[0]))
        img_out = np.add(img_out, laplacianList[i])

    np.clip(img_out, 0, 255, out=img_out)
    return img_out.astype('uint8')


def combine_images(img0, img1, h_matrix):
    # print('拼接图像... ')

    points0 = np.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=np.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = np.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img0.shape[0]], [img1.shape[1], 0]], dtype=np.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h_matrix)
    points = np.concatenate((points0, points2), axis=0)
    [x_min, y_min] = np.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(points.max(axis=0).ravel() + 0.5)

    # print('points2:', points2[0][0][0])
    # print('points2:', points2[1][0][0])
    # 重叠区域的左边界,开始位置
    start = min(points2[0][0][0], points2[1][0][0])
    # 重叠区域的宽度
    width = img0.shape[1] - start
    # img1中像素的权重
    alpha = 1
    # print('start:',start)
    # print('width:',width)
    # print('img0[0]:', img0[0][0][0])

    #('xmin:', x_min, 'xmax', x_max, 'ymin', y_min, 'ymax', y_max)
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    # print('单应性矩阵Homography:', h_matrix)
    # print('WARP右图...')
    # cv2.imshow('img0', img0)

    # img0是左图
    cv2.imshow('img1', img0)

    # output_img暂时是经过变换之后的右图
    output_img = cv2.warpPerspective(img1, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min),
                                     borderMode=cv2.BORDER_TRANSPARENT)
    #cv2.imwrite('img2.jpg', output_img)
    cv2.imshow('warped', output_img)
    H_img1 = output_img.copy()

    # 拷贝左图的0到start之内的到右图上
    H_img1[:img0.shape[0], :int(start)] = img0[:img0.shape[0], :int(start)]
    output_size = output_img.shape
    cv2.imshow('H_img1', H_img1)
    # print(output_size)
    x_offset = x_max - img1.shape[0]
    # y_offset = y_max-img1.shape[1]

    # 要将左图无缝地加入到变换过后的右图上
    # force combine them together
    output_img[-y_min:img0.shape[0] - y_min,
               - x_min:img0.shape[1] - x_min] = img0

    tmp_r = H_img1[0:y_max, x_max - x_offset:x_max]
    tmp_l = output_img[0:y_max, 0:img0.shape[1]]
    #cv2.imwrite('img1.jpg', output_img)
    # cv2.waitKey(1999)
    # output_image = Laplacian_blending(H_img1.copy(), output_img.copy())

    # cv2.imshow('blend', output_image)
    # 待融合ROI[start:start+width, 0: img0.shape[1]]
    # for x in H_img1[0:img0.shape[1],int(start):int(start)+int(width)]:
    #    for i in x:

    #cv2.imshow(u'Left Img', tmp_r)
    #cv2.imshow(u'Right Img', tmp_l)

    # result = blender.multi_band_blending(tmp_l, tmp_r, overlap_w)
    # cv2.imshow('output_stage_1', result)
    '''
    start to blend
    '''

    return output_img


video1 = 'videos/01/left.avi'
video2 = 'videos/01/right.avi'
img1 = 'videos/01/left.jpg'
img2 = 'videos/01/right.jpg'
# if using OpenCV3
sift = cv2.xfeatures2d.SIFT_create()

result = None
result_gry = None

flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

cap1 = cv2.VideoCapture(video1)
cap2 = cv2.VideoCapture(video2)

img_left = cv2.imread(img1)
img_right = cv2.imread(img2)

img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

features_L = sift.detectAndCompute(img_left_gray, None)
features_R = sift.detectAndCompute(img_right_gray, None)

matches_src, matches_dst, n_matches = compute_matches(
    features_R, features_L, flann, knn=2)

if n_matches < 10:
    print("error! too few correspondences")

H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    '''
    frame_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    features0 = sift.detectAndCompute(frame_gray1, None)
    features1 = sift.detectAndCompute(frame_gray2, None)

    matches_src, matches_dst, n_matches = compute_matches(
        features1, features0, flann, knn=2)

    if n_matches < 10:
        print("error! too few correspondences")
        continue

    H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
    '''
    t0 = time.time()
    result = combine_images(frame1, frame2, H)
    t1 = time.time()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    cv2.waitKey(1)
    #cv2.imshow('Left', frame1)
    #cv2.imshow('Right', frame2)
    cv2.imshow('blended', result)
    print('拼接时间:')
    print((t1 - t0) * 1000, 'ms')

cap1.release()
cv2.destroyAllWindows()
