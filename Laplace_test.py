import cv2

img = cv2.imread('videos/01/left.jpg')
cv2.imshow('src', img)

lower_reso = cv2.pyrDown(img)
print(lower_reso.shape)
cv2.imshow('downsample', lower_reso)

higher_reso = cv2.pyrUp(lower_reso)
print(higher_reso.shape)
cv2.imshow('upsample', higher_reso)

cv2.imshow('laplace', img - higher_reso)

cv2.waitKey(0)
