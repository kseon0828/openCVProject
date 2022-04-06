import cv2
import matplotlib.pyplot as plt
import numpy as np

whiteImg = cv2.imread('./white.png')
srcImg = cv2.imread('./testImage.JPG')

#직선그리기
cv2.line(whiteImg, (50, 50), (200, 50), 5)

#텍스트 그리기
text = "Hello, This is OpenCV project"
cv2.putText(whiteImg, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

cv2.imshow("TextInputTest", whiteImg)
#컬러 매핑
grayImage = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

srcImg = cv2.resize(srcImg, dsize=(640,480), interpolation=cv2.INTER_AREA)
grayImage = cv2.resize(grayImage, dsize=(640, 480), interpolation=cv2.INTER_AREA)

cv2.imshow("src Image", srcImg)

cv2.imshow("gray Image", grayImage)

#이미지 대칭
dst = cv2.flip(srcImg, -1)
cv2.imshow("Flip Image", dst)


#회색 영상의 히스토그램 구하기
hist = cv2.calcHist([grayImage], [0,], None, [256], [0,256])
plt.plot(hist)

cv2.imshow("gray Image", grayImage)

plt.show()

#rgb 영상을 채널별로 분리하여 각 색상별 그래프 plot
channels = cv2.split(srcImg)

colors = ['b', 'g', 'r']
for ch, color in zip(channels, colors):
    hist = cv2.calcHist([ch], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)

cv2.imshow("src Image", srcImg)
plt.show()


#affine transform
rows, cols = srcImg.shape[:2]
pts1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
pts2 = np.float32([[100, 0], [cols-100, 100], [cols-400, 250]])

mat = cv2.getAffineTransform(pts1, pts2)
affineImg = cv2.warpAffine(srcImg, mat, None)

cv2.imshow("affine Image", affineImg)

cv2.waitKey()
cv2.destroyAllWindows()

#해리스 코너 추출
gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

corner = cv2.cornerHarris(gray, 2, 3, 0.04)
# 변화량 결과의 최대값 10% 이상의 좌표 구하기 ---②
coord = np.where(corner > 0.1* corner.max())
coord = np.stack((coord[1], coord[0]), axis=-1)

# 코너 좌표에 동그리미 그리기
for x, y in coord:
    cv2.circle(srcImg, (x,y), 5, (0,0,255), 1, cv2.LINE_AA)

# 변화량을 영상으로 표현하기 위해서 0~255로 정규화
#corner_norm = cv2.normalize(corner, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#corner_norm = cv2.cvtColor(corner_norm, cv2.COLOR_GRAY2BGR)
#merged = np.hstack((corner_norm, srcImg))
cv2.imshow('Harris Corner', srcImg)
cv2.waitKey()
cv2.destroyAllWindows()

#FAST 특징점 검출기
gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

# FASt 특징 검출기 생성
fast = cv2.FastFeatureDetector_create(50)
# 특징점 검출
keypoints = fast.detect(gray, None)
# 특징점 그리기
srcImg = cv2.drawKeypoints(srcImg, keypoints, None)

cv2.imshow('FAST', srcImg)
cv2.waitKey()
cv2.destroyAllWindows()
