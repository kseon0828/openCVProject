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