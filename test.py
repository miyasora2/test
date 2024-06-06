import cv2

# 画像読み込み
image = cv2.imread('image.jpg')

# グレースケール変換
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔検出
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# 検出された顔を矩形で囲む
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 結果を表示
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
