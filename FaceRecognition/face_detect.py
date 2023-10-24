import cv2 as cv


def face_detect_demo(img):
    # 将图片转换为灰度图片
    gray = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    # 加载特征数据
    face_detector = cv.CascadeClassifier('E:\\OpenCV\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml')
    faces = face_detector.detectMultiScale(gray)
    print(faces)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    cv.imshow('result',img)

# 加载图片
img = cv.imread('girl04.jpg')
# 人脸检测
face_detect_demo(img)
cv.waitKey(0)
cv.destroyAllWindows()
