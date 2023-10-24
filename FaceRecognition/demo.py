import cv2 as cv


for i in range(1,11):
    path = 'girl'+'0'+str(i)+'.jpg'
    if i == 10:
        path = 'girl10.jpg'
    # 读取图片，路径不能有中文
    img = cv.imread(path)
    # 画矩形框
    x,y,w,h=100,100,80,80#(x,y)表示矩形框左上角顶点的坐标，w和h表示宽度和高度
    cv.rectangle(img,(x,y,x+w,y+h),color=(0,255,0),thickness=2) #color=BGR
    # 绘制圆形 center元组指的是原点坐标 radius：半径
    x,y,r=200,200,100
    cv.circle(img,center=(x,y),radius=r,color=(0,0,255),)
    # 第一个参数是给图像起一个名称，第二个参数是要加载的图片
    cv.imshow('input image', mat=img)
    # 等待键盘的输入 单位是毫秒 传入0会无限等待
    cv.waitKey(0)

    # 转换为灰度图
    # cv2读取图片的通道是BGR
    # PIL读取图片的通道是RGB
    gray_img = cv.cvtColor(img, code=cv.COLOR_BGR2GRAY)
    cv.imshow('gray_img', gray_img)
    cv.waitKey(0)


    # 查看原来图片的形状
    print('原图的大小为：', img.shape)
    # 修改图片的大小
    resize_img = cv.resize(img, dsize=(int(img.shape[1] * 0.9), int(img.shape[0] * 0.9)))
    print('修改后的大小为：', resize_img.shape)
    cv.imshow('resize_img', resize_img)
    cv.waitKey(0)
    cv.imwrite(('resize_' + path), resize_img)

    # C++语言  使用完内存必须释放
    cv.destroyAllWindows()
