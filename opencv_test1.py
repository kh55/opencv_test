import cv2 as cv


imagefile = 'adult.jpg'
image = cv.imread(imagefile)

grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# 顔検出
#cascade = cv.CascadeClassifier("./opencv_test1/opencv_test1/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml")
# 瞳検出
cascade = cv.CascadeClassifier("./opencv_test1/opencv_test1/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml")

#results = cascade.detectMultiScale(grayscale, minSize=(50, 50))
results = cascade.detectMultiScale(grayscale)

if len(results) > 0:
    """
    for rectangle in results:
        cv.rectangle(image,tuple(rectangle[0:2]), tuple(rectangle[0:2]+rectangle[2:4]), (0,0,255),thickness=2)

        cv.imwrite('result-'+imagefile,image)

        cv.imshow('imshow: '+imagefile, image)
    """

    for x, y, w, h in results:
        print(x)
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imwrite('./sample_after.png', image)
        
