#!C:/Users/foxtrot_delta/.conda/envs/yolov5/python.exe

# print ("Content-type: text/html")
print ()
print ("<html>")
print ("<head>")
print ("<title>Elvenware CGI</title>")
print ("</head>")
print ("<body>")
print ("<h1>Basho!</h1>")
print ("<p>By a Peaceful dark pond</p>")
print ("<p>A frog plops</p>")
print ("<p>Into the still water</p>")
print ("</body>")
print ("</html>")

def main():
    import cv2
    import imutils
    import numpy as np
    from matplotlib import pyplot as plt
    import easyocr

    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized


    img = cv2.imread('C:/Users/foxtrot_delta/Downloads/2.jpg')
    # img = cv2.imread('F:/NUMBERPLATE/yolov4-custom-functions/data/images/mill/M4.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    # plt.show()

    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    # plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    # plt.show()

    kpt = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(kpt)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    print(location)


    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    # plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # plt.show()

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    # new_image = image_resize(cropped_image, height=500)

    # new_image = cv2.medianBlur(new_image, 3)
    # new_image = cv2.GaussianBlur(new_image, (3, 3), 0)
    # ret, new_image = cv2.threshold(new_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # rect_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # new_image = cv2.dilate(new_image, rect_kern, iterations=2)

    # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    # plt.show()

    # path = 'F:/NUMBERPLATE/yolov4-custom-functions/data/images/training'
    # cv2.imwrite(os.path.join(path, str(i) + '.jpg'), new_image)
    # cv2.waitKey(0)
    # best = cv2.imread('C:/Users/foxtrot_delta/Downloads/2.jpg')
    # text = pytesseract.image_to_string(best, lang='ben')
    # print(text)

    reader = easyocr.Reader(['bn'])
    text = reader.readtext(new_image)
    print("<p>Plate number is: "+text[0][-2]+"</p>")

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
    exit()
    text = text[0][-2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
    color=(0, 255, 0), thickness=2, lineType=0)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    main()