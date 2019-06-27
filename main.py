import cv2
import keras
import numpy as np
import sys
import math


def showimg(string, a):
    cv2.imshow(string, a)
    cv2.waitKey(0)


# Prvi i jedini argument komandne linije je indeks test primera
if len(sys.argv) != 2:
    print("Neispravno pozvan fajl, koristiti komandu \"python3 main.py X\" za pokretanje na test primeru sa brojem X")
    exit(0)

tp_idx = sys.argv[1]
img = cv2.imread('tests/{}.png'.format(tp_idx))
imgCopy = cv2.imread('tests/{}.png'.format(tp_idx))
#################################################################################
# Ucitavamo model
model = keras.models.load_model('model.h5')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
showimg('original', gray)

blurred = cv2.medianBlur(gray, 3)
showimg('blurred', blurred)

_, processing_img = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)
showimg('threshed', processing_img)

contours, _ = cv2.findContours(processing_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(processing_img, (x, y), (x + w, y + h), (255, 255, 255), -1)
showimg('filled', processing_img)

contours, _ = cv2.findContours(processing_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
clothes = np.array([])
classes = ['thsirt', 'trousers', 'pullover', 'dress', 'coat', 'sandals', 'shirt', 'sneakers', 'bag', 'ankle boot']
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w > h:
        x1 = x
        x2 = x + w
        y1 = round(y - (w - h) // 2)
        y2 = round(y + h + (w - h) // 2)
    else:
        x1 = round(x - (h - w) // 2)
        x2 = round(x + w + (h - w) // 2)
        y1 = y
        y2 = y + h
    resized_cloth = cv2.resize(blurred[y1:y2, x1:x2], (28, 28))
    # resized_cloth = cv2.cvtColor(resized_cloth, cv2.COLOR_BGR2GRAY)
    resized_cloth = cv2.bitwise_not(resized_cloth)
    resized_cloth = resized_cloth[np.newaxis, ..., np.newaxis]
    predictions = model.predict_classes(np.array(resized_cloth), batch_size=1, verbose=0)
    cv2.putText(img, classes[predictions[0]], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
showimg('final', img)
solution = img.copy()
#################################################################################

# Cuvamo resenje u izlazni fajl
cv2.imwrite("tests/{}_out.png".format(tp_idx), solution)