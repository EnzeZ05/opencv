import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation = inter)

def get_points(contours):
    pts = np.asarray(contours, dtype = np.float32).reshape(4, 2)
    s = pts.sum(axis = 1)
    d = pts[:, 1] - pts[:, 0]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype = np.float32)

def p_transformation(image, contours):
    rect = get_points(contours)
    (t1, t2, d1, d2) = rect
    w1 = np.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2)
    w2 = np.sqrt((d1[0] - d2[0]) ** 2 + (d1[1] - d2[1]) ** 2)

    w = int(np.ceil(max(w1, w2)))

    h1 = np.sqrt((t1[0] - d2[0]) ** 2 + (t1[1] - d2[1]) ** 2)
    h2 = np.sqrt((t2[0] - d1[0]) ** 2 + (t2[1] - d1[1]) ** 2)

    h = int(np.ceil(max(h1, h2)))

    mat = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype = np.float32)

    dmat = cv2.getPerspectiveTransform(rect, mat)
    return cv2.warpPerspective(image, dmat, (w, h),
                               flags = cv2.INTER_CUBIC,
                               borderMode = cv2.BORDER_REPLICATE)

img = cv2.imread('images\\page.jpg')
r = img.shape[0] / 500.0
cpy = img.copy()

img = resize(cpy, height = 500)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(gray, 75, 200)

c = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
c = sorted(c, key = cv2.contourArea, reverse = True)[:5]

outer = c[0]
for s in c:
    peri = cv2.arcLength(s, True)
    approx = cv2.approxPolyDP(s, 0.02 * peri, True)

    if len(approx) == 4:
        outer = approx
        break

cv2.drawContours(img, [outer], -1, (0, 255, 0), 2)

warped = p_transformation(cpy, (outer.reshape(4, 2) * r).astype(np.float32))
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]

warped = cv2.medianBlur(warped, 3)
text = pytesseract.image_to_string(warped, lang='eng')
print(text)
