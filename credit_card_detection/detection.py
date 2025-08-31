import cv2
import imutils
import numpy as np

def show(ref):
    cv2.imshow('image', ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('images\\ocr_a_reference.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

c, h = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
res = cv2.drawContours(img, c, -1, (0, 0, 255), 3)
# show(img)

c = sorted(c, key = lambda elt: cv2.boundingRect(elt)[0])
mp = {}

for i, elt in enumerate(c):
    x, y, w, h = cv2.boundingRect(elt)
    bound = gray[y: y + h, x: x + w]
    bound = cv2.resize(bound, (57, 88))
    mp[i] = bound

ker1 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 9))

s = cv2.imread('images\\credit_card_01.png')
s = imutils.resize(s, width = 300)
gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, ker1)
gx = cv2.Sobel(tophat, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)

gx = np.absolute(gx)
mn, mx = np.min(gx), np.max(gx)

gx = (255 * ((gx - mn) / (mx - mn)))
gx = gx.astype("uint8")

gx = cv2.morphologyEx(gx, cv2.MORPH_CLOSE, ker1)
# show(gx)

thresh = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# show(thresh)

thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, ker1)
t = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

cpy = s.copy()
cv2.drawContours(cpy, t, -1, (0, 0, 255), 2)
show(cpy)

pts = []
for (i, c) in enumerate(t):
    x, y, w, h = cv2.boundingRect(c)

    r = w / float(h)
    if r > 2.5 and r < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            pts.append((x, y, w, h))

pts = sorted(pts, key = lambda x: x[0])

digits = []
for (i, (x, y, w, h)) in enumerate(pts):
    group = gray[y - 5: y + h + 5, x - 5: x + w + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    _ = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    _ = sorted(_, key = lambda e: cv2.boundingRect(e)[0])

    for e in _:
        dx, dy, dw, dh = cv2.boundingRect(e)
        roi = group[dy: dy + dh, dx: dx + dw]
        roi = cv2.resize(roi, (57, 88))
        digits.append(roi)

s = ''

for (id, a) in enumerate(digits):
    mn = 1
    num = 0
    # show(a)
    for (i, b) in mp.items():
        res = cv2.matchTemplate(a, b, cv2.TM_SQDIFF_NORMED)
        # print(res)
        if float(res[0, 0]) < mn:
            mn = float(res[0, 0])
            num = i

    if id % 4 == 0 and id > 0:
        s = s + ' '
    s = s + str(num)

print(s)

