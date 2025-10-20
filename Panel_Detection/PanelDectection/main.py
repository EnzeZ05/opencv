import re
import pytesseract
import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def show(ref):
    cv2.imshow('image', ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_points(contours):
    pts = np.asarray(contours, dtype = np.float32).reshape(4, 2)
    s = pts.sum(axis = 1)
    d = pts[:, 1] - pts[:, 0]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype = np.float32)

def resize(quad):
    tl, tr, br, bl = quad
    W = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    H = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    W = max(W, 1); H = max(H, 1)
    return W, H

def get_mask(img, team = 'blue'):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
    return gray

def extend_min_area_rect(rect, extra_ratio=0.5):
    (cx, cy), (w, h), ang = rect

    if w >= h:
        w_new = w * (1 + 2 * extra_ratio)
        h_new = h
    else:
        w_new = w
        h_new = h * (1 + 2 * extra_ratio)

    rect_ext = ((cx, cy), (w_new, h_new), ang)
    box_ext = cv2.boxPoints(rect_ext).astype(np.float32)
    return rect_ext, box_ext

def tesseract_digit(roi_bgr):
    import re, pytesseract

    if roi_bgr is None or roi_bgr.size == 0:
        return None, -1.0, "empty_roi", roi_bgr
    h0, w0 = roi_bgr.shape[:2]
    if h0 < 3 or w0 < 3:
        return None, -1.0, "roi_too_small", roi_bgr

    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    target_h = 160.0
    scale = target_h / float(h0)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    g = cv2.medianBlur(g, 3)

    adp = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
    adpi = cv2.bitwise_not(adp)
    cands = [("gray", g), ("gray_inv", cv2.bitwise_not(g)), ("adp", adp), ("adp_inv", adpi)]

    cfg = ("--oem 1 --psm 10 -l eng "
           "-c tessedit_char_whitelist=0123456789 "
           "-c load_system_dawg=0 -c load_freq_dawg=0 "
           "-c user_defined_dpi=300")

    best = (None, -1.0, "none", g)
    for name, img in cands:
        try:
            data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
            for t, c in zip(data["text"], data["conf"]):
                t = (t or "").strip()
                try: c = float(c)
                except: c = -1.0
                if t.isdigit() and c > best[1]:
                    best = (t[0], c, name, img)
        except Exception:
            pass
        if best[0] is None:
            s = pytesseract.image_to_string(img, config=cfg).strip()
            m = re.findall(r"\d", s)
            if m:
                best = (m[0], max(best[1], 30.0), name, img)
    return best

# cap = cv2.VideoCapture("move.mp4")
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

best = {
    "conf": -1.0,
    "digit": None,
    "Hinv": None,
    "rect_patch_full": None,
    "ij": None
}

img = cv2.imread("issue_red_001.jpg")
last = img.copy()

mask = get_mask(img, team = 'blue')
c, h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
res = cv2.drawContours(img, c, -1, (0, 0, 255), 2)
c = sorted(c, key = lambda elt: cv2.boundingRect(elt)[0])

for (i, a) in enumerate(c):
    if cv2.contourArea(a) < 100:
        continue

    for(j, b) in enumerate(c):
        if cv2.contourArea(b) < 100 or j <= i:
            continue

        rectA = cv2.minAreaRect(a)
        rectB = cv2.minAreaRect(b)
        boxA = cv2.boxPoints(rectA).astype(np.float32)
        boxB = cv2.boxPoints(rectB).astype(np.float32)

        _, boxA_ext = extend_min_area_rect(rectA, extra_ratio = 0.5)
        _, boxB_ext = extend_min_area_rect(rectB, extra_ratio = 0.5)

        pts_all = np.vstack([boxA_ext, boxB_ext]).astype(np.float32)
        hull = cv2.convexHull(pts_all.reshape(-1, 1, 2))
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        src_quad = approx.reshape(4, 2).astype(np.float32)
        quad = get_points(src_quad)

        Wp, Hp = resize(quad)
        mat = np.array([[0, 0], [Wp - 1, 0], [Wp - 1, Hp - 1], [0, Hp - 1]], np.float32)

        dmat = cv2.getPerspectiveTransform(quad, mat)

        try:
            inv = np.linalg.inv(dmat)
        except (ValueError, KeyError) as e:
            continue

        A = cv2.perspectiveTransform(boxA.reshape(-1, 1, 2), dmat).reshape(-1, 2)
        B = cv2.perspectiveTransform(boxB.reshape(-1, 1, 2), dmat).reshape(-1, 2)

        patch = cv2.warpPerspective(img, dmat, (int(Wp), int(Hp)), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)

        A_r = cv2.perspectiveTransform(boxA.reshape(-1, 1, 2), dmat).reshape(-1, 2)
        B_r = cv2.perspectiveTransform(boxB.reshape(-1, 1, 2), dmat).reshape(-1, 2)

        vis = patch.copy()
        xA, yA, wA, hA = cv2.boundingRect(np.intp(A))
        xB, yB, wB, hB = cv2.boundingRect(np.intp(B))

        # cv2.polylines(vis, [np.int32(A_r)], True, (0, 255, 0), 2)
        # cv2.polylines(vis, [np.int32(B_r)], True, (0, 255, 0), 2)
        # show(vis)

        H, W = patch.shape[:2]

        if xA <= xB:
            L = (xA, yA, wA, hA)
            R = (xB, yB, wB, hB)
        else:
            L = (xB, yB, wB, hB)
            R = (xA, yA, wA, hA)

        pad_x = max(2, int(0.000001 * W))
        x1 = max(0, L[0] + L[2] - pad_x)
        x2 = min(W, R[0] + pad_x)
        y1, y2 = 0, H

        dx1, dx2, dy1, dy2 = x1, x2, y1, y2

        min_w = max(12, int(0.03 * W))
        keep_ratio = 0.60

        w = x2 - x1
        if w < min_w:
            continue

        cx = (x1 + x2) // 2
        w_keep = max(min_w, int(w * keep_ratio))
        x1 = int(max(0, cx - w_keep // 2))
        x2 = int(min(W, x1 + w_keep))

        if x2 > x1:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            digit_roi = patch[y1 : y2, x1 : x2].copy()
            digit, conf, used, dbg = tesseract_digit(digit_roi)
            # print(digit, conf)

            if digit is not None and conf > best["conf"]:
                try:
                    Hinv = np.linalg.inv(dmat)
                except np.linalg.LinAlgError:
                    Hinv = None

                best.update(
                    conf = float(conf),
                    digit = str(digit),
                    Hinv = Hinv,
                    rect_patch_full = np.array([[dx1, dy1], [dx2, dy1], [dx2, dy2], [dx1, dy2]], dtype=np.float32),
                    ij = (i, j),
                )

if best["digit"] is not None and best["Hinv"] is not None and best["rect_patch_full"] is not None:
    rect_img = cv2.perspectiveTransform(best["rect_patch_full"][None, :, :], best["Hinv"])[0]
    img_poly = last.copy()

    poly = np.int32(np.round(rect_img)).reshape(-1, 1, 2)
    cv2.polylines(img_poly, [poly], True, (0, 0, 255), 2)

    cv2.putText(img_poly, f"{best['digit']} ({best['conf']:.0f})",
                tuple(poly[0,0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    show(img_poly)
    print(f"BEST @ pair {best['ij']}: digit={best['digit']}, conf={best['conf']:.1f}")
else:
    print("No valid OCR hit found.")


