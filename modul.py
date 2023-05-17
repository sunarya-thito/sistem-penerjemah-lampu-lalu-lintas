import math

import cv2
import numpy as np

_cascade_path = 'cascade.xml'
cascade = None

gunakan_library = False

_cache_kernel_gaussian = {}


def muat_cascade():
    global cascade
    cascade = cv2.CascadeClassifier(_cascade_path)


def log(message):
    print(message)


# tidak digunakan di GUInya langsung
def tampilkan(gambar, as_hsv=False):
    # resize ke 800x600
    gambar = cv2.resize(gambar, (gambar.shape[1] * 600 // gambar.shape[0], 600))
    # tampilkan gambar
    if as_hsv:
        gambar = cv2.cvtColor(gambar, cv2.COLOR_HSV2BGR)
    cv2.imshow('gambar', gambar)
    cv2.waitKey(0)


def buat_gaussian_kernel(ukuran_kernel):
    cached = _cache_kernel_gaussian.get(ukuran_kernel)
    if cached is not None:
        return cached
    kernel = np.zeros((ukuran_kernel, ukuran_kernel))
    x = int(ukuran_kernel / 2)
    y = int(ukuran_kernel / 2)
    for i in range(-x, x + 1):
        for j in range(-y, y + 1):
            kernel[i + x][j + y] = 1 / (2 * math.pi * 1) * math.exp(-(i * i + j * j) / (2 * 1))
    _cache_kernel_gaussian[ukuran_kernel] = kernel
    return kernel


def gaussian_blur_grayscale(gambar, ukuran_kernel):
    if gunakan_library:
        return cv2.GaussianBlur(gambar, (ukuran_kernel, ukuran_kernel), 0)
    # tanpa library
    # calculate gaussian kernel
    kernel = buat_gaussian_kernel(ukuran_kernel)
    hasil = np.zeros(gambar.shape)

    kernel_w = (kernel.shape[0]) // 2
    kernel_h = (kernel.shape[1]) // 2

    h = gambar.shape[0]
    w = gambar.shape[1]

    for i in range(kernel_h, h - kernel_h):
        for j in range(kernel_w, w - kernel_w):
            sum = 0
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    sum += kernel[k][l] * gambar[i - kernel_h + k][j - kernel_w + l]
            hasil[i][j] = sum

    return hasil.astype(np.uint8)


def bgr_ke_grayscale(gambar):
    if gunakan_library:
        return cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    # tanpa library
    hasil = np.zeros((gambar.shape[0], gambar.shape[1]))
    for i in range(gambar.shape[0]):
        for j in range(gambar.shape[1]):
            hasil[i][j] = 0.299 * gambar[i][j][2] + 0.587 * gambar[i][j][1] + 0.114 * gambar[i][j][0]
    return hasil.astype(np.uint8)


def bgr_hsv(b, g, r):
    # ubah uint8 ke int
    b = int(b)
    g = int(g)
    r = int(r)
    v = max(r, g, b)
    if v == 0:
        s = 0
    else:
        s = (v - min(r, g, b)) / v

    h = 0
    if r == g == b:
        return 0, 0, v
    else:
        if v == r:
            h = 60 * (g - b) / (v - min(r, g, b))
        elif v == g:
            h = 120 + 60 * (b - r) / (v - min(r, g, b))
        elif v == b:
            h = 240 + 60 * (r - g) / (v - min(r, g, b))

    if h < 0:
        h += 360

    # convert to 0-180
    h /= 2

    return round(h), round(s * 255), round(v)


def hsv_bgr(h, s, v):
    h *= 2
    s /= 255
    v /= 255
    c = s * v
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    r, g, b = 0, 0, 0
    if 0 <= h < 60:
        r = c
        g = x
        b = 0
    elif 60 <= h < 120:
        r = x
        g = c
        b = 0
    elif 120 <= h < 180:
        r = 0
        g = c
        b = x
    elif 180 <= h < 240:
        r = 0
        g = x
        b = c
    elif 240 <= h < 300:
        r = x
        g = 0
        b = c
    elif 300 <= h < 360:
        r = c
        g = 0
        b = x

    r = (r + m) * 255
    g = (g + m) * 255
    b = (b + m) * 255

    return round(b), round(g), round(r)


def bgr_ke_hsv(gambar):
    if gunakan_library:
        return cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    # tanpa library
    hasil = np.zeros(gambar.shape)
    for i in range(gambar.shape[0]):
        for j in range(gambar.shape[1]):
            b, g, r = gambar[i][j]
            h, s, v = bgr_hsv(b, g, r)
            hasil[i][j] = [h, s, v]
    return hasil.astype(np.uint8)


def hsv_ke_bgr(gambar):
    if gunakan_library:
        return cv2.cvtColor(gambar, cv2.COLOR_HSV2BGR)
    # tanpa library
    hasil = np.zeros(gambar.shape)
    for i in range(gambar.shape[0]):
        for j in range(gambar.shape[1]):
            h, s, v = gambar[i][j]
            b, g, r = hsv_bgr(h, s, v)
            hasil[i][j] = [b, g, r]
    return hasil.astype(np.uint8)


def hsv_ke_grayscale(gambar):
    if gunakan_library:
        return cv2.cvtColor(cv2.cvtColor(gambar, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    # tanpa library
    hasil = np.zeros((gambar.shape[0], gambar.shape[1]))
    for i in range(gambar.shape[0]):
        for j in range(gambar.shape[1]):
            hasil[i][j] = gambar[i][j][2]
    return hasil.astype(np.uint8)


def crop_gambar(gambar, x, y, w, h):
    return gambar[y:y + h, x:x + w].copy()


def operasi_and(gambar, mask):
    if gunakan_library:
        return cv2.bitwise_and(gambar, gambar, mask=mask)
    # mask berupa 0 - 255
    hasil = np.zeros(gambar.shape)
    # gambar bisa berupa grayscale atau BGR
    # mask bisa berupa grayscale atau BGR
    # gambar dan mask bisa berbeda jenis
    for i in range(gambar.shape[0]):
        for j in range(gambar.shape[1]):
            if len(gambar.shape) == 3:
                if len(mask.shape) == 3:
                    hasil[i][j] = [gambar[i][j][0] & mask[i][j][0],
                                   gambar[i][j][1] & mask[i][j][1],
                                   gambar[i][j][2] & mask[i][j][2]]
                else:
                    hasil[i][j] = [gambar[i][j][0] & mask[i][j],
                                   gambar[i][j][1] & mask[i][j],
                                   gambar[i][j][2] & mask[i][j]]
            else:
                if len(mask.shape) == 3:
                    hasil[i][j] = [gambar[i][j] & mask[i][j][0],
                                   gambar[i][j] & mask[i][j][1],
                                   gambar[i][j] & mask[i][j][2]]
                else:
                    hasil[i][j] = gambar[i][j] & mask[i][j]

    return hasil.astype(np.uint8)


def histogram_equalization(gambar):
    if gunakan_library:
        return cv2.equalizeHist(gambar)
    # tanpa library
    hasil = np.zeros(gambar.shape)
    hist = np.zeros(256)
    for i in range(gambar.shape[0]):
        for j in range(gambar.shape[1]):
            hist[gambar[i][j]] += 1
    for i in range(1, 256):
        hist[i] += hist[i - 1]
    for i in range(gambar.shape[0]):
        for j in range(gambar.shape[1]):
            hasil[i][j] = 255 * hist[gambar[i][j]] / (gambar.shape[0] * gambar.shape[1])
    return hasil.astype(np.uint8)


def filter_biru(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hue = img[i][j][0]
            sat = img[i][j][1]
            if 100 < hue < 130 or sat < 150:
                img[i][j] = [0, 0, 0]
    return img


def m_dilasi_grayscale(img, ukuran_kernel):
    if gunakan_library:
        kernel = np.ones((ukuran_kernel, ukuran_kernel), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)
    # tanpa library
    hasil = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            maks = 0
            for k in range(-ukuran_kernel // 2, ukuran_kernel // 2 + 1):
                for l in range(-ukuran_kernel // 2, ukuran_kernel // 2 + 1):
                    if 0 <= i + k < img.shape[0] and 0 <= j + l < img.shape[1]:
                        maks = max(maks, img[i + k][j + l])
            hasil[i][j] = maks
    return hasil.astype(np.uint8)


if __name__ == '__main__':
    # check bgr_hsv dan hsv_bgr
    # untuk mengecek apakah fungsi bgr_hsv dan hsv_bgr sudah benar
    # hasilnya harus sama dengan inputannya
    input_bgr = [25, 25, 25]
    h, s, v = bgr_hsv(*input_bgr)
    print(input_bgr, [h, s, v])
    b, g, r = hsv_bgr(h, s, v)
    print(input_bgr, [b, g, r])

    # bandingkan hasilnya dengan fungsi bawaan opencv
    bgr = np.array([[input_bgr]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    print(bgr, hsv)
