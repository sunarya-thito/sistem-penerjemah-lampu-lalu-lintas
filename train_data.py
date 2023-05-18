import json

import cv2
import numpy as np

import modul

histogram_lampu = {
    'merah': [],
    'kuning': [],
    'hijau': [],
}

histogram_lampu_sat = {
    'merah': [],
    'kuning': [],
    'hijau': [],
}

histogram_background_hue = []
histogram_background_sat = []

test_path = 'test.txt'
test_file = open(test_path, 'r')

# load test.txt as json
test_images = json.load(test_file)


def deteksi_warna(gambar):
    # to hsv
    img = modul.bgr_ke_hsv(gambar)
    modul.tampilkan(img, True)
    # hapus warna biru dan warna lain dengan saturasi rendah
    modul.filter_biru(img)
    modul.tampilkan(img, True)
    # to gray
    img = modul.hsv_ke_grayscale(img)
    modul.tampilkan(img)
    img = modul.m_dilasi_grayscale(img, 5)
    modul.tampilkan(img)
    img = modul.gaussian_blur_grayscale(img, 7)
    modul.tampilkan(img)
    # otus threshold
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    modul.tampilkan(img)
    # cari kontur
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # pertahankan kontur dengan area terbesar
    max_area = 0
    max_index = None
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > max_area:
            max_area = area
            max_index = i
    if max_index is not None:
        # hapus kontur lain
        for i in range(len(contours)):
            if i != max_index:
                cv2.drawContours(img, contours, i, (0, 0, 0), -1)
    modul.tampilkan(img)
    # gunakan hsv sebagai mask
    gambar = modul.operasi_and(gambar, img)
    modul.tampilkan(gambar)
    # menentukan apakah yang menyala lampu merah, kuning, atau hijau
    # berdasarkan histogram hue
    img = modul.bgr_ke_hsv(gambar)
    hue_hist = np.zeros(180, dtype=np.int32)
    sat_hist = np.zeros(256, dtype=np.int32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][2] <= 0:
                continue
            hue = img[i][j][0]
            sat = img[i][j][1]
            hue_hist[hue] += 1
            sat_hist[sat] += 1

    return hue_hist, sat_hist

total = 0
for gmb in test_images.items():
    # total+=1
    # if total > 40:
    #     break
    name = gmb[0]
    box = gmb[1]['box']
    image = cv2.imread('test/' + name)
    print('reading ', name)
    for b in box:
        x = round(b['x'])
        y = round(b['y'])
        w = round(b['w'])
        h = round(b['h'])
        crop = image[y:y + h, x:x + w].copy()
        color = b['warna']
        hue_hist, sat_hist = deteksi_warna(crop)
        histogram_lampu[color].append(hue_hist)
        histogram_lampu_sat[color].append(sat_hist)
        hist_bg_hue = np.zeros(180, dtype=np.int32)
        hist_bg_sat = np.zeros(256, dtype=np.int32)
        crop = modul.bgr_ke_hsv(crop)
        for i in range(crop.shape[0]):
            for j in range(crop.shape[1]):
                if crop[i][j][2] <= 0:
                    continue
                hue = crop[i][j][0]
                sat = crop[i][j][1]
                hist_bg_hue[hue] += 1
                hist_bg_sat[sat] += 1
        histogram_background_hue.append(hist_bg_hue)
        histogram_background_sat.append(hist_bg_sat)

# rata-rata dari histogram background hue
hbh = np.zeros(180, dtype=np.int32)
for i in range(len(histogram_background_hue)):
    hbh += histogram_background_hue[i]
hbh = hbh / len(histogram_background_hue)
# rata-rata dari histogram background sat
hbs = np.zeros(256, dtype=np.int32)
for i in range(len(histogram_background_sat)):
    hbs += histogram_background_sat[i]
hbs = hbs / len(histogram_background_sat)

for warna in histogram_lampu.items():
    nama = warna[0]
    hist_hue = warna[1]
    hist_sat = histogram_lampu_sat[nama]
    # dapatkan histogram rata-rata dari semua hist
    h = np.zeros(180, dtype=np.int32)
    for i in range(len(hist_hue)):
        h += hist_hue[i]
    h = h / len(hist_hue)

    h_s = np.zeros(256, dtype=np.int32)
    for i in range(len(hist_sat)):
        h_s += hist_sat[i]
    h_s = h_s / len(hist_sat)

    # kurangi histogram background dengan histogram lampu
    for i in range(len(hbh)):
        hbh[i] = max(hbh[i] - h[i], 0)

    for i in range(len(hbs)):
        hbs[i] = max(hbs[i] - h_s[i], 0)

    # tampilkan dalam plot
    import matplotlib.pyplot as plt

    plt.plot(h)
    plt.title(nama)
    plt.show()

# tampilkan dalam plot
import matplotlib.pyplot as plt

plt.plot(hbh)
plt.title('background hue')
plt.show()

plt.plot(hbs)
plt.title('background saturation')
plt.show()
