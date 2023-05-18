import json
import os

import cv2

import detector
import modul

path = 'test'

modul.muat_cascade()

test_path = 'test.txt'
test_file = open(test_path, 'r')

# load test.txt as json
test_images = json.load(test_file)

result_path = 'result.txt'
result_file = open(result_path, 'w')


def is_collided(x, y, w, h, x1, y1, w1, h1):
    if x < x1 + w1 and x + w > x1 and y < y1 + h1 and y + h > y1:
        return True
    return False


overall_akurasi = 0
jumlah_overall_akurasi = 0
overall_terdeteksi = 0
jumlah_overall_terdeteksi = 0
overall_false_positive = 0
jumlah_overall_false_positive = 0
total = 0
result_file.write(
    'file, jumlah, tak_terdeteksi, false_positive, benar_deteksi, jumlah_terdeteksi, persentase_terdeteksi, persentase_false_positive, akurasi')
for file in os.listdir(path):
    if file.endswith('.jpg') or file.endswith('.png'):
        if file not in test_images:
            continue
        test_image = test_images.get(file)
        if test_image['box'] is None or len(test_image['box']) == 0:
            continue
        total += 1
        img = cv2.imread(path + '/' + file)
        _, hasil = detector.proses(img)

        benar_deteksi = 0
        false_positive = 0
        jumlah = len(test_image['box'])
        jumlah_terdeteksi = 0

        detected_boxes = []

        for h in hasil.items():
            coord = h[0]
            deteksi = h[1]

            x = coord[0]
            y = coord[1]
            w = coord[2]
            h = coord[3]

            apakah_salah_deteksi = False
            apakah_false_positive = True
            for test in test_image['box']:
                if test in detected_boxes:
                    continue
                x1 = test['x']
                y1 = test['y']
                w1 = test['w']
                h1 = test['h']
                warna = test['warna']
                # check apakah collision
                if is_collided(x, y, w, h, x1, y1, w1, h1):
                    detected_boxes.append(test)
                    apakah_false_positive = False
                    if warna != deteksi:
                        apakah_salah_deteksi = True
                    break

            if apakah_false_positive:
                false_positive += 1
            elif not apakah_salah_deteksi:
                benar_deteksi += 1
            jumlah_terdeteksi += 1
        jumlah_overall_terdeteksi += jumlah_terdeteksi
        jumlah_tak_terdeteksi = len(test_image['box']) - len(detected_boxes)

        jumlah_sebenar_terdeteksi = jumlah_terdeteksi - false_positive

        akurasi = None if jumlah_sebenar_terdeteksi == 0 else benar_deteksi / jumlah_sebenar_terdeteksi * 100

        persentase_false_positive = None if jumlah_terdeteksi == 0 else false_positive / jumlah_terdeteksi * 100

        persentase_terdeteksi = None if jumlah == 0 else jumlah_sebenar_terdeteksi / jumlah * 100

        result_file.write(file + ', ' + str(jumlah) + ', ' + str(jumlah_tak_terdeteksi) + ', ' + str(false_positive) +
                          ', ' + str(benar_deteksi) + ', ' + str(jumlah_terdeteksi) + ', ' + str(persentase_terdeteksi)
                          + ', ' + str(persentase_false_positive) + ', ' + str(akurasi))
        result_file.write('\n')

        print('file: ' + file + ', jumlah: ' + str(jumlah) + ', tak_terdeteksi: ' + str(jumlah_tak_terdeteksi) +
                ', false_positive: ' + str(false_positive) + ', benar_deteksi: ' + str(benar_deteksi) +
                ', jumlah_terdeteksi: ' + str(jumlah_terdeteksi) + ', persentase_terdeteksi: ' +
                str(persentase_terdeteksi) + ', persentase_false_positive: ' + str(persentase_false_positive) +
                ', akurasi: ' + str(akurasi))

        if jumlah_sebenar_terdeteksi > 0:
            overall_akurasi += akurasi
            jumlah_overall_akurasi += 1

        if jumlah_terdeteksi > 0:
            overall_false_positive += persentase_false_positive
            jumlah_overall_false_positive += 1

        if jumlah > 0:
            overall_terdeteksi += persentase_terdeteksi
            jumlah_terdeteksi += 1

print('Total test: ' + str(total))
print('Overall akurasi: ' + str(overall_akurasi / jumlah_overall_akurasi))
print('Overall false positive: ' + str(overall_false_positive / jumlah_overall_false_positive))
print('Overall terdeteksi: ' + str(overall_terdeteksi / jumlah_overall_terdeteksi))

result_file.write('Total test: ' + str(total) + '\n')
result_file.write('Akurasi Deteksi Warna: ' + str(overall_akurasi / jumlah_overall_akurasi) + '\n')
result_file.write('False Positive Haarcascade: ' + str(overall_false_positive / jumlah_overall_false_positive) + '\n')
result_file.write('Akurasi Object Detection Haarcascade: ' + str(overall_terdeteksi / jumlah_overall_terdeteksi) + '\n')

result_file.close()
