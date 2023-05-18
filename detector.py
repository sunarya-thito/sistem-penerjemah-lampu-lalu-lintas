import os

import cv2
import numpy as np

import modul



def proses_cari_lokasi(controller, image):
    return [
        InputImage(controller, image),
        RGBKeGrayscale(controller),
        HistogramEqualization(controller),
        GaussianBlur(controller, 7),
        HaarCascadeDetection(controller),
        PerlihatkanHasil(controller),
    ]


def proses_deteksi_warna(hasil, controller, image):
    return [
        InputHasilDeteksi(controller, hasil, image),
        FilterBiruDanSaturasiRendah(controller),
        BGRKeGrayscale(controller),
        OtsuThresholding(controller, 150, 255),
        Dilasi(controller, 5),
        HapusAreaKonturKecil(controller),
        OperasiAnd(controller),
        KalkulasiHistogramHue(controller),
    ]


class Process:
    def __init__(self, controller):
        self.image = None
        self.controller = controller
        self.previous_process = None

    def execute_process(self, previous_process, tampilkan_gambar):
        self.previous_process = previous_process
        self.process()
        tampilkan_gambar(self.image)

    def process(self):
        if self.previous_process is not None and self.previous_process.image is not None:
            self.image = self.previous_process.image.copy()

    def cari_proses(self, tipe_proses):
        proses = self
        while proses is not None:
            if isinstance(proses, tipe_proses):
                return proses
            proses = proses.previous_process
        return None

    def to_string(self):
        return self.__class__.__name__


class InputImage(Process):
    def __init__(self, controller, image):
        super().__init__(controller)
        self.image = image.copy()

    def to_string(self):
        # return Input Image (width, height)
        return 'Input Image ({}, {})'.format(self.image.shape[1], self.image.shape[0])


class RGBKeGrayscale(Process):
    def process(self):
        self.image = modul.bgr_ke_grayscale(self.previous_process.image)

    def to_string(self):
        return 'RGB ke Grayscale'


class GaussianBlur(Process):

    def __init__(self, controller, ukuran_kernel):
        super().__init__(controller)
        self.ukuran_kernel = ukuran_kernel

    def process(self):
        self.image = modul.gaussian_blur_grayscale(self.previous_process.image, self.ukuran_kernel)

    def to_string(self):
        return 'Gaussian Blur ({}x{})'.format(self.ukuran_kernel, self.ukuran_kernel)


class HasilDeteksi:
    def __init__(self, x, y, w, h, hasil, controller):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.hasil = hasil
        self.controller = controller


class HaarCascadeDetection(Process):
    def __init__(self, controller):
        super().__init__(controller)
        self.hasil_deteksi = []

    def process(self):
        rects = modul.cascade.detectMultiScale(self.previous_process.image, scaleFactor=1.1, minNeighbors=5,
                                               minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        gambar = self.controller.image.copy()
        gambar_modif = gambar.copy()
        for (x, y, w, h) in rects:
            modul.log('deteksi di {},{} dengan ukuran {}x{}'.format(x, y, w, h))
            crop = modul.crop_gambar(gambar, x, y, w, h)
            from controller import DetectController
            detection = DetectController(self.controller, crop, '-')
            det = HasilDeteksi(x, y, w, h, None, detection)
            self.controller.consoleController.add_detection(detection)
            self.hasil_deteksi.append(det)
            subprocess = proses_deteksi_warna(det, self.controller, crop)

            def tampilkan_gambar(img, det):
                if img is None:
                    return
                det.controller.set_image(img)
                # update gambar yang dari crop
                # check apakah gambar grayscale atau tidak
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # resize img ke ukuran deteksi
                img = cv2.resize(img, (det.w, det.h))
                gambar_modif[det.y:det.y + det.h, det.x:det.x + det.w] = img
                self.controller.set_image(gambar_modif)

            self.controller.push_process(subprocess, lambda img, det=det: tampilkan_gambar(img, det))
            cv2.rectangle(gambar_modif, (x, y), (x + w, y + h), (255, 0, 255), 2)

        self.image = gambar_modif.copy()

    def to_string(self):
        return 'Haar Cascade Detection'


class PerlihatkanHasil(Process):
    def process(self):
        self.image = self.controller.image.copy()
        cascade = self.cari_proses(HaarCascadeDetection)
        for hasil_deteksi in cascade.hasil_deteksi:
            hasil = hasil_deteksi.hasil
            x = hasil_deteksi.x
            y = hasil_deteksi.y
            w = hasil_deteksi.w
            h = hasil_deteksi.h
            color = (255, 0, 0)
            if hasil == 'merah':
                color = (0, 0, 255)
            elif hasil == 'kuning':
                color = (0, 255, 255)
            elif hasil == 'hijau':
                color = (0, 255, 0)
            if hasil is None:
                hasil = 'Tidak terdeteksi'
            cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(self.image, hasil, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # save hasil ke .txt
        # buka save dialog menggunakan library qt
        from PyQt5.QtWidgets import QFileDialog
        filename, _ = QFileDialog.getSaveFileName(self.controller.view, 'Save File', 'hasil', 'Text files (*.txt)')
        if filename:
            with open(filename, 'w') as f:
                f.write('x, y, width, height, hasil\n')
                for hasil_deteksi in cascade.hasil_deteksi:
                    hasil = hasil_deteksi.hasil
                    if hasil is None:
                        hasil = 'Tidak terdeteksi'
                    x = hasil_deteksi.x
                    y = hasil_deteksi.y
                    w = hasil_deteksi.w
                    h = hasil_deteksi.h
                    f.write('{}, {}, {}, {}, {}\n'.format(x, y, w, h, hasil))

    def to_string(self):
        return 'Perlihatkan Hasil'


class InputHasilDeteksi(InputImage):
    def __init__(self, controller, hasil, image):
        super().__init__(controller, image)
        self.hasil = hasil

    def to_string(self):
        return 'Input Hasil Deteksi (x: {}, y: {}, width: {}, height: {})'.format(self.hasil.x, self.hasil.y, self.hasil.w, self.hasil.h)


class FilterBiruDanSaturasiRendah(Process):
    def process(self):
        image = self.previous_process.image.copy()
        image = modul.bgr_ke_hsv(image)
        image = modul.filter_biru(image)
        self.image = modul.hsv_ke_bgr(image)

    def to_string(self):
        return 'Filter Biru dan Saturasi Rendah'


class BGRKeGrayscale(Process):
    def process(self):
        self.image = modul.bgr_ke_grayscale(self.previous_process.image)

    def to_string(self):
        return 'BGR ke Grayscale'


class Dilasi(Process):
    def __init__(self, controller, ukuran_kernel):
        super().__init__(controller)
        self.ukuran_kernel = ukuran_kernel

    def process(self):
        self.image = modul.m_dilasi_grayscale(self.previous_process.image, self.ukuran_kernel)

    def to_string(self):
        return 'Dilasi (ukuran kernel: {})'.format(self.ukuran_kernel)


class OtsuThresholding(Process):

    def __init__(self, controller, lower, upper):
        super().__init__(controller)
        self.lower = lower
        self.upper = upper

    def process(self):
        _, self.image = cv2.threshold(self.previous_process.image, self.lower, self.upper, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def to_string(self):
        return 'Otsu Thresholding (min: {}, max: {})'.format(self.lower, self.upper)


class HapusAreaKonturKecil(Process):
    def process(self):
        max_area = 0
        max_index = None
        contours, _ = cv2.findContours(self.previous_process.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img = self.previous_process.image.copy()
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
        self.image = img

    def to_string(self):
        return 'Hapus Area Kontur Kecil'


class HistogramEqualization(Process):
    def process(self):
        self.image = modul.histogram_equalization(self.previous_process.image)

    def to_string(self):
        return 'Histogram Equalization'


class OperasiAnd(Process):
    def process(self):
        mask = self.cari_proses(HapusAreaKonturKecil).image
        gambar = self.cari_proses(InputImage).image
        self.image = modul.operasi_and(gambar, mask)

    def to_string(self):
        return 'Operasi AND'


class KalkulasiHistogramHue(Process):
    def buat_histogram_hue(self, hue_hist):
        width = 200
        height = 200
        img = np.zeros((height, width, 3), np.uint8)
        max_value = max(hue_hist)
        # buat background HUE warna-warni tapi tidak terlalu terang
        for i in range(180):
            hue = i
            # ubah hue ke rgb
            color = modul.hsv_ke_bgr(np.array([[[hue, 255, 255]]], dtype=np.uint8))[0][0]
            # kurangi brightness - 50, jangan lupa di clamp
            color = tuple([int(max(0, x - 200)) for x in color])
            # buat garis
            x1 = int(i * width / 180)
            y1 = 0
            y2 = height
            cv2.line(img, (x1, y1), (x1, y2), color, 2)
        for i in range(180):
            hue = i
            value = hue_hist[i]
            # ubah hue ke rgb
            color = modul.hsv_ke_bgr(np.array([[[hue, 255, 255]]], dtype=np.uint8))[0][0]
            color = tuple([int(x) for x in color])
            # buat garis
            x1 = int(i * width / 180)
            y1 = int(height - value * height / max_value)
            y2 = height
            cv2.line(img, (x1, y1), (x1, y2), color, 2)
        return img


    def process(self):
        img = self.previous_process.image.copy()
        img = modul.bgr_ke_hsv(img)
        hue_hist = np.zeros(180, dtype=np.int32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j][2] <= 0:
                    continue
                hue = img[i][j][0]
                hue_hist[hue] += 1
        avg_merah = 0
        avg_kuning = 0
        avg_hijau = 0
        for i in range(0, 10):
            avg_merah += hue_hist[i]
        for i in range(170, 180):
            avg_merah += hue_hist[i]
        for i in range(10, 40):
            avg_kuning += hue_hist[i]
        for i in range(40, 110):
            avg_hijau += hue_hist[i]
        avg_merah /= 20
        avg_kuning /= 30
        avg_hijau /= 70
        hasil = self.cari_proses(InputHasilDeteksi).hasil
        if avg_merah > avg_kuning and avg_merah > avg_hijau:
            hasil.hasil = 'merah'
        elif avg_kuning > avg_merah and avg_kuning > avg_hijau:
            hasil.hasil = 'kuning'
        elif avg_hijau > avg_merah and avg_hijau > avg_kuning:
            hasil.hasil = 'hijau'
        else:
            hasil.hasil = '?'
        hasil.controller.view.label_2.setText(hasil.hasil + ' (x: ' + str(hasil.x) + ', y: ' + str(hasil.y) + ', w: ' + str(hasil.w) + ', h: ' + str(hasil.h) + ')')
        self.image = self.buat_histogram_hue(hue_hist)

    def to_string(self):
        return 'Kalkulasi Histogram Hue'



def proses(gambar):
    if not modul.gunakan_library:
        modul.log('Proses berjalan tanpa bantuan library! Proses akan lebih lambat dibanding menggunakan library.')
    if modul.cascade is None:
        raise Exception('cascade belum diinisialisasi')
    modul.log('proses gambar...')
    modul.log('mengubah dari BGR ke Grayscale')
    # ubah ke grayscale
    gray = modul.bgr_ke_grayscale(gambar)
    modul.tampilkan(gray)
    modul.log('menerapkan gaussian blur')
    # gaussian
    gray = modul.gaussian_blur_grayscale(gray, 5)
    modul.tampilkan(gray)
    modul.log('mendeteksi menggunakan cascade')
    modul.log(gray)
    # deteksi menggunakan cascade
    rects = modul.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    modul.log('didapatkan {} deteksi'.format(len(rects)))
    hasil_deteksi = {}
    # loop seluruh deteksi
    for (x, y, w, h) in rects:
        modul.log('deteksi di {},{} dengan ukuran {}x{}'.format(x, y, w, h))
        # gambar kotak di area deteksi
        crop = modul.crop_gambar(gambar, x, y, w, h)
        # crop gambar sesuai area deteksi
        hasil = deteksi(crop)
        modul.log('hasil deteksi: {}'.format(hasil))
        hasil_deteksi[(x, y, w, h)] = hasil
    for (x, y, w, h), hasil in hasil_deteksi.items():
        color = (255, 0, 0)
        if hasil == 'merah':
            color = (0, 0, 255)
        elif hasil == 'kuning':
            color = (0, 255, 255)
        elif hasil == 'hijau':
            color = (0, 255, 0)
        cv2.rectangle(gambar, (x, y), (x + w, y + h), color, 2)
        # tulis teks hasil deteksi
        cv2.putText(gambar, hasil, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return gambar, hasil_deteksi


def deteksi(gambar):
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
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][2] <= 0:
                continue
            hue = img[i][j][0]
            hue_hist[hue] += 1
    # cari hue dengan frekuensi terbesar
    avg_merah = 0
    avg_kuning = 0
    avg_hijau = 0
    for i in range(0, 10):
        avg_merah += hue_hist[i]
    for i in range(170, 180):
        avg_merah += hue_hist[i]
    for i in range(10, 40):
        avg_kuning += hue_hist[i]
    for i in range(40, 110):
        avg_hijau += hue_hist[i]
    avg_merah /= 20
    avg_kuning /= 30
    avg_hijau /= 70
    if avg_merah > avg_kuning and avg_merah > avg_hijau:
        return 'merah'
    elif avg_kuning > avg_merah and avg_kuning > avg_hijau:
        return 'kuning'
    elif avg_hijau > avg_merah and avg_hijau > avg_kuning:
        return 'hijau'
    else:
        return 'tidak terdeteksi'


if __name__ == '__main__':
    modul.muat_cascade()
    # daftar semua gambar di folder test
    path = os.listdir('test')
    for i in range(len(path)):
        gambar = cv2.imread('test/' + path[i])
        gambar, _ = proses(gambar)
        modul.tampilkan(gambar)
    cv2.destroyAllWindows()
