# -*- coding: utf-8 -*-

import sys
from enum import Enum
from types import SimpleNamespace
from typing import List, Literal, get_args

import cv2
import numpy as np
from settings import get_settings

CONTOUR_LINEWIDTH = 2


class CV2ImRead(Enum):
    COLOR = cv2.IMREAD_COLOR
    GRAYSCALE = cv2.IMREAD_GRAYSCALE
    UNCHANGED = cv2.IMREAD_UNCHANGED


CV2ImReadValue = Literal[cv2.IMREAD_COLOR,
                         cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED]

assert set(get_args(CV2ImReadValue)) == {member.value for member in CV2ImRead}


class Image:
    def __init__(self):
        self.warning = False
        self.images = SimpleNamespace(current=None,
                                      original=None,
                                      output=None,
                                      blurred=None,
                                      masked=None,
                                      cropped=None)

    def load(self, filename, flag: CV2ImReadValue = 'COLOR'):
        self.images.original = cv2.imread(filename, CV2ImRead[flag].value)
        if self.images.original is None:
            print(f"ERROR: Incorrect image path {filename}")
            sys.exit(0)
        self.images.current = self.images.original.copy()

    # pre-processing of rgb image
    def rgb_processing(self, adjust=False):
        if adjust:
            print('adjusting gamma and saturation')
            self.images.current = self.images.original.copy()
            self.images.current = self._gammaCorrection(
                self.images.current, get_settings().gamma)
            self.images.current = self._saturationCorrection(
                self.images.current, get_settings().saturation)
        self._crop()
        self._blur()
        self._mask('masked',
                   get_settings().yellow[0],
                   get_settings().green[1])
        self._mask('masked_yellow',
                   get_settings().yellow[0],
                   get_settings().yellow[1])
        self._morph('masked', 'morphed')
        self._morph('masked_yellow', 'morphed_yellow')
        self._mask_original_image('masked')

    # pre-processing of luminescence image
    def lum_processing(self):
        self.images.equ = cv2.equalizeHist(self.images.current)
        self.images.current = self.images.equ.copy()

    def show(self, image: str = 'current'):
        """Show image"""
        cv2.namedWindow(image, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(image, getattr(self.images, image))
        height, width = getattr(self.images, image).shape[:2]
        cv2.resizeWindow(image, int(height / 2), int(width / 2))
        cv2.waitKey(0)

    def get_global_intensity(self):
        temp = self.images.current.ravel()
        mini = min(temp)
        # background is not always 0
        normalized = [elem if elem > mini else 0 for elem in temp]
        indices = np.nonzero(normalized)
        data = [normalized[i] for i in indices[0]]
        return (sum(data) / len(data))

    # draw enclosing circles around luminescence contours
    # returns image with marked contours and contour centers/areas/intensities
    def _get_lum_info(self, days, image: str = 'current'):
        img = getattr(self.images, image)
        # fully transparent rgb image
        lum_markings = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        # 127 found through trial and error, adaptive might do a better job
        void, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        areas = []
        for cnt in contours:
            if cv2.contourArea(cnt) < get_settings().noise_threshold_min:
                continue
            if cv2.contourArea(cnt) > get_settings().noise_threshold_max:
                self.warning = True
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (np.uint32(x), np.uint32(y))
            radius = np.uint32(radius)
            centers.append(center)
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = cv2.countNonZero(img[y:y+h, x:x+w])
            areas.append(area)
            cv2.circle(lum_markings, center, radius, (0, 0, 255), 2)
        return lum_markings, centers, areas

    # create overlay of enclosing circles and luminescence contours
    def create_overlay(self, days):
        overlay_img, centers, areas = self._get_lum_info(days)
        h, w = self.images.current.shape
        clr = np.asarray(get_settings().luminescence_bgr, dtype=np.uint8)
        base_img = np.zeros((h, w, 3), dtype=np.uint8)
        normalized = self.images.current / 255
        for i in range(3):
            base_img[:, :, i] = (clr[i] * normalized).astype(np.uint8)
        self.images.current = cv2.addWeighted(
            overlay_img, get_settings().luminescence_alpha, base_img, 1, 0)
        return centers, areas

    def grey(self):
        """Grey out region in output image not selected by morphed mask."""
        grey_bg = cv2.addWeighted(np.full_like(self.images.cropped, 255),
                                  0.4, self.images.cropped, 0.6, 0)
        black_fg = cv2.bitwise_and(grey_bg,
                                   grey_bg,
                                   mask=cv2.bitwise_not(self.images.morphed))
        plant_fg = cv2.bitwise_and(self.images.cropped,
                                   self.images.cropped,
                                   mask=self.images.morphed)
        plant_fg_grey_bg = cv2.add(plant_fg, black_fg)
        self.images.greyed = plant_fg_grey_bg.copy()
        self.images.output = self.images.greyed
        self.images.marked = self.images.greyed

    def _sort_clockwise(self, points):
        x = np.asarray([p[0] for p in points])
        y = np.asarray([p[1] for p in points])
        x0 = np.mean(x)
        y0 = np.mean(y)
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        angles = np.where((y-y0) > 0, np.arccos((x-x0)/r),
                          2*np.pi-np.arccos((x-x0)/r))
        mask = np.argsort(angles)
        x_sorted = x[mask]
        y_sorted = y[mask]
        return [[i[0], i[1]] for i in zip(x_sorted, y_sorted)]

    def _merge_contours(self, c1, c2):
        pts = [point[0] for point in c1] + [point[0] for point in c2]
        pts = self._sort_clockwise(pts)
        return np.array(pts).reshape((-1, 1, 2)).astype(np.int32)

    def _merge_contour_surplus(self, contours):
        circles = [cv2.minEnclosingCircle(c) for c in contours]
        for i, (c1, r1) in enumerate(circles):
            for j, (c2, r2) in enumerate(circles):
                if c1 == c2:
                    continue
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                if dist <= r1 + r2:  # TODO: test against max size
                    merged = self._merge_contours(contours[i], contours[j])
                    if (cv2.contourArea(merged) > get_settings().merged_threshold_max):
                        continue
                    contours.pop(max(i, j))
                    contours.pop(min(i, j))
                    contours.append(merged)
                    return self._merge_contour_surplus(contours)
        return contours

    def _find_contours(self):
        contoured = self.images.morphed.copy()
        contours, _ = cv2.findContours(contoured,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        self.images.contoured = contoured.copy()
        return [contour for contour in contours
                if cv2.contourArea(contour) > get_settings().threshold]

    def _draw_contour(self, contour):
        cv2.drawContours(self.images.contoured,
                         [contour], 0, (255, 255, 255), 3)
        cv2.drawContours(self.images.marked, [contour], 0, (0, 0, 0),
                         CONTOUR_LINEWIDTH * 3)
        cv2.drawContours(self.images.marked, [contour], 0, (255, 255, 255),
                         CONTOUR_LINEWIDTH)

    def sort_contours(self, contours):
        """Sort contours l-to-r t-to-b"""
        if len(contours) < get_settings().rows * get_settings().columns:
            moments = [cv2.moments(c) for c in contours]
            (contours, _) = zip(*sorted(zip(contours, moments),
                                        key=lambda m: (int(m[1]['m01'] / m[1]['m00'] / 100) * 100,
                                                       int(m[1]['m10'] / m[1]['m00'] / 100) * 100)))
            return contours
        # get centroids
        centroids = []
        for contour in contours:
            m = cv2.moments(contour)
            cX = int(m["m10"] / m["m00"])
            cY = int(m["m01"] / m["m00"])
            centroids.append((cX, cY, contour))
        # sort vertically
        centroids.sort(key=lambda k: k[1])
        # sort per row
        row_size = 4
        num_rows = int(len(contours) / row_size)
        sorted_centroids = []
        for i in range(num_rows):
            i *= row_size
            row = centroids[i:i+row_size]
            row.sort(key=lambda k: k[0])
            sorted_centroids.extend(row)
        sorted_contours = [c[2] for c in sorted_centroids]
        return sorted_contours

    def _has_intersections(self, contours):
        circles = [cv2.minEnclosingCircle(c) for c in contours]
        for i, (c1, r1) in enumerate(circles):
            for j, (c2, r2) in enumerate(circles):
                if c1 == c2:
                    continue
                dist = np.linalg.norm(np.array(c1) - np.array(c2))
                if dist <= r1 + r2:
                    return True
        return False

    # find and draw enclosing circles around all plants
    def find(self):
        """ """
        contours = self._find_contours()
        valid_contours = []
        color_areas = []

        if len(contours) > get_settings().rows * get_settings().columns:
            print('too many rosettes: restarting analysis')
            self.rgb_processing(True)
            self.grey()
            contours = self._find_contours()
        if (self._has_intersections(contours)):
            print('too many contours: merging')
            contours = self._merge_contour_surplus([c for c in contours])

        contours = self.sort_contours(contours)
        for i, contour in enumerate(contours):
            try:
                (cir_center_x,
                 cir_center_y), radius = cv2.minEnclosingCircle(contour)
            except ZeroDivisionError:
                continue
            valid_contours.append(contour)

            center = (int(cir_center_x), int(cir_center_y))
            cv2.circle(self.images.marked, center,
                       int(radius) + 5, (255, 0, 0), 4)

            x, y, w, h = cv2.boundingRect(contour)
            mask = np.zeros_like(self.images.masked_original)
            cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            img = cv2.bitwise_and(self.images.masked_original, mask)
            img = img[y:y + h, x:x + w]

            mask = np.zeros_like(self.images.masked)
            cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
            hsv_all = cv2.bitwise_and(self.images.masked, mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            eroded = cv2.morphologyEx(
                self.images.masked_yellow, cv2.MORPH_OPEN, kernel)
            eroded = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
            hsv_yellow = cv2.bitwise_and(eroded, mask)
            hsv_yellow = hsv_yellow[y:y + h, x:x + w]

            color_areas.append([i // get_settings().columns,
                                i % get_settings().columns,
                                cv2.countNonZero(hsv_all),
                                cv2.countNonZero(hsv_all) -
                                cv2.countNonZero(hsv_yellow),
                                cv2.countNonZero(hsv_yellow)])
        return ([cv2.minEnclosingCircle(contour) for contour in valid_contours], color_areas)

    # below methods are all image processing utilities

    def _gammaCorrection(self, img, gamma):
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(img, lookUpTable)

    def _saturationCorrection(self, img, factor):
        (h, s, v) = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        s *= factor
        s = np.clip(s, 0, 255)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def _meanBrightness(self, img):
        (h, s, v) = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        return v.mean()

    def _crop(self):
        """Crop image."""
        self.images.cropped = self.images.current[get_settings().crop_h_min:
                                                  get_settings().crop_h_max,
                                                  get_settings().crop_w_min:
                                                  get_settings().crop_w_max]
        temp = np.zeros(self.images.original.shape, dtype=np.uint8)
        temp[get_settings().crop_h_min:get_settings().crop_h_max,
             get_settings().crop_w_min:get_settings().crop_w_max] = self.images.cropped
        self.images.cropped = temp
        self.images.current = self.images.cropped.copy()

    def _blur(self):
        """Blur image."""
        if get_settings().blur % 2 == 0:
            get_settings().blur += 1
        self.images.blurred = cv2.medianBlur(self.images.current,
                                             get_settings().blur)
        self.images.current = self.images.blurred.copy()

    def _mask(self, mask_name: str, hsv_min: List, hsv_max: List):
        """Create mask using HSV range from blurred image."""
        hsv = cv2.cvtColor(self.images.blurred, cv2.COLOR_BGR2HSV)
        setattr(self.images, mask_name, cv2.inRange(hsv,
                                                    np.array(hsv_min),
                                                    np.array(hsv_max)))
        self.images.current = getattr(self.images, mask_name).copy()

    def _morph(self, mask_name: str, morph_name: str):
        """Process mask to try to make plants more coherent."""
        if get_settings().morph == 0:
            get_settings().morph = 1
        if get_settings().iterations == 0:
            get_settings().iterations = 1
        kernel = cv2.getStructuringElement(get_settings().kernel_type,
                                           (get_settings().morph,
                                            get_settings().morph))
        setattr(self.images,
                morph_name,
                cv2.morphologyEx(getattr(self.images, mask_name),
                                 get_settings().morph_type,
                                 kernel,
                                 iterations=get_settings().iterations))
        self.images.current = getattr(self.images, morph_name).copy()

    def _mask_original_image(self, mask_name):
        """Apply a mask to the original image, showing the regions selected."""
        result_name = mask_name + '_original'
        setattr(self.images, result_name, cv2.bitwise_and(
            self.images.cropped, self.images.cropped,
            mask=getattr(self.images, mask_name)))
        self.images.current = getattr(self.images, result_name).copy()
