# -*- coding: utf-8 -*-

from types import SimpleNamespace

import numpy as np
import cv2
import csv
import math
import os
import sys

from image import Image
import argparse
from settings import get_settings


class PlantAnalyze:
    def __init__(self, **kwargs):
        self.args = SimpleNamespace(**kwargs)

    def _detection_input(self):
        if self.args.env:
            try:
                get_settings(env_file=self.args.env)
            except:
                print('error retrieving settings')
        if self.args.verbose:
            get_settings().model_dump()

    # read 3x3 matrix from a string
    def _unpack_calibration_matrix(self, string):
        lst = string.split('\n')
        row1 = lst[0][2:-1]
        row2 = lst[1][2:-1]
        row3 = lst[2][2:-2]
        return np.array([np.array(row1.split(), dtype=np.float32),
                        np.array(row2.split(), dtype=np.float32),
                        np.array(row3.split(), dtype=np.float32)])

    # display overlay of rgb and luminescence images
    def _get_overlay(self):
        overlay = cv2.addWeighted(self.img_lum.images.current, get_settings().luminescence_alpha,
                                  self.img_rgb.images.marked, 0.8, 0)
        return overlay

    # count number of luminscence contours and sum of contour areas inside enclosing circle of each plant
    def _find_stats_per_plant(self, plants, contours, areas):
        def in_range(x1, y1, x2, y2, r):
            return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2)) <= r
        contour_per_plant = np.zeros(len(plants), dtype=int)
        area_per_plant = np.zeros(len(plants), dtype=int)
        for (plant, i) in zip(plants, range(len(plants))):
            for (contour, area) in zip(contours, areas):
                if in_range(plant[0][0], plant[0][1], contour[0], contour[1], plant[1]):
                    contour_per_plant[i] += 1
                    area_per_plant[i] += area
        return contour_per_plant, area_per_plant

    def _unique_csv_filename(self):
        if not os.path.exists(get_settings().csv_dir):
            print('creating csv directory')
            os.makedirs(get_settings().csv_dir)
        name = get_settings().csv_dir + 'results_'
        name += self.args.rgb_file[self.args.rgb_file.find(
            '/') + 1:self.args.rgb_file.find('RGB')]
        filenumber = 0
        while os.path.isfile(name + str(filenumber).zfill(3) + '.csv'):
            filenumber += 1
        return name + str(filenumber).zfill(3) + '.csv'

    # generate csv file with analysis data
    def generate_csv(self, days, data):
        print('writing csv data')
        last_column = 'no. hydathodes' if days == 7 else 'luminescence area'
        with open(self._unique_csv_filename(), 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(['row', 'column', 'rosette area', 'green area',
                                'yellow area', last_column, 'global intensity', 'warning'])
            for row in data:
                csv_writer.writerow(row)

    # analyze the green and yellow objects in the rgb image
    def analyze(self, days):
        print('analyzing images')
        self._detection_input()
        self.img_rgb = Image()
        self.img_rgb.load(self.args.rgb_file)

        self.img_lum = Image()
        self.img_lum.load(self.args.lum_file, 'GRAYSCALE')

        self.img_rgb.rgb_processing()
        self.img_rgb.grey()
        plant_contours, color_areas = self.img_rgb.find()

        avg_intensity = self.img_lum.get_global_intensity()
        self.img_lum.lum_processing()
        m = self._unpack_calibration_matrix(get_settings().calibration)
        self.img_lum.images.current = cv2.warpPerspective(self.img_lum.images.current, m,
                                                          (self.img_rgb.images.original.shape[1],
                                                           (self.img_rgb.images.original.shape[0])))
        lum_contours, lum_areas = self.img_lum.create_overlay(days)
        num_contours, total_areas = self._find_stats_per_plant(
            plant_contours, lum_contours, lum_areas)
        if days == 7:
            for (entry, count) in zip(color_areas, num_contours):
                entry.append(count)
                entry.append('n/a')
                entry.append(self.img_lum.warning)
        else:
            for (entry, area) in zip(color_areas, total_areas):
                entry.append(area)
                entry.append(avg_intensity)
                entry.append(self.img_lum.warning)
        debug_img = self._get_overlay()
        debug_filename = self._unique_csv_filename(
        )[:-3].replace('results', 'debug') + 'png'
        cv2.imwrite(debug_filename, debug_img)
        return color_areas


if __name__ == "__main__":
    if len(sys.argv) == 2:
        env_file = sys.argv[1]
    else:
        env_file = '.env'
    input_dir = get_settings(env_file).input_dir
    rgb_files = [file for file in os.listdir(input_dir) if 'RGB' in file]
    for rgb in rgb_files:
        es = rgb.replace('RGB', 'ES')
        days = 7 if rgb[:2] == '07' else 14
        pa = PlantAnalyze(rgb_file=input_dir + rgb,
                          lum_file=input_dir + es,
                          env=env_file,
                          verbose=False)
        data = pa.analyze(days)
        pa.generate_csv(days, data)
