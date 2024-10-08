# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np


class Calibrator(object):
    """Class to find desired transformation matrix given calibration images"""

    GRID_SIZE = (4, 11)

    def __init__(self, rgb, es):
        self.rgb = cv2.imread(rgb)
        self.es = cv2.imread(es)
        if self.rgb is None or self.es is None:
            print('cannot open calibration image', file=sys.stderr)
            sys.exit(1)

    def _findCorners(self, points):
        # sort by row and get the min and max of first and last row
        points = sorted(points, key=lambda i: i[1])
        size_x, size_y = self.GRID_SIZE[0], self.GRID_SIZE[1]
        first_row = points[0:size_x]
        last_row = points[size_x * (size_y - 1):]
        result = []

        result.append(min(first_row, key=lambda i: i[0]))
        result.append(max(first_row, key=lambda i: i[0]))
        result.append(min(last_row, key=lambda i: i[0]))
        result.append(max(last_row, key=lambda i: i[0]))
        return result

    def _drawCircles(self, img, color, circles, points):
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, radius, color, 3)
        for j in points:
            center = (j[0], j[1])
            cv2.circle(img, center, 3, color, 3)

    def _findCircles(self, mode):
        match mode:
            case 'RGB':
                img = self.rgb
                color = (255, 0, 0)
            case 'ES':
                img = self.es
                color = (0, 0, 255)
            case _:
                print('unknown mode passed to circle detection', file=sys.stderr)
                return None

        # convert to grayscale and add smoothing, then find all circles
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 32,
                                   param1=200, param2=30, minRadius=20, maxRadius=200)

        # error check
        # TODO: use selectROI() to manually correct circle detection
        if circles is None or (len(circles[0]) != self.GRID_SIZE[0] * self.GRID_SIZE[1]):
            print('wrong no. of circles found', file=sys.stderr)
            sys.exit(1)

        # find corner circles and draw in all circles
        circles = np.uint16(np.around(circles))
        points = [(i[0], i[1]) for i in circles[0, :]]
        points = self._findCorners(points)
        self._drawCircles(img, color, circles, points)
        return np.asarray(points, dtype=np.float32)

    def findTransformationMatrix(self):
        rgb_points = self._findCircles('RGB')
        es_points = self._findCircles('ES')
        m_transform = cv2.getPerspectiveTransform(es_points, rgb_points)
        return m_transform

    # visual confirmation to check if calibration was successful
    def display(self, matrix):
        self.es = cv2.warpPerspective(
            self.es, matrix, (self.rgb.shape[1], self.rgb.shape[0]))

        window = 'calibration'
        height, width = self.rgb.shape[:2]
        overlay = cv2.addWeighted(self.es, 0.5, self.rgb, 1, 0)

        cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)
        cv2.imshow(window, overlay)
        cv2.resizeWindow(window, int(height / 2), int(width / 2))
        cv2.waitKey(0)


if __name__ == "__main__":
    if (len(sys.argv) != 4):
        print('Usage: calibration.py <path_to_rgb> <path_to_es> <path_to_env>', file=sys.stderr)
        sys.exit(1)
    c = Calibrator(sys.argv[1], sys.argv[2])
    m = c.findTransformationMatrix()
    c.display(m)
    with open(sys.argv[3]) as infile:
        lines = infile.readlines()
    for line in lines:
        if 'CALIBRATION' in line:  # if there is already a calibration entry overwrite it
            lines = lines[:-3]
            mode = 'w'
            break
        else:                     # no entry yet, append it
            mode = 'a'
    with open(sys.argv[3], mode) as outfile:
        if mode == 'w':
            outfile.writelines(lines)
        outfile.write('CALIBRATION=')
        outfile.write('\'{}\'\n'.format(np.array2string(m)))
        outfile.close()
