import time
# --------

import os
import json

from PIL import Image
import numpy as np
import cv2
import pygame
from sklearn.cluster import KMeans
from scipy.stats import mode
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean as distance

# from colors import mondrian_palette
from helpers.colors import mondrian_palette

class ColorBuilder:
    """
    ColorBuilder is a class that determines the colors used in a Mondrian painting.

    METHODS
    get_color_point: determine the primary color and the location of that color on an image
    get_color_box: given the co
    """

    def __init__(self, image_in):
        """
        """
        self.mondrian_palette = mondrian_palette
        self.mondrian_rgb = [v['rgb'] for k, v in mondrian_palette.items()]
        self.image_in = image_in

        im = Image.open(image_in)
        self.height = im.height
        self.width = im.width

        self.white = mondrian_palette['white']['rgb']
        self.black = mondrian_palette['black']['rgb']

        self.primary_color_palette = [
            color_codes['rgb'] for color_name, color_codes in mondrian_palette.items() 
                if any([c in color_name for c in ['red', 'yellow', 'blue']])
        ]
        
        self.primary_color_coordinate = None
        self.primary_color = None
        self.primary_color_box = None

    def get_color_point(self, REDUCE=5):
        """Find the point on the image that falls closest to the primary colors
        in Mondrian's palette. Set that point and it's respective mondrian color
        as the instance variables `primary_color_coordinate` and `primary_color`

        Comparing every pixel in an image to a palette can be a lengthy process. 
        The variable `REDUCE` shrinks the image proportionally to reduce runtime.
        """

        im = Image.open(self.image_in)
        new_height = self.height // REDUCE
        new_width = self.width // REDUCE
        im = im.resize((new_width, new_height))

        im_array = np.asarray(im)
        colors_flat = im_array.reshape((-1, 3))

        closest = min([
            min([{
                    'id': i,
                    'mondrian_color': mc,
                    'image_color': ic,
                    'distance': distance(mc, ic)
                } for mc in self.primary_color_palette], key=lambda x: x['distance']) 
             for i, ic in enumerate(colors_flat)
        ], key=lambda x: x['distance'])

        small_coordinates = [closest['id'] % new_width, closest['id'] // new_width]
        large_coordinates = [small_coordinates[0]*REDUCE, small_coordinates[1]*REDUCE]

        self.primary_color_coordinate = large_coordinates
        self.primary_color = closest['mondrian_color']


    def get_color_box(self, segments):
        """Given the cleaned segments provided by a LineBuilder, find the box 
        that the primary_color_coordinate point is surrounded by.
        """

        def between(p, seg, ind):
            return (seg[0][ind] <= p[ind] <= seg[1][ind]) or (seg[1][ind] <= p[ind] <= seg[0][ind])

        test_point = self.primary_color_coordinate

        height = self.height
        width = self.width

        if test_point[0] == width:
            test_point[0] -= 1
        if test_point[1] == height:
            test_point[1] -= 1

        horiz_segs = segments['y']
        # all segments north of test point
        north_segs = [seg for seg in horiz_segs if seg[1][1] < test_point[1]]
        # includes edges
        north_segs.append([[0, 0], [width, 0]])
        # all north segments that intersect with the point's x coordinates
        north_segs_intersect = [seg for seg in north_segs if between(test_point, seg, 0)]
        # the closest northern segment
        north_segs_closest = max(north_segs_intersect, key=lambda seg: seg[0][1])

        south_segs = [seg for seg in horiz_segs if seg[1][1] >= test_point[1]]
        south_segs.append([[0, height], [width, height]])
        south_segs_intersect = [seg for seg in south_segs if between(test_point, seg, 0)]
        south_segs_closest = min(south_segs_intersect, key=lambda seg: seg[0][1])

        vert_segs = segments['x']
        west_segs = [seg for seg in vert_segs if seg[1][0] < test_point[0]]
        west_segs.append([[0, 0], [0, height]])
        west_segs_intersect = [seg for seg in west_segs if between(test_point, seg, 1)]
        west_segs_closest = max(west_segs_intersect, key=lambda seg: seg[0][0])

        east_segs = [seg for seg in vert_segs if seg[1][0] >= test_point[0]]
        east_segs.append([[width, 0], [width, height]])
        east_segs_intersect = [seg for seg in east_segs if between(test_point, seg, 1)]
        east_segs_closest = min(east_segs_intersect, key=lambda seg: seg[0][0])

        # all relavent points
        rel_points = north_segs_closest + south_segs_closest + west_segs_closest + east_segs_closest

        # the four coordinates that define the box surrounding the test point
        x1 = max([p[0] for p in rel_points if p[0] <= test_point[0]])
        x2 = min([p[0] for p in rel_points if p[0] > test_point[0]])
        y1 = max([p[1] for p in rel_points if p[1] <= test_point[1]])
        y2 = min([p[1] for p in rel_points if p[1] > test_point[1]])

        '''
        (x1, y1) ------------- (x1, y2)
        |                            |
        |                            |
        |                            |
        |                            |
        |                            |
        (x2, y1) ------------- (x2, y2)

        '''
        # self.color_box = [x1, x2, y1, y2]
        # pygame requires different format
        self.primary_color_box = [x1, y1, x2-x1, y2-y1]
        return self.primary_color_box
