import os
import json
import random

import wget
from PIL import Image
import requests
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.stats import mode
import matplotlib.pyplot as plt

# pygame is great but I need clean logs
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" 
import pygame

from helpers.BorderBuilder import BorderBuilder
from helpers.LineBuilder import LineBuilder
from helpers.ColorBuilder import ColorBuilder
from helpers.Painting import Painting

class MondrianPipeline:
    """The full input to output pipeline for transforming an image into a 
    Mondrian painting
    """
    def __init__(self, 
        image_in,
        random=False,
        output_dir='output/',
        hed_threshold=190,
        SIZE=500
    ):
        self.image_in = image_in
        self.output_dir = output_dir
        self.hed_threshold = hed_threshold
        self.SIZE = SIZE

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if random:
            self.get_random_image()

        self.step = 0

        # Vars to be set later
        self.line_builder = None
        self.color_builder = None
        self.painting = None

    def _step_files_forward(self, function):
        """Step the pipeline forward and name the new file after the current function"""
        old_file = self.image_in
        new_file = os.path.join(self.output_dir, f'{self.step}-{function}.jpg')
        self.image_in = new_file
        self.step += 1
        return old_file, new_file


    def resize(self):
        """Proportionally resize the input image so that the max height or 
        width is not bigger than SIZE
        """
        old_file, new_file = self._step_files_forward('resize')

        im = Image.open(old_file)

        width, height = im.width, im.height
        if width > height:
            new_width = self.SIZE
            new_height = int(new_width / width * height)
        else:
            new_height = self.SIZE
            new_width = int(new_height / height * width)

        im = im.resize((new_width, new_height))
        im.save(new_file)


    def find_primary_colors(self):
        """Make a ColorBuilder"""
        color_builder = ColorBuilder(self.image_in)
        color_builder.get_color_point()
        self.color_builder = color_builder

    
    def find_borders(self):
        """Make a BorderBuilder and save the images"""
        old_file, new_file = self._step_files_forward('apply-hed')

        border_builder = BorderBuilder(old_file)
        border_builder.apply_hed()
        border_builder.save_hed(new_file)


        old_file, new_file = self._step_files_forward('apply-hed-threshold')
        
        border_builder.apply_hed_threshold()
        border_builder.save_threshold(new_file)


    def find_structure(self):
        """Make a LineBuilder and save the image"""
        old_file, new_file = self._step_files_forward('find-structure')

        line_builder = LineBuilder(old_file)
        line_builder.analyze_image()
        line_builder.save(new_file)

        self.line_builder = line_builder


    def create_painting(self):
        """Make a Painting and save the image"""
        old_file, new_file = self._step_files_forward('create-painting')

        segments = self.line_builder.segments

        painting = Painting(self.line_builder, self.color_builder)
        painting.create()
        painting.save(new_file)

        self.painting = painting


    def create_overlay(self):
        """Convert to PNGs and overlay the input image on top of the painting"""
        old_file, new_file = self._step_files_forward('create-overlay')

        resize_im = [self.output_dir + x for x in os.listdir(self.output_dir) if 'resize' in x][0]
        background = Image.open(resize_im)
        overlay = Image.open(old_file)

        background = background.convert("RGBA")
        overlay = overlay.convert("RGBA")

        new_img = Image.blend(background, overlay, 0.5)
        new_img = new_img.convert('RGB')
        new_img.save(new_file)



    def apply_image_transform(self):
        """Usher the user through the pipeline"""
        self.resize()
        
        self.find_primary_colors()
        self.find_borders()
        self.find_structure()

        self.create_painting()

        self.create_overlay()


    def get_random_image(self):
        """Download random image of random dimensions from Unsplash"""
        if os.path.exists(self.image_in):
            os.remove(self.image_in)

        PIXEL_RANGE = range(200, 1000, 25)

        url = f'https://source.unsplash.com/random/{random.choice(PIXEL_RANGE)}x{random.choice(PIXEL_RANGE)}'

        r = requests.get(url)
        print(f'Random image url: {r.url}')

        wget.download(url, self.image_in)
        print() # for cleaner shell logs


def main():
    image = 'unsplash-random.jpg'
    # mp = MondrianPipeline(image)
    
    mp = MondrianPipeline(image, random=True)
    mp.apply_image_transform()


if __name__ == '__main__':
    main()
