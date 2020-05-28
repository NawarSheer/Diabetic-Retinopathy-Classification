#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Credits 
https://github.com/mikevoets/jama16-retina-replication
Code for JAMA 2016; 316(22) Replication Study
@author: Nawar
"""

import csv
import sys
from shutil import rmtree
from PIL import Image
from glob import glob
from os import makedirs, rename
from os.path import join, splitext, basename, exists
from preprocess import resize_and_center_fundus
import pandas as pd

#parser = argparse.ArgumentParser(description='Preprocess EyePACS data set.')
#parser.add_argument("--data_dir", help="Directory where EyePACS resides.",
#                    default="data/eyepacs")

#args = parser.parse_args()
data_dir ="/data1/visionlab/data/EyePACS_2015/original/diabetic-retinopathy-detection/original_2/EyePACS_original_all"

train_labels = join(data_dir, 'EyePACS_2015_all.csv')
#test_labels = join(data_dir, 'testLabels.csv')

## Create directories for grades.
#[makedirs(join(data_dir, str(i))) for i in [0, 1, 2, 3, 4]
#        if not exists(join(data_dir, str(i)))]


new_path = join(data_dir, 'resized')
if exists(new_path):
    rmtree(new_path)
makedirs(new_path)



# Create a tmp directory for saving temporary preprocessing files.
tmp_path = join(data_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

failed_images = []

for labels in [train_labels]:
    with open(labels, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        for i, row in enumerate(reader):
            basename = row[0]

            im_path = glob(join(data_dir, "{}*".format(basename)))[0]

            # Find contour of eye fundus in image, and scale
            #  diameter of fundus to 299 pixels and crop the edges.
            res = resize_and_center_fundus(save_path=tmp_path,
                                           image_path=im_path,
                                           diameter=256, verbosity=0)

            # Status message.
            msg = "\r- Preprocessing image: {0:>7}".format(i+1)
            sys.stdout.write(msg)
            sys.stdout.flush()

            if res != 1:
                failed_images.append(basename)
                continue

            new_filename = "{0}.jpg".format(basename)

            # Move the file from the tmp folder to the right grade folder.
            rename(join(tmp_path, new_filename),
                   join(data_dir, new_path, new_filename))

# Clean tmp folder.
rmtree(tmp_path)

print("Could not preprocess {} images.".format(len(failed_images)))
print(", ".join(failed_images))