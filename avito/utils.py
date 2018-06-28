from zipfile import ZipFile
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import imghdr

from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3


def get_index_ranges(total_rows, rows_per_step=5000):
    """Get the ranges of row indices as a list of tuples. Each tuple contains
    the start and end row indices for a step.
    """
    # Get the total number of steps to process a dataframe
    num_steps = int(math.ceil(total_rows/rows_per_step))
    # Create a list of tuples, each indicating the start and end row indices of a step
    row_i_ranges = []
    for i in range(0, num_steps):
        if i < num_steps - 1:
            row_i_ranges.append((i * rows_per_step, (i + 1) * rows_per_step))
        else: 
            row_i_ranges.append((i * rows_per_step, total_rows))
    return row_i_ranges
    
def extract_images(zip_file_path, zipped_image_path, images_names):
    """Extract images with images_names from a given zipped file."""
    with ZipFile(zip_file_path, 'r') as zip_file:
        for image_name in images_names:
            # Exclude nan values
            if isinstance(image_name, str):
                image_path = zipped_image_path + image_name + ".jpg"
                zip_file.extract(image_path, path=image_path.split('/')[3])

def image_classify(model, model_class, image_name, images_path, top_n=1):
    """Classify image and return top matches."""
    if not os.path.isfile(images_path + image_name +".jpg"):
        return 0.0
    if not imghdr.what(images_path + image_name +".jpg"):
        return 0.0
    target_size = (224, 224)
    img = Image.open(images_path + image_name +".jpg")
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = model_class.preprocess_input(x)
    preds = model.predict(x)
    # Return the prediction score only
    return model_class.decode_predictions(preds, top=top_n)[0][0][2]