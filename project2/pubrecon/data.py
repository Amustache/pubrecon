import os
import pandas as pd
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm
import cv2
from collections import defaultdict
import numpy as np


class DataFrame:
    def __init__(self, data_path, pickle_path=None):
        '''
        Define a new DataFrame.
        :param data_path: Folder to the data. Requires both `.jpg` and `.xml`, cf. README.
        :param pickle_path: Where to save the pickled DataFrame.
        '''
        self.data_path = data_path
        if pickle_path is None:
            self.pickle_path = os.path.join(data_path, 'dataframe.pickle')
        self.df = None
        self.files = None
        self.classes = None

    def prepare_data(self, force_preparation=False, subsamples=-1, verbose=1):
        '''
        Data preparator.
        :param force_preparation: If we want to enforce data preparation.
        :param subsamples: If `subsamples > -1`, will only use `subsamples` datapoints.
        :param verbose: 0:=no output, 1:=errors only, 2:= everything.
        '''
        if not os.path.exists(self.pickle_path) or force_preparation:  # If we need to convert data
            # Collect files
            names = []
            for root, dirs, files in os.walk(self.data_path):
                for name in files:
                    if name.split('.')[-1] == 'xml':
                        basename = '.'.join(name.split('.')[:-1])

                        # Check if we have the corresponding .jpg file
                        if os.path.exists(os.path.join(root, basename + '.jpg')):
                            out = os.path.join(root, basename)
                            names.append(out)
                            if verbose == 2:
                                print(out)
                        else:
                            if verbose:
                                print("Error with file {}: .jpg does not exists.".format(basename))

            # Columns
            columns = ['file_name', 'class_name', 'xmin', 'ymin', 'xmax', 'ymax']

            # Data
            data = []
            for name in names:
                xml_path = name + '.xml'
                try:
                    tree = ElementTree.parse(xml_path)
                except ElementTree.ParseError:  # The annotation file is missing.
                    if verbose:
                        print("Error with file {}: Error while parsing.".format(xml_path))
                    continue
                root = tree.getroot()

                for obj in root.findall('object'):
                    temp = [name + '.jpg', obj.find('name').text]
                    for child in obj.find('bndbox'):
                        temp.append(child.text)
                    data.append(temp)

                if subsamples != -1:
                    if subsamples > 0:
                        subsamples -= 1
                    else:
                        break

            # Create a new pandas dataframe
            self.df = pd.DataFrame(data, columns=columns)
            if verbose == 2:
                print(self.df.head())

            # Save pickle
            self.df.to_pickle(self.pickle_path)
        else:  # The data is already available
            self.df = pd.read_pickle(self.pickle_path)

        self.files = self.df.groupby('file_name').size()\
            .reset_index(name='counts')
        self.classes = self.df.groupby('class_name').size()\
            .reset_index(name='counts').sort_values(by='counts', ascending=False)

    # Why use len(foo.index)? https://stackoverflow.com/a/15943975
    def get_num_files(self):
        '''
        :return: Number of files.
        '''
        return len(self.files.index)

    def get_num_classes(self):
        '''
        :return: Number of classes.
        '''
        return len(self.classes.index)

    def get_num_pod(self):
        '''
        :return: Number of points of data.
        '''
        return len(self.df.index)

class ImagesData:
    def __init__(self, DataFrame, pickle_path=None):
        self.DataFrame = DataFrame
        self.df = self.DataFrame.df
        self.files = self.DataFrame.files
        if pickle_path is None:
            self.pickle_path = os.path.join(self.DataFrame.data_path, 'imagesdata.pickle')
        self.images = np.array([])
        self.labels = np.array([])

    def get_iou(self, bbox_1, bbox_2):
        '''
        Intersection over Union (IoU).
        https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        Basically, Area of Intersection / Area of Union.
        :param bbox_1: Dict, {xmin, ymin, xmax, ymax}
        :param bbox_2: Dict, {xmin, ymin, xmax, ymax}
        :return:
        '''
        x_left = max(bbox_1['xmin'], bbox_2['xmin'])
        y_top = max(bbox_1['ymin'], bbox_2['ymin'])
        x_right = min(bbox_1['xmax'], bbox_2['xmax'])
        y_bottom = min(bbox_1['ymax'], bbox_2['ymax'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (bbox_1['xmax'] - bbox_1['xmin']) * (bbox_1['ymax'] - bbox_1['ymin'])
        bb2_area = (bbox_2['xmax'] - bbox_2['xmin']) * (bbox_2['ymax'] - bbox_2['ymin'])

        return intersection_area / float(bb1_area + bb2_area - intersection_area)

    def prepare_images_and_labels(self, number_of_results=2000, iou_threshold=0.9, max_samples=10):
        # Uses optimized selective search
        # https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        # Loop over each image
        for index, row in tqdm(self.files.iterrows(), total=self.DataFrame.get_num_files(),
                               desc="Iterating through files..."):
            current_file = row[0]
            data = self.df.loc[self.df['file_name'] == current_file]
            img = cv2.imread(current_file)

            # Set image as the base for selective search
            ss.setBaseImage(img)

            # Initialising fast selective search and getting proposed regions
            ss.switchToSelectiveSearchFast()
            rects = ss.process()

            img_out = img.copy()

            # We need an uniform sample between classes
            classes_counter = defaultdict(int)

            # Iterate over the first N results of selective search
            # Calculate IOU of proposed region and annoted region
            used = False  # Check if that bbox is used as a class example

            # For each rectangle in the results of selective search
            for i, rect in enumerate(tqdm(rects, desc="Iterating through rectangles...", leave=False)):
                if i < number_of_results:  # We don't want to waste ressources on too many possibilities.
                    x, y, w, h = rect
                    rect_bbox = {'xmin': x, 'xmax': x + w, 'ymin': y, 'ymax': y + h}

                    # For each bbox within the image
                    for index, row in data.iterrows():
                        ground_truth_bbox = {'xmin': int(row['xmin']), 'xmax': int(row['xmax']),
                                             'ymin': int(row['ymin']), 'ymax': int(row['ymax'])}
                        ground_truth_class_name = row['class_name']

                        # Compare them
                        iou = self.get_iou(ground_truth_bbox, rect_bbox)

                        if iou > iou_threshold and classes_counter[ground_truth_class_name] < max_samples:
                            # Get the sample
                            img_sample = cv2.resize(img_out[y:y + h, x:x + w], (224, 224),
                                                    interpolation=cv2.INTER_AREA)
                            np.append(self.images, img_sample)  # Check if this shit does not bug...
                            np.append(self.labels, ground_truth_class_name)  # Check if this shit does not bug...
                            classes_counter[ground_truth_class_name] += 1
                            used = True
                        else:
                            continue

                    if not used and classes_counter['background'] < max_samples:
                        # We can use that bbox as a background example!
                        img_sample = cv2.resize(img_out[y:y + h, x:x + w], (224, 224),
                                                interpolation=cv2.INTER_AREA)  # Get the sample
                        np.append(self.images, img_sample)
                        np.append(self.labels, 'background')
                        classes_counter['background'] += 1
                else:
                    break
