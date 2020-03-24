import math
import os
import xml.etree.ElementTree

import numpy as np
import paddle
import six
from PIL import Image

from utils import image_util


class Settings(object):
    def __init__(self,
                 label_file_path=None,
                 resize_h=300,
                 resize_w=300,
                 mean_value=127.5,
                 std_value=0.007843,
                 apply_distort=True,
                 apply_expand=True,
                 ap_version='11point'):
        self._ap_version = ap_version
        self._label_list = []
        for line in open(label_file_path):
            self._label_list.append(line.strip())

        self._apply_distort = apply_distort
        self._apply_expand = apply_expand
        self._resize_height = resize_h
        self._resize_width = resize_w
        self._img_mean = mean_value
        self._img_std = std_value
        self._expand_prob = 0.5
        self._expand_max_ratio = 4
        self._hue_prob = 0.5
        self._hue_delta = 18
        self._contrast_prob = 0.5
        self._contrast_delta = 0.5
        self._saturation_prob = 0.5
        self._saturation_delta = 0.5
        self._brightness_prob = 0.5
        self._brightness_delta = 0.125

    @property
    def ap_version(self):
        return self._ap_version

    @property
    def apply_expand(self):
        return self._apply_expand

    @property
    def apply_distort(self):
        return self._apply_distort

    @property
    def label_list(self):
        return self._label_list

    @property
    def resize_h(self):
        return self._resize_height

    @property
    def resize_w(self):
        return self._resize_width

    @property
    def img_mean(self):
        return self._img_mean

    @property
    def img_std(self):
        return self._img_std


def preprocess(img, bbox_labels, mode, settings):
    img_width, img_height = img.size
    sampled_labels = bbox_labels
    if mode == 'train':
        if settings._apply_distort:
            img = image_util.distort_image(img, settings)
        if settings._apply_expand:
            img, bbox_labels, img_width, img_height = image_util.expand_image(
                img, bbox_labels, img_width, img_height, settings)
        # sampling
        batch_sampler = []
        # hard-code here
        batch_sampler.append(image_util.sampler(1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0))
        batch_sampler.append(image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 0.0))
        batch_sampler.append(image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 0.0))
        batch_sampler.append(image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 0.0))
        batch_sampler.append(image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 0.0))
        batch_sampler.append(image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 0.0))
        batch_sampler.append(image_util.sampler(1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0))
        sampled_bbox = image_util.generate_batch_samples(batch_sampler, bbox_labels)

        img = np.array(img)
        if len(sampled_bbox) > 0:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            img, sampled_labels = image_util.crop_image(img, bbox_labels, sampled_bbox[idx], img_width, img_height)

        img = Image.fromarray(img)
    img = img.resize((settings.resize_w, settings.resize_h), Image.ANTIALIAS)
    img = np.array(img)

    if mode == 'train':
        mirror = int(np.random.uniform(0, 2))
        if mirror == 1:
            img = img[:, ::-1, :]
            for i in range(len(sampled_labels)):
                tmp = sampled_labels[i][1]
                sampled_labels[i][1] = 1 - sampled_labels[i][3]
                sampled_labels[i][3] = 1 - tmp
    # HWC to CHW
    if len(img.shape) == 3:
        img = np.swapaxes(img, 1, 2)
        img = np.swapaxes(img, 1, 0)
    img = img.astype('float32')
    img -= settings.img_mean
    img = img * settings.img_std
    return img, sampled_labels


def pascalvoc(settings, file_list, mode, batch_size, shuffle):
    def reader():
        if mode == 'train' and shuffle:
            np.random.shuffle(file_list)
        batch_out = []
        cnt = 0
        for image in file_list:
            image_path, label_path = image.split('\t')
            if not os.path.exists(image_path):
                raise ValueError("%s is not exist, you should specify data path correctly." % image_path)
            im = Image.open(image_path)
            if im.mode == 'L':
                im = im.convert('RGB')
            im_width, im_height = im.size

            # layout: label | xmin | ymin | xmax | ymax | difficult
            bbox_labels = []
            root = xml.etree.ElementTree.parse(label_path).getroot()
            for object in root.findall('object'):
                bbox_sample = []
                # start from 1
                bbox_sample.append(float(settings.label_list.index(object.find('name').text)))
                bbox = object.find('bndbox')
                difficult = float(object.find('difficult').text)
                bbox_sample.append(float(bbox.find('xmin').text) / im_width)
                bbox_sample.append(float(bbox.find('ymin').text) / im_height)
                bbox_sample.append(float(bbox.find('xmax').text) / im_width)
                bbox_sample.append(float(bbox.find('ymax').text) / im_height)
                bbox_sample.append(difficult)
                bbox_labels.append(bbox_sample)
            im, sample_labels = preprocess(im, bbox_labels, mode, settings)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) == 0: continue
            im = im.astype('float32')
            boxes = sample_labels[:, 1:5]
            lbls = sample_labels[:, 0].astype('int32')
            difficults = sample_labels[:, -1].astype('int32')

            batch_out.append((im, boxes, lbls, difficults))
            if len(batch_out) == batch_size:
                yield batch_out
                cnt += len(batch_out)
                batch_out = []

        if mode == 'test' and len(batch_out) > 1:
            yield batch_out
            cnt += len(batch_out)

    return reader


def train(settings, file_list_path, batch_size, shuffle=True, use_multiprocess=True, num_workers=4):
    readers = []
    images = [line.strip() for line in open(file_list_path)]
    np.random.shuffle(images)
    n = int(math.ceil(len(images) // num_workers)) if use_multiprocess else len(images)
    image_lists = [images[i:i + n] for i in range(0, len(images), n)]
    for l in image_lists:
        readers.append(pascalvoc(settings, l, 'train', batch_size, shuffle))

    if use_multiprocess:
        return paddle.reader.multiprocess_reader(readers, False)
    else:
        return readers[0]


def test(settings, file_list_path, batch_size):
    image_list = [line.strip() for line in open(file_list_path)]
    return pascalvoc(settings, image_list, 'test', batch_size, False)
