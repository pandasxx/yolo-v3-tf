import xml.etree.ElementTree as ET
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from net.config import cfg

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]


def convert_annotation(image_path, classes):
    
    in_file = image_path.split('.')[0] + '.xml'

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    for i, obj in enumerate(root.iter('object')):
        if i > 29:
            break
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b) + [cls_id]
        bboxes.extend(bb)
    if len(bboxes) < 30*5:
        bboxes = bboxes + [0, 0, 0, 0, 0]*(30-int(len(bboxes)/5))

    return np.array(bboxes, dtype=np.float32).flatten().tolist()

def convert_img(image_path):
    image = Image.open(image_path)
    resized_image = image.resize((cfg.data.img_size[0], cfg.data.img_size[1]), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')/255
    #print(image_data.shape)
    img_raw = image_data.tobytes()
    return img_raw

'''
filename = os.path.join('trainval'+'0712'+'.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)
for year, image_set in sets:
    image_ids = open('/home/raytroop/Dataset4ML/VOC%s/VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (
        year, year, image_set)).read().strip().split()
    # print(filename)
    for image_id in image_ids:
        xywhc = convert_annotation(year, image_id)
        img_raw = convert_img(year, image_id)

        example = tf.train.Example(features=tf.train.Features(feature={
            'xywhc':
                    tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
            'img':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
        writer.write(example.SerializeToString())
writer.close()
'''

def make_one_tfrecord_file(images_path, filename, classes):
    writer = tf.python_io.TFRecordWriter(filename)
    
    count = 0
    for image_path in images_path:
        if (image_path != ''):
            xywhc = convert_annotation(image_path, classes)
            img_raw = convert_img(image_path)

            example = tf.train.Example(features=tf.train.Features(feature={
                'xywhc':
                    tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
                'img':
                    tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
        writer.write(example.SerializeToString())
        count += 1
        if (count % 1000 == 0):
            print(count)
    writer.close()

def make_tfrecord_files(src_path, dst_path, nvme_path, name, classes, num_split = 10000):
    with open(src_path) as file_object:
        images_path = (file_object.read()).split("\n")

    tfrecord_file_nums = int(len(images_path) / num_split) + 1

    for i in range(tfrecord_file_nums):
        filename = os.path.join(nvme_path + name + '_%d' % (i) +'.tfrecords')
    
        print('make %d file' % (i))
        make_one_tfrecord_file(images_path[i*num_split : (i+1) * num_split], filename, classes)
        print('make %d file done' % (i))
        os.system("mv" + " " + filename + " " + dst_path)
        print('move %d file done' % (i))
