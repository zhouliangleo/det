import os
import json
import pandas as pd
import numpy as np
import cv2 as cv
from datetime import datetime


"""
Convert old annotations to coco format, only for check detection

Just for test, object categories may be different in relation network.

1: car

"""

INTERACT_CATS = ["Car"]

# global variable
# ToDo:
version=1
IMAGE_COUNT = 0
ANN_COUNT = 0


def make_coco_json_file(description, version, coco_like_info):
    # description = 'This is unstable 0.1 version of the VIRAT interact dataset,
    # from interact, heavy_carry, pull activity'

    info = {'description': description, 'url': 'https://github.com/zhouliangleo',
            'version': version, 'year': 2018, 'contributor': 'zl',
            'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    images = coco_like_info['coco_images_info']
    licenses = ['', '']
    annotations = coco_like_info['coco_annotations_info']

    categories = []
    for cat_i, cat in enumerate(INTERACT_CATS):
        if cat == 'Person':
            supercategory = 'person'
        elif cat in ['Bike', 'Car', 'Construction_Vehicle', 'Vehicle']:
            supercategory = 'vehicle'
        elif cat == 'Animal':
            supercategory = 'animal'
        elif cat == 'Other':
            supercategory = 'other'
        else:
            supercategory = 'object'
        categories.append({'supercategory': supercategory, 'id': cat_i+1, 'name': cat})

    coco_json = dict(
        info=info,
        images=images,
        licenses=licenses,
        annotations=annotations,
        categories=categories
    )

    return coco_json


def fill_coco_image_info(image_name, image_h, image_w):
    global IMAGE_COUNT
    IMAGE_COUNT += 1

    image_names = ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured',
                   'flickr_url', 'id', 'video_name']
    image_info = {k: '' for k in image_names}

    image_info['file_name'] = image_name
    image_info['height'] = 540   #int(image_h)
    image_info['width'] = 960    #int(image_w)
    image_info['date_captured'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_info['id'] = IMAGE_COUNT
    image_info['license'] = 1

    return image_info


def fill_coco_annotation_info(x1,y1,x2,y2,image_id,category_id):
    global ANN_COUNT
    # ANN_COUNT + 1, each the function is called
    # id can't be zero
    ANN_COUNT += 1
    # ToDo: add track_id for detection track
    annotation_names = ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']
    annotation_info = {k: '' for k in annotation_names}

    annotation_info['segmentation'] = [[]]
    annotation_info['area'] = 1.
    annotation_info['iscrowd'] = 0
    annotation_info['image_id'] = image_id
    annotation_info['bbox'] = [int(x1),int(y1),int(x2)-int(x1),int(y2)-int(y1)]
    annotation_info['category_id'] = category_id
    annotation_info['id'] = ANN_COUNT

    return annotation_info


def get_coco_annotation_info(line):
    """

    :param objects: objects of one frame
    :param image_id:
    :param merge_thresh:
    :return:
    """
    global IMAGE_COUNT
    coco_annotation_info = []
    line=line.split()[3:]
    num_bbox=int(len(line)/4)
    for i in range(num_bbox):
        coco_annotation_t = fill_coco_annotation_info(line[i*4],line[i*4+1],line[i*4+2],line[i*4+3],IMAGE_COUNT,1)
        coco_annotation_info.append(coco_annotation_t)

    return coco_annotation_info


def aggregate_coco_info(video_name, objects_array, interval=2):
    """

    :param video_name:
    :param objects_array: n * 8, additional column: activity_type
    :param merge_thresh:
    :return:
    """
    if video_name.startswith('VIRAT_S_0002'):
        image_size = (720, 1280)
    else:
        image_size = (1080, 1920)

    note_frames = np.unique(objects_array[:, 2])
    max_frame_id = objects_array[:, 2].max()
    print('Frames number', note_frames.size)

    coco_like_info = {'coco_images_info': [],
                      'coco_annotations_info': []}

    for frame_id in note_frames:
        if frame_id % interval != 0 or frame_id >= max_frame_id - interval:
            continue
        image_name = 'frame{}.jpg'.format(frame_id)
        # dict
        coco_image_info_this_frame = fill_coco_image_info(image_name, image_size, video_name)
        objects_this_frame = objects_array[objects_array[:, 2] == frame_id]
        # Add one column of coco_annotation id
        # list
        coco_annotation_info_this_frame = \
            get_coco_annotation_info(objects_this_frame, coco_image_info_this_frame['id'])
        coco_like_info['coco_images_info'].append(coco_image_info_this_frame)
        coco_like_info['coco_annotations_info'].extend(coco_annotation_info_this_frame)

    return coco_like_info


def main(mode):
    '''
    Recognize loading activity as heavy_carry
    :param mode:
    :return:
    '''
    # ToDo: add image from detection results
    # ToDo:
    global version
    description = 'This is unstable {} version of the VIRAT dataset, from old annotations, only used for detection'.format(version)

    if mode == 'train':
        index_path = './detrac_train.txt'

    elif mode == 'val':
        index_path = './detrac_val.txt'
    else:
        raise ValueError('Unsupported mode: {}'.format(mode))
    with open(index_path, 'r') as f:
        lines=f.readlines()
    coco_like_info={'coco_images_info':[],'coco_annotations_info':[]}
    for line in lines:
        image_name=line.split()[0]
        image_h=line.split()[1]
        image_w=line.split()[2]
        coco_image_info=fill_coco_image_info(image_name,image_h,image_w)
        coco_annotion_info=get_coco_annotation_info(line)
        coco_like_info['coco_images_info'].append(coco_image_info)
        coco_like_info['coco_annotations_info'].extend(coco_annotion_info)
    final_coco=make_coco_json_file(description,version,coco_like_info)
    json_file_name=os.path.join('annotations','car_coco_{}.json'.format(mode))
    with open(json_file_name,'w') as f:
        json.dump(final_coco,f)












if __name__ == '__main__':
    main('train')
    main('val')

