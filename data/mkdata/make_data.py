# -*- coding: utf-8 -*-
import os
try:
    from xml.etree.cElementTree import parse
except:
    from xml.etree.ElementTree import parse

ims_folder = "./split_img"
ann_folder = "./split_annotation"

f_train = open('./detrac_train.txt', 'a')
f_val = open('./detrac_val.txt', 'a')

folders = os.listdir(ims_folder)
count = 0
for folder in folders:
    ims_path = os.path.join(ims_folder, folder)
    xml_path = os.path.join(ann_folder, folder)
    ims = os.listdir(ims_path)
    for im in ims:
        count += 1
        im_path = os.path.join(folder, im)
        an_path = os.path.join(xml_path, im.split('.')[0]+'.xml')
        tree = parse(an_path)
        s = im_path
        s+=' '+tree.find('size/width').text+' '+tree.find('size/height').text
        for ob in tree.iterfind('object/bndbox'):
            s+=' '+ob.find('xmin').text
            s+=' '+ob.find('ymin').text
            s+=' '+ob.find('xmax').text
            s+=' '+ob.find('ymax').text
            #s+=' '+'0'
        if count<8000:
            f_train.write(s+'\n')
        else:
            f_val.write(s+'\n')
f_train.close()
f_val.close()
