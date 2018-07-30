import os
import cv2
import numpy as np

def gen_my_sample(label_path="/Users/xingoo/Documents/brands/labels/001",
                  img_path="/Users/xingoo/Documents/brands/images/001", batch_size=1):
    files = os.listdir(label_path)
    for file in files:
        basename, postfix = os.path.splitext(file)

        if postfix.lower() not in ['.txt']:
            continue

        # 读取label
        with open(label_path + "/" + file, 'r') as f:
            lines = f.readlines()
        if int(lines[0]) > 0:

            # 读取图片
            image = [file for file in os.listdir(img_path) if file.startswith(basename)][0]
            img = cv2.imread(img_path + "/" + image)
            w, h, c = np.shape(img)

            datas = []

            for line in lines[1:]:
                x1_origin, y1_origin, x2_origin, y2_origin = line.strip().split(' ')

                x1 = float(x1_origin) * w
                y1 = float(y1_origin) * h
                x2 = float(x2_origin) * w
                y2 = float(y2_origin) * h

                xmin = int(min(x1, x2))
                xmax = int(max(x1, x2))
                ymin = int(min(y1, y2))
                ymax = int(max(y1, y2))

                print(str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax))
                datas.append([xmin,ymin,xmax,ymax])

            trans_dict_to_xml(image, datas)

def trans_dict_to_xml(name, datas):
    xml = []
    xml.append('\n<filename>{}</filename>'.format(name))
    for data in datas:
        xml.append('\n<object>\n<xmin>{xmin}</xmin>\n<ymin>{ymin}</ymin>\n<xmax>{xmax}</xmax>\n<ymax>{ymax}</ymax>\n</object>'\
                   .format(xmin=data[0],ymin=data[1],xmax=data[2],ymax=data[3]))

    result = '<annotation>{}</annotation>'.format(''.join(xml))
    print(result)

#trans_dict_to_xml('123.jpg',[[1,2,3,4],[1,2,3,4]])
gen_my_sample()