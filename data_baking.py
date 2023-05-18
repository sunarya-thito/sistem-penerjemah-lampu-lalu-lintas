import json
import os

path = 'train.json'

test = 'test.txt'

test_file = open(test, 'w')

baked_data = {}

with open(path) as json_file:
    # read json file
    data = json.load(json_file)
    annotations = data['annotations']
    for an in annotations:
        if an['ignore'] == 1:
            continue
        filename = an['filename']
        # replace train_images\\ with ""
        filename = filename.replace('train_images\\', '')
        data = baked_data.get(filename)
        if data is None:
            baked_data[filename] = {}
            data = baked_data[filename]

        inbox = an['inbox']
        box = data.get('box')
        if box is None:
            data['box'] = []
            box = data['box']
        for inb in inbox:
            shape = inb['shape']
            if shape == '-1':
                continue
            color = inb['color']
            bndbox = inb['bndbox']
            if bndbox is None:
                continue
            xmin = bndbox['xmin']
            ymin = bndbox['ymin']
            xmax = bndbox['xmax']
            ymax = bndbox['ymax']
            luas = (xmax - xmin) * (ymax - ymin)
            if luas < 300:
                continue
            warna = None
            if color == 'red':
                warna = 'merah'
            elif color == 'yellow':
                warna = 'kuning'
            elif color == 'green':
                warna = 'hijau'
            else:
                raise Exception('Unknown color: ' + color)
            box.append({
                'warna': warna,
                'x': xmin,
                'y': ymin,
                'w': xmax - xmin,
                'h': ymax - ymin
            })
    # remove empty data
    for key in list(baked_data):
        if len(baked_data[key]['box']) == 0:
            del baked_data[key]
    # write baked_data to json file
    json.dump(baked_data, test_file)