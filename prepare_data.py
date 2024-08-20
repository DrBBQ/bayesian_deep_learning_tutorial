import numpy as np
import cv2
import untangle
import os

from data_utils import LABEL_MAP as label_map

annotations_root = "C:\\Users\\Matt\\Downloads\\archive\\annotations"
image_root = "C:\\Users\\Matt\\Downloads\\archive\\images"
cropped_image_root = "C:\\Users\\Matt\\Downloads\\archive\\images_cropped"

os.makedirs(cropped_image_root, exist_ok=True)

inputs = []
outputs = []


def process_image(img, xmin, xmax, ymin, ymax):
    cropped_image = img[ymin:ymax, xmin:xmax]

    resized_cropped_image = cv2.resize(
        cropped_image, (64, 64), interpolation=cv2.INTER_LINEAR
    )
    return resized_cropped_image


for i, f in enumerate(os.listdir(annotations_root)):
    data_file = os.path.join(annotations_root, f)

    data = untangle.parse(data_file)

    filename = data.annotation.filename.cdata

    img = cv2.imread(os.path.join(image_root, filename))

    if type(data.annotation.object) == list:
        idx = 0
        while 1:
            try:
                xmin = int(data.annotation.object[idx].bndbox.xmin.cdata)
                xmax = int(data.annotation.object[idx].bndbox.xmax.cdata)
                ymin = int(data.annotation.object[idx].bndbox.ymin.cdata)
                ymax = int(data.annotation.object[idx].bndbox.ymax.cdata)
                label = data.annotation.object[idx].name.cdata
            except:
                break
            processed_image = process_image(img, xmin, xmax, ymin, ymax)
            inputs.append(processed_image)
            outputs.append(label_map[label])
            filename_idx = f"{filename[:-4]}_{idx}.png"
            cv2.imwrite(os.path.join(cropped_image_root, filename_idx),
                                     processed_image)
            idx += 1
    else:
        xmin = int(data.annotation.object.bndbox.xmin.cdata)
        xmax = int(data.annotation.object.bndbox.xmax.cdata)
        ymin = int(data.annotation.object.bndbox.ymin.cdata)
        ymax = int(data.annotation.object.bndbox.ymax.cdata)
        label = data.annotation.object.name.cdata
        processed_image = process_image(img, xmin, xmax, ymin, ymax)
        inputs.append(processed_image)
        outputs.append(label_map[label])
        cv2.imwrite(os.path.join(cropped_image_root, filename),
                                 processed_image)


inputs = np.array(inputs) / 255.0
outputs = np.array(outputs)
np.save("inputs.npy", inputs)
np.save("outputs.npy", outputs)
