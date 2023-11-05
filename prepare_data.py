from tqdm import tqdm
from pathlib import Path
import shutil
import os
import json

DATA = "training"
EXTRACTED = "extracted"

dst_directory = Path(DATA)
if dst_directory.exists(): shutil.rmtree(dst_directory)

dst_directory.mkdir()
dst_directory.joinpath("train").mkdir()
dst_directory.joinpath("valid").mkdir()

src_directory = Path(EXTRACTED)



with open("extracted/publaynet/train.json") as handle:
    train_labels = json.load(handle)

filename2id = {img['file_name']: img['id'] for img in tqdm(train_labels['images'])}
id2filename = {img['id']: img['file_name'] for img in tqdm(train_labels['images'])}
id2filedims = {img['id']: [img['width'], img['height']] for img in tqdm(train_labels['images'])}
filename2filedims = {img['file_name']: [img['width'], img['height']] for img in tqdm(train_labels['images'])}

categorymapping = {cat['id']: cat['name'] for cat in tqdm(train_labels['categories'])}
categories = sorted(list(categorymapping.values()))

annotations = {}

for annot in tqdm(train_labels['annotations']):
    image_id = annot['image_id']
    bbox = annot['bbox']
    category_id = annot['category_id']

    if image_id not in annotations:
        annotations[image_id] = []
    
    annotations[image_id].append([category_id] + bbox)


# https://christianbernecker.medium.com/convert-bounding-boxes-from-coco-to-pascal-voc-to-yolo-and-back-660dc6178742
# x_yolo = (2 * x_coco + w_coco) / (2 * w_img)
# y_yolo = (2 * y_coco + h_coco) / (2 * h_img)
# w_yolo = w_coco / w_img
# h_yolo = h_coco / h_img
def coco_to_yolo_custom(x_coco, y_coco, w_coco, h_coco, w_img, h_img):
    x_yolo = (2 * x_coco + w_coco) / (2 * w_img)
    y_yolo = (2 * y_coco + h_coco) / (2 * h_img)
    w_yolo = w_coco / w_img
    h_yolo = h_coco / h_img
    
    return x_yolo, y_yolo, w_yolo, h_yolo

def yolo_to_coco(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    return [x1, y1, w, h]

for src in tqdm(src_directory.joinpath("publaynet", "train").glob("*.jpg")):
    # Copy image to path
    dst = dst_directory.joinpath("train", "images")
    dst.mkdir(exist_ok=True)
    shutil.copy(src, dst)

    # get annotations
    file_id = filename2id[os.path.basename(src)]

    annot = annotations[file_id]
    w, h = id2filedims[file_id]

    writable = []
    for ann in annot:
        category = ann[0] - 1
        bbox = ann[1:]

        x_yolo, y_yolo, w_yolo, h_yolo = coco_to_yolo_custom(bbox[0], bbox[1], bbox[2], bbox[3], w, h)

        writable.append(f"{category} {x_yolo} {y_yolo} {w_yolo} {h_yolo}")
    
    dst = dst_directory.joinpath("train", "labels")
    dst.mkdir(exist_ok=True)

    with open(dst.joinpath(src.stem + ".txt"), 'w') as handle:
        handle.write("\n".join(writable))