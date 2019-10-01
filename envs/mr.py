import copy
import PIL
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


def blur(image, bboxes=None):
    return transform(iaa.AverageBlur(k=3),
                     image,
                     bboxes)


def fliplr(image, bboxes=None):
    return transform(iaa.Fliplr(1.0),
                     image,
                     bboxes)


def flipud(image, bboxes=None):
    return transform(iaa.Flipud(1.0),
                     image,
                     bboxes)


def grayscale(image, bboxes=None):
    return transform(iaa.Grayscale(alpha=1.0),
                     image,
                     bboxes)


def invert(image, bboxes=None):
    return transform(iaa.Invert(1.0),
                     image,
                     bboxes)


def rotate(degrees, image, bboxes=None):
    return transform(iaa.Affine(rotate=degrees),
                     image,
                     bboxes)


def scale(ratio, image, bboxes=None):
    return transform(iaa.Affine(scale=ratio),
                     image,
                     bboxes)


def shear(degrees, image, bboxes=None):
    return transform(iaa.Affine(shear=degrees),
                     image,
                     bboxes)


def transform(augmenter, image, bboxes=None):
    if isinstance(image, PIL.Image.Image):
        is_image = True
        image = np.array(image)
    else:
        is_image = False

    mod_image = augmenter.augment_image(image)

    if is_image:
        mod_image = PIL.Image.fromarray(mod_image)

    if bboxes is not None:
        box_objects = []

        for annotation in bboxes:
            bb = annotation['bbox']
            x1, x2, y1, y2 = bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]
            box = ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            box_objects.append(box)

        bbs = ia.BoundingBoxesOnImage(box_objects, shape=image.shape)
        mod_bbs = augmenter.augment_bounding_boxes([bbs])

        mod_bboxes = copy.deepcopy(bboxes)

        for i, mod_bb in enumerate(mod_bbs[0].bounding_boxes):
            mod_bboxes[i]['bbox'] = [float(mod_bb.x1), float(mod_bb.y1), float(mod_bb.x2 - mod_bb.x1), float(mod_bb.y2 - mod_bb.y1)]
            del mod_bboxes[i]['segmentation']

        return mod_image, mod_bboxes
    else:
        return mod_image
