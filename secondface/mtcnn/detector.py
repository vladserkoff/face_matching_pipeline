"""
Pytorch implementation of MTCNN face detection and alignment.
"""
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from secondface.mtcnn.box_utils import (calibrate_box, convert_to_square,
                                        get_image_boxes, nms)
from secondface.mtcnn.first_stage import run_first_stage
from secondface.mtcnn.get_nets import ONet, PNet, RNet


class MTCNN:
    """
    Detect faces on an image.
    """

    def __init__(self,
                 min_face_size: int,
                 thresholds: Tuple[float, float, float]) -> None:
        """
        Parameters
        ----------
        min_face_size : int
            Don't detect faces less than `min_face_size` along any axis.
        thresholds : list or tuple
            3 thresholds for 3 stages of MTCNN face detection algorithm
        """
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.pnet, self.rnet, self.onet = PNet(), RNet(), ONet()
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()

    def detect_faces(self, image: Image.Image) -> np.ndarray:
        """
        Detect faces on the image.

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        bounding_boxes : np.ndarray
            Numpy array of shape (number_of_faces, 5). 0:4 are coordinates
            of the box, 4th element is confidence of the model.
        """
        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / self.min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1
        preliminary_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = run_first_stage(
                image, self.pnet, scale=s, threshold=self.thresholds[0])
            preliminary_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        preliminary_boxes = [i for i in preliminary_boxes if i is not None]
        bounding_boxes = np.vstack(preliminary_boxes)

        keep = nms(bounding_boxes[:, 0:5], 0.7)
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5],
                                       bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = torch.tensor(img_boxes, dtype=torch.float)
        output = self.rnet(img_boxes)
        offsets = output[0].data.numpy()  # shape [n_boxes, 4]
        probs = output[1].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, 0.7)
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = torch.tensor(img_boxes, dtype=torch.float)
        output = self.onet(img_boxes)
        offsets = output[1].data.numpy()  # shape [n_boxes, 4]
        probs = output[2].data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
        offsets = offsets[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, 0.7, mode='min')
        bounding_boxes = bounding_boxes[keep]

        return bounding_boxes
