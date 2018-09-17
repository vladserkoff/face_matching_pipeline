# pylint: disable = all
"""
Run face matching
"""

import json
import os

import falcon

from facematch import Reference, FaceMatcher, read_image

REFERENCE_PATH = os.getenv('REFERENCE_PATH', '/mnt/reference.sqlite')
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', 0.0))


class FaceMatching(object):
    def __init__(self, reference_db):
        self._reference = Reference(reference_db)
        self.sface = FaceMatcher(self._reference)

    def on_post(self, req, resp):
        img_bytes = req.stream.read()
        image = read_image(img_bytes)
        result = self.sface.recoginze(image, threshold=DETECTION_THRESHOLD)
        resp.body = json.dumps(result, ensure_ascii=False)


app = falcon.API()
face_rec = FaceMatching(REFERENCE_PATH)
app.add_route('/detect', face_rec)
