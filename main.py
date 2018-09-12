# pylint: disable = all
"""
Run secondface
"""

import json

import falcon

from secondface import Reference, FaceMatcher, read_image

REFERENCE_PATH = '/mnt/reference.sqlite'


class FaceRecognition(object):
    def __init__(self, reference_db):
        self._reference = Reference(reference_db)
        self.sface = FaceMatcher(self._reference)

    def on_post(self, req, resp):
        img_bytes = req.stream.read()
        image = read_image(img_bytes)
        result = self.sface.recoginze(image)
        resp.body = json.dumps(result, ensure_ascii=False)


app = falcon.API()
face_rec = FaceRecognition(REFERENCE_PATH)
app.add_route('/detect', face_rec)
