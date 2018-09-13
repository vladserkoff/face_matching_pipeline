# Face recognition and matching pipeline

Build reference database for known faces, then test new images against this database.

## Build dataset (optional)

To test with [LFW](http://vis-www.cs.umass.edu/lfw/) dataset, run

```bash
python make_lfw.py --lfw_dir /mnt/datasets/lfw --db_path /mnt/datasets/lfw/reference.sqlite
```

This will download LFW dataset and create sqlite database that stores reference faces (~15 minutes).  

If you want to test with your own dataset, run

```bash
python make_dataset.py --reference_dir /mnt/datasets/lfw/reference --db_path /mnt/datasets/lfw/reference.sqlite
```

This will look for images stored in `reference_dir` and store faces that it would find in a sqlite db.
It assumes that images are named as `person_name.#.ext` where `#` is optional number of the image of particular
person, `ext` is one of `jpg`, `jpeg`, `png`, `ppm`, `bmp`, `pgm`, `tif`. If there are several images of a
person, face embedding is an average of embeddings oth several images.

### Note

To run previous code you might want to first activate conda environment.

```bash
# conda env create -f environment.yml
source activate facematch
```

## Run

To run change the path to sqlite database file in `docker-compose.yml`. Then

```bash
docker-compose up --build -d

# I'm using httpie (conda install httpie)
% http POST 0.0.0.0:8000/detect < Maria_Shriver_0003.jpg                                     ~
HTTP/1.1 200 OK
Connection: keep-alive
Content-Length: 225
Content-Type: application/json; charset=UTF-8
Date: Thu, 13 Sep 2018 12:50:07 GMT
Server: nginx/1.15.3

[
    {
        "best_match": "maria shriver",
        "coordinates": [
            72,
            55,
            171,
            197
        ],
        "distance": 0.767863392829895,
        "id": 0
    },
    {
        "best_match": "arnold schwarzenegger",
        "coordinates": [
            178,
            5,
            250,
            153
        ],
        "distance": 0.6311142444610596,
        "id": 1
    }
]
```

![maria](Maria_Shriver_0003.jpg)

## Credits

* FaceBoxes by Zhang et al., as implemented in [FaceBoxes-tensorflow](https://github.com/TropComplique/FaceBoxes-tensorflow)  
* FaceNet by Schroff et al., as implemented in [facenet](https://github.com/davidsandberg/facenet)
