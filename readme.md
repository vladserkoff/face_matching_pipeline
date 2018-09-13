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
python make_dataset.py --reference_dir /mnt/datasets/reference --db_path /mnt/datasets/reference.sqlite
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
% http POST 0.0.0.0:8000/detect < secondface/tests/data/11_Meeting_Meeting_11_Meeting_Meeting_11_633.jpg

HTTP/1.1 200 OK
Connection: keep-alive
Content-Length: 329
Content-Type: application/json; charset=UTF-8
Date: Wed, 12 Sep 2018 16:02:51 GMT
Server: nginx/1.15.3

[
    {
        "best_match": "mariah carey",
        "coordinates": [
            222,
            100,
            342,
            263
        ],
        "distance": 0.5442394018173218,
        "id": 0
    },
    {
        "best_match": "irina framtsova",
        "coordinates": [
            85,
            213,
            177,
            331
        ],
        "distance": 0.7241394519805908,
        "id": 1
    },
    {
        "best_match": "ann morgan",
        "coordinates": [
            349,
            172,
            438,
            295
        ],
        "distance": 0.581076443195343,
        "id": 2
    }
]
```
