# Face recognition for Second Memory

## Run

```bash
docker-compose up --build -d

# I'm using httpie
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
