# deep-asteroids: detecting Near-Earth Asteroids (NEAs) in ZTF data with deep learning

## Models: architecture, data, training, and performance

### Network architecture

We are using two deep residual networks (`ResNet50`):

![](doc/resnet.png) 

Input image dimensions - `144x144x1` (gray scale).

The first resnet outputs a real/bogus (`rb`) score (streak -- no streak); the second - short/long streak (`sl`) score.

The models are implemented using `Keras` with a `TensorFlow` backend (GPU-enabled). 


### Data sets

The data were prepared on `Zwickyverse` (`https://private.caltech.edu`).

#### bogus vs real (2018.9)

- 905 long streaks from ZTF data
- 5000 bogus cutouts from ZTF data
- 8270 (4135 + 4135) synthetic short streaks from QZ (generated May 25, 2018)
- 669 real short streaks from ZTF data from May 1 to Aug 31, 2018

#### long streak vs short streak (2018.9)  

- 905 long streaks from ZTF data
- 905 synthetic short streaks from QZ (generated May 25, 2018)
- 669 real short streaks from ZTF data from May 1 to Aug 31, 2018


### Training and performance

The models were trained on `rico`'s Nvidia Tesla P100 GPU (12G) 
for 20 epochs with a mini-batch size of 32 (see `resnet50.py`), which takes 20 minutes for the `rb` model.

*rb: (0 is bogus, 1 is real)*

Confusion matrix:
```
[[473   5]
 [ 22 985]]
```

Normalized confusion matrix:
```
[[0.98953975 0.01046025]
 [0.02184707 0.97815293]]
```

*sl: (0 is long, 1 is short)*

Confusion matrix:
```
[[ 85   4]
 [  2 157]]
```

Normalized confusion matrix:
```
[[0.95505618 0.04494382]
 [0.01257862 0.98742138]]
```

The (mis)classifications are based on an `0.5` score cut: completeness can be increased by lowering the threshold.


## Production service  

### Set-up instructions

#### Pre-requisites

Clone the repo and cd to the directory:
```bash
git clone https://github.com/dmitryduev/zwickyverse.git
cd zwickyverse
```

Create `secrets.json` with the `Kowalski` login credentials and admin user/password for the website:
```json
{
  "kowalski": {
    "user": "USER",
    "password": "PASSWORD"
  },
  "database": {
    "admin_username": "ADMIN",
    "admin_password": "PASSWORD"
  }
}
```

#### Using `docker-compose` (for production)

Change `private.caltech.edu` on line 34 in `docker-compose.yml` and line 88 in `traefik/traefik.toml` to your domain. 

Run `docker-compose` to start the service:
```bash
docker-compose up --build -d
```

To tear everything down (i.e. stop and remove the containers), run:
```bash
docker-compose down
```

---

#### Using plain `Docker` (for dev/testing)

If you want to use `docker run` instead:

Create a persistent Docker volume for MongoDB and to store thumbnails etc.:
```bash
docker volume create zwickyverse-mongo-volume
docker volume create zwickyverse-volume
```

Launch the MongoDB container. Feel free to change u/p for the admin, 
but make sure to change `config.json` correspondingly.
```bash
docker run -d --restart always --name zwickyverse-mongo -p 27020:27017 -v zwickyverse-mongo-volume:/data/db \
       -e MONGO_INITDB_ROOT_USERNAME=mongoadmin -e MONGO_INITDB_ROOT_PASSWORD=mongoadminsecret \
       mongo:latest
```

Build and launch the main container:
```bash
docker build -t zwickyverse:latest -f Dockerfile .
docker run --name zwickyverse -d --restart always -p 8000:4000 -v zwickyverse-volume:/data --link zwickyverse-mongo:mongo zwickyverse:latest
# test mode:
docker run -it --rm --name zwickyverse -p 8000:4000 -v zwickyverse-volume:/data --link zwickyverse-mongo:mongo zwickyverse:latest
```

The service will be available on port 8000 of the `Docker` host machine