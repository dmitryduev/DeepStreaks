# deep-asteroids: detecting Near-Earth Asteroids (NEAs) in the Zwicky Transient Facility (ZTF) data with deep learning

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

*rb: (0 is bogus, 1 is real) \[ResNet50_rb_20e_20181024_154924\]*

Confusion matrix:
```
[[481   3]
 [ 19 982]]
```

Normalized confusion matrix:
```
[[0.99380165 0.00619835]
 [0.01898102 0.98101898]]
```

*sl: (0 is long, 1 is short) \[ResNet50_sl_20e_20181024_163759\]*

Confusion matrix:
```
[[ 87   4]
 [  6 151]]
```

Normalized confusion matrix:
```
[[0.95604396 0.04395604]
 [0.03821656 0.96178344]]
```

The (mis)classifications are based on an `0.5` score cut: completeness can be increased by lowering the threshold.

---

## Production service  

### Set-up instructions

#### Pre-requisites

Clone the repo and cd to the `service` directory:
```bash
git clone https://github.com/dmitryduev/deep-asteroids.git
cd deep-asteroids/service
```

Create `secrets.json` with the admin user/password for the web app:
```json
{
  "database": {
    "admin_username": "ADMIN",
    "admin_password": "PASSWORD"
  }
}
```

#### Using `docker-compose` (for production)

Change `rico.caltech.edu` in `docker-compose.yml` and in `traefik/traefik.toml` to your domain. 

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

Create a persistent Docker volume for MongoDB and to store data:
```bash
docker volume create deep-asteroids-mongo-volume
docker volume create deep-asteroids-volume
```

Launch the MongoDB container. Feel free to change u/p for the admin, 
but make sure to change `config.json` correspondingly.
```bash
docker run -d --restart always --name deep-asteroids-mongo -p 27023:27017 -v deep-asteroids-mongo-volume:/data/db \
       -e MONGO_INITDB_ROOT_USERNAME=mongoadmin -e MONGO_INITDB_ROOT_PASSWORD=mongoadminsecret \
       mongo:latest
```

Build and launch the app container:
```bash
docker build -t deep-asteroids:latest -f Dockerfile .
#docker run --name deep-asteroids -d --restart always -p 8001:4000 -v deep-asteroids-volume:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest
# test mode:
#docker run -it --rm --name deep-asteroids -p 8001:4000 -v deep-asteroids-volume:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest
#docker run -it --rm --name deep-asteroids -p 8001:4000 -v /scratch/ztf/streaks:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest
# test mode on Dima's mpb:
docker run -it --rm --name deep-asteroids -p 8001:4000 -v /Users/dmitryduev/_caltech/python/deep-asteroids/_tmp:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest

```

The service will be available on port 8000 of the `Docker` host machine.