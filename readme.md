# DeepStreaks: identifying Near-Earth Asteroids (NEAs) in the Zwicky Transient Facility (ZTF) data with deep learning

DeepStreaks is a deep learning framework developed to efficiently identify streaking near-Earth objects in the data of 
the [Zwicky Transient Facilty (ZTF)](https://ztf.caltech.edu), a wide-field time-domain survey using a dedicated 47 sq. deg. camera 
attached to the Samuel Oschin Telescope at the Palomar Observatory in California, United States. 
The performance is great: well above 95% accuracy when compared to the performance of human scanners, 
reaching ~200% in some cases. The system is deployed and is adapted for usage within the ZTF Solar system framework.

From December 15, 2018 - January 15, 2018, over 10 NEAs were discovered with the help of DeepStreaks.

For details, please see Duev et al., MNRAS, 2019 (in prep.).

## Models: architecture, data, training, and performance

### Network architecture

We are using three "families' of binary classifiers. Individual classifiers from each such family are trained 
to answer one of the following questions, respectively:

- "rb": bogus or real streak? All streak-like objects are marked as real, including actual streaks from 
fast moving objects, long streaks from satellites, and cosmic rays.

- "sl": long or short streak? 

- "kd": ditch ot keep? Is this a real streak, or a cosmic ray/other artifact?

![](doc/DeepStreaks.png) 

Input image dimensions - `144x144x1` (gray scale).

The models are implemented using `Keras` with a `TensorFlow` backend (GPU-enabled). 


### Data sets

The data were prepared using [Zwickyverse](https://github.com/dmitryduev/zwickyverse).

todo: update to 2019.1

#### bogus vs real (2018.9)

- 905 long streaks from ZTF data
- 5000 bogus cutouts from ZTF data
- 8270 (4135 + 4135) synthetic short streaks from QZ (generated May 25, 2018)
- 669 real short streaks from ZTF data from May 1 to Aug 31, 2018

#### long streak vs short streak (2018.9)  

- 905 long streaks from ZTF data
- 905 synthetic short streaks from QZ (generated May 25, 2018)
- 669 real short streaks from ZTF data from May 1 to Aug 31, 2018

...


### Training and performance

The models were trained on-premise at Caltech on a Nvidia Tesla P100 GPU (12G) 
for 200 epochs with a mini-batch size of 32 (see `deepstreaks.py` for the details).


![](doc/all_acc.png)

![](doc/roc_rb_sl_kd.png)

![](doc/cm_rb_sl_kd_annotated.png)

![](doc/venn3_rb_sl_kd_adapted.png)


#### Example of real Near-Earth Objects identified by DeepStreaks

![](doc/reals_zoo.png)

---

## Production service  

### Set-up instructions

#### Pre-requisites

Clone the repo and cd to the `service` directory:
```bash
git clone https://github.com/dmitryduev/DeepStreaks.git
cd DeepStreaks/service
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
docker build --rm -t deep-asteroids:latest -f Dockerfile .
# rico:
#docker run --runtime=nvidia --name deep-asteroids -d --restart always -p 8001:4000 -v /data/ztf/streaks:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest
# test mode:
#docker run --rm -it --runtime=nvidia --name deep-asteroids -p 8001:4000 -v /data/ztf/streaks:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest
# private:
#docker run --name deep-asteroids -d --restart always -p 8001:4000 -v /scratch/ztf/streaks:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest
# test mode:
#docker run -it --rm --name deep-asteroids -p 8001:4000 -v deep-asteroids-volume:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest
#docker run -it --rm --name deep-asteroids -p 8001:4000 -v /scratch/ztf/streaks:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest
# test mode on Dima's mpb:
docker run -it --rm --name deep-asteroids -p 8001:4000 -v /Users/dmitryduev/_caltech/python/deep-asteroids/_tmp:/data --link deep-asteroids-mongo:mongo deep-asteroids:latest

```

The service will be available on port 8000 of the `Docker` host machine.