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

