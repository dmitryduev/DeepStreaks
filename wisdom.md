SSH-tunnel TensorBoard
```bash
ssh -L 16006:127.0.0.1:6006 ztf_root@rico.caltech.edu
cd /data/ztf/dev/deep-asteroids
tensorboard --logdir=logs/
```

QZ's cutouts on yupana:
Cutouts and ADES reports are here: /scr/apache/htdocs/marshals/ssm/zsrs/. 
Other metadata is at /scr/qye/ssmBackend/streaks/