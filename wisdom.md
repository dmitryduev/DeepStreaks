SSH-tunnel TensorBoard
```bash
ssh -L 16006:127.0.0.1:6006 ztf_root@rico.caltech.edu
cd /data/ztf/dev/deep-asteroids
tensorboard --logdir=logs/
```

QZ's cutouts on yupana:
Cutouts and ADES reports are here: /scr/apache/htdocs/marshals/ssm/zsrs/. 
Other metadata is at /scr/qye/ssmBackend/streaks/


Implementing Batch Normalization in Tensorflow:

If you're just looking for a working implementation, `Tensorflow` has an easy to use `batch_normalization` layer 
in the `tf.layers` module. Just be sure to wrap your training step in a 
```python
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    ... 
```
and it will work.