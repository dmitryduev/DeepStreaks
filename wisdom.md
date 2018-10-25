SSH-tunnel TensorBoard
```bash
ssh -L 16006:127.0.0.1:6006 ztf_root@rico.caltech.edu
cd /data/ztf/dev/deep-asteroids
tensorboard --logdir=logs/
```

---

QZ's cutouts on yupana:
Cutouts and ADES reports are here: /scr/apache/htdocs/marshals/ssm/zsrs/. 
Other metadata is at /scr/qye/ssmBackend/streaks/

---

`rsync`'ing `yupana` with `private`:
```bash
/usr/bin/rsync -av --delete-after -e 'ssh -p 22' duev@yupana.caltech.edu:/scr/apache/htdocs/marshals/ssm/zsrs/stamps_\* /scratch/ztf/streaks/stamps/
/usr/bin/rsync -av --delete-after -e 'ssh -p 22' duev@yupana.caltech.edu:/scr/qye/ssmBackend/streaks/2018\* /scratch/ztf/streaks/meta/
```

---

Implementing Batch Normalization in Tensorflow:

If you're just looking for a working implementation, `Tensorflow` has an easy to use `batch_normalization` layer 
in the `tf.layers` module. Just be sure to wrap your training step in a 
```python
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    ... 
```
and it will work.