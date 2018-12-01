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

crontab:
```crontab
* * * * * /usr/bin/rsync -av --delete-after -e 'ssh -p 22' duev@yupana.caltech.edu:/scr/apache/htdocs/marshals/ssm/zsrs/stamps_`date -u "+\%Y\%m\%d"`/ /scratch/ztf/streaks/stamps/stamps_`date -u "+\%Y\%m\%d"`/ >/scratch/duev/cron.log 2>&1
*/5 * * * * /usr/bin/rsync -av --delete-after -e 'ssh -p 22' duev@yupana.caltech.edu:/scr/qye/ssmBackend/streaks/`date -u "+\%Y\%m\%d"`/ /scratch/ztf/streaks/meta/`date -u "+\%Y\%m\%d"`/ >/scratch/duev/cron.log 2>&1
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

---

Mark all cutouts on page as "no_streak" (check all checkboxes)

```javascript
$("input[type=checkbox][value='no_streak']").click();
//$("input[type=checkbox][value='no_streak']").prop("checked",true);
```

---

Unzip datasets
```bash
for f in *.zip; do unzip -d ${f%.zip} $f; done;
```
