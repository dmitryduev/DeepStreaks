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
import tensorflow as tf
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

---

Train all
```bash
for p in 5b96af9c0354c9000b0aea36 5b99b2c6aec3c500103a14de 5be0ae7958830a0018821794 5c05bbdc826480000a95c0bf; do for m in VGG6 ResNet50 DenseNet121; do echo $p $m; python deepstreaks.py --project_id $p --model $m --class_weight --verbose; done; done
```

---

Iterate over range of dates:
```bash
for d in {50..279}; do echo `date +%Y%m%d -d "$d day ago"`; done
for d in {50..279}; do echo `date +%Y%m%d -d "$d day ago"`; python fetcher.py --obsdate `date +%Y%m%d -d "$d day ago"` --enforce --looponce; python watcher.py config.json --obsdate `date +%Y%m%d -d "$d day ago"` --enforce --looponce; done
for d in {50..279}; do echo `date +%Y%m%d -d "$d day ago"`; python fetcher.py --obsdate `date +%Y%m%d -d "$d day ago"` --enforce --looponce; done
for d in {272..100..-1}; do echo `date +%Y%m%d -d "$d day ago"`; python fetcher_async.py --obsdate `date +%Y%m%d -d "$d day ago"` --looponce; done
python fetcher.py --obsdate `date +%Y%m%d -d "$d day ago"` --enforce --looponce
python watcher.py config.json --obsdate `date +%Y%m%d -d "$d day ago"` --enforce --looponce
python fetcher.py --obsdate 20181213 --enforce --looponce
python watcher.py config.json --obsdate 20181213 --enforce --looponce
```