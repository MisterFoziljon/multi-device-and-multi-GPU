## Bitta tarmoqqa ulangan super kompyuterlarning GPU larini bir vaqtda parallel ishlashini ta'minlab beruvchi dastur.


### **Requirements:**
```python
 pip install tensorflow[and-cuda]
```

### **Qo'llanma:**

* [tensorflow.org](https://www.tensorflow.org/guide/distributed_training?hl=ru)
* [github](https://github.com/tensorflow/docs/blob/master/site/en/guide/distributed_training.ipynb)
* [google colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/distributed_training.ipynb)

### **Information:**

* ``` tf.distribute.MultiWorkerMirroredStrategy() ``` - bitta tarmoqqa ulangan nechta device larni sinxron ishlatish uchun qo'llaniladi. Bunda har bir device bir nechta GPU ga ega bo'lishi mumkin.

* ``` TF_CONFIG ``` - har bir device ning iplari va ishchi sifatidagi indexlari keltirilgan konfiguratsiya. Uni terminal yordamida export qilish yoki kod yordamida ishga tushirish mumkin.


**Kod yordamida ishga tushirish**

```python
 tf_config = {
    'cluster': {'worker': ['192.169.0.146:12345', '192.169.0.128:12345']},
    'task': {'type': 'worker', 'index': 0}
}
tf_config['task']['index'] = 0
os.environ['TF_CONFIG'] = json.dumps(tf_config)
```


**Terminal yordamida ishga tushirish**
```shell
user@User$: export TF_CONFIG='{"cluster": {"worker": ["192.169.0.146:12345", "192.169.0.128:12345"]}, "task": {"index": 0, "type": "worker"}}'
```

### **How can start train?:**
1. Kerakli requiremetlarni o'rnating.
2. TF_CONFIG scriptini export qiling.
3. ```train.py``` faylini foydalanayotgan device laringizda quyidagi tartibda ishga tushiring (2 ta device uchun):
    - 192.169.0.146 ipga ega device uchun index:0
    - 192.169.0.128 ipga ega device uchun index:1
    - device lar soni ko'p bo'lsa, boshqa devicelarga 2, 3 va hokazo index beriladi.
    - yoki ```train.py``` file har bir qurilma uchun o'z indexi bilan yoziladi: tf_config['task']['index'] = 1

