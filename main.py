import tensorflow as tf
from numpy import random
import tqdm
%matplotlib inline
import csv
import numpy as np
from scipy.stats import expon
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
data1 = np.loadtxt('C:/Users/ImmaTech/Downloads/Heart_ECG/bit1.csv')
data2 = np.loadtxt('C:/Users/ImmaTech/Downloads/Heart_ECG/bit2.csv')
a = data1.tolist()
p = data2.tolist()
print(a)
btod1 = []
btod2 = []

fbtod = []
for x in range(3000):
b = str(a[x]).strip('[]')
c = b.split('.')
d = int(c[0],2)
btod1.append(d)
for y in range(3000):
q = str(p[y]).strip('[]')
r = q.split('.')
s = int(r[0],2)
btod2.append(s)
for z in range(3000):
fbtod.append(str(btod1[z])+"."+str(btod2[z]))
write_data = open('C:/Users/ImmaTech/Downloads/Heart_ECG/Dataset/Final.csv','w')
out = csv.writer(write_data)
out.writerows(map(lambda x:[x],fbtod))
write_data.close()

read_data = np.loadtxt('C:/Users/ImmaTech/Downloads/Heart_ECG/Dataset/Final.csv')
sf = 100.
time = np.arange(read_data.size) / sf
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt.plot(time, read_data, lw=1.5, color='k')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.xlim([time.min(), time.max()])
plt.title('EEG Wave Signal')
sns.despine()
import tensorflow as tf
writer_val =
tf.summary.FileWriter('C:/Users/ImmaTech/Downloads/Heart_ECG/log/dataset_ecg')
writer_train =
tf.summary.FileWriter('C:/Users/ImmaTech/Downloads/Heart_ECG/log/normalization_e
cg')
loss_var = tf.Variable(0.0)
tf.summary.scalar('ECG_Wave_Signal', loss_var)

write_op = tf.summary.merge_all()
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
for i in range(3000):
summary = session.run(write_op, {loss_var: read_data[i]})
writer_val.add_summary(summary, i)
writer_val.flush()
summary = session.run(write_op, {loss_var: random.uniform(-5,30)})
writer_train.add_summary(summary, i)
writer_train.flush()
writer_val.close()
writer_train.close()
import os
import tqdm
import tensorflow as tf
def tb_test():

result = -30
if(result < 30):
sess = tf.Session()
x = tf.placeholder(dtype=tf.float32)
summary = tf.summary.scalar('High_Level', x)
merged = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
writer_1 = tf.summary.FileWriter(os.path.join('log', 'Normal'))
writer_2 = tf.summary.FileWriter(os.path.join('log', 'Level'))
for i in tqdm.tqdm(range(200)):
summary_1 = sess.run(merged, feed_dict={x: i-10})
writer_1.add_summary(summary_1, i)
summary_2 = sess.run(merged, feed_dict={x: i+30})
writer_2.add_summary(summary_2, i)
writer_1.close()
writer_2.close()
else:

sess = tf.Session()
x = tf.placeholder(dtype=tf.float32)
summary = tf.summary.scalar('Low_Level', x)
merged = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
writer_1 = tf.summary.FileWriter(os.path.join('log', 'Normal'))
writer_2 = tf.summary.FileWriter(os.path.join('log', 'Level'))
for i in tqdm.tqdm(range(200)):
summary_1 = sess.run(merged, feed_dict={x: i-10})
writer_1.add_summary(summary_1, i)
summary_2 = sess.run(merged, feed_dict={x: i+30})
writer_2.add_summary(summary_2, i)
writer_1.close()
writer_2.close()
if __name__ == '__main__':

