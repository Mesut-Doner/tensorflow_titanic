# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import numpy as np
import functools

train_path=tf.keras.utils.get_file('train.csv','https://storage.googleapis.com/tf-datasets/titanic/train.csv')
test_path=tf.keras.utils.get_file('eval.csv','https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

train_data=tf.data.experimental.make_csv_dataset(train_path,batch_size=5,label_name='survived',na_value='?',num_epochs=1,ignore_errors='True')
test_data=tf.data.experimental.make_csv_dataset(train_path,batch_size=5,label_name='survived',na_value='?',num_epochs=1,ignore_errors='True')

# Numerik olan Feature-Label-TargetName 'ları Paketleme Sınıfı
class Pack():
    def __init__(self,names):
        self.names = names
    def __call__(self,features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features["numeric"] = numeric_features
        return features, labels


NUMERIC_FEATURES = ['age','n_siblings_spouses','parch','fare']
train_packed=train_data.map(Pack(NUMERIC_FEATURES))
test_packed=test_data.map(Pack(NUMERIC_FEATURES))

# NORMALIZATION

def normalized(data,mean,std):
    return (data-mean)/std

ozet_bilgi=pd.read_csv(train_path)[NUMERIC_FEATURES].describe()
print(ozet_bilgi)
mean=np.array(ozet_bilgi.T['mean'])
std=np.array(ozet_bilgi.T['std'])
# Fonksiyonu Tek Parametre Alır Hale Getirmek
normalize=functools.partial(normalized,mean=mean,std=std)
numeric_column=tf.feature_column.numeric_column('numeric',normalizer_fn=normalize,shape=[len(NUMERIC_FEATURES)])
numeric_columns=[numeric_column]

# Kategorik Columnları Ayırmak
categories={
    'sex' : ['male','female'],
    'class' : ['First','Second','Third'],
    'deck' : ['A','B','C','D','E','F','G','H','I','J'],
    'embark_town' : ['Cherbourg','Southampton','Queenstown'],
    'alone' : ['y','n']
    }

categorical_columns=[]
for feature,vocabulary in categories.items():
    categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(key=feature,vocabulary_list=vocabulary)
    categorical_columns.append(tf.feature_column.indicator_column(categorical_column))


preprocessing_layer=tf.keras.layers.DenseFeatures(numeric_columns+categorical_columns)


model=tf.keras.Sequential(
    [
     preprocessing_layer,
     tf.keras.layers.Dense(128,activation='relu'),
     tf.keras.layers.Dense(128,activation='relu'),
     tf.keras.layers.Dense(1)
    ]
    )

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
    )
train_packed=train_packed.shuffle(1000)
model.fit(train_packed,epochs=10)



predictions=model.predict(test_packed)
# tf.print(predictions)
for prediction,survived in zip ( predictions[:10], list(test_packed)[0][1][:10]):
    prediction=tf.sigmoid(prediction).numpy()[0]
    survived=bool(survived)
    print('%4.2f'%prediction,'Tahmin:', ('KALDI' if survived else 'ÖLDÜ'))

loss,accuracy=model.evaluate(test_packed)
print('Kayıp:',loss)
print('İsabet:',accuracy)

















