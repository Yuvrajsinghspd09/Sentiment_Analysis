#Preprocessing the Data

from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len=100

x_train = pad_sequences(x_train,maxlen=max_len)
x_test = pad_sequences(x_test,maxlen=max_len)
