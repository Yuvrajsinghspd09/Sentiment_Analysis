# Loading the Dataset

from tensorflow.keras.datasets import imdb

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=10000)

print("first review:", x_train[0])
print("first review label:",y_train[0])
