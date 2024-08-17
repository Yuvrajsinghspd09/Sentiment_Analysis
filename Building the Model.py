from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense,Embedding

model = Sequential([Embedding(10000,16,input_length=max_len),
                    Flatten(),
                    Dense(1,activation = 'sigmoid')])


model.build(input_shape=(None, max_len))
model.summary()
