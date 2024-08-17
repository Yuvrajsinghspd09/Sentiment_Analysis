from tqdm.keras import TqdmCallback

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=32,validation_split=0.2,callbacks=[TqdmCallback()])


# Evaluate the model on test data
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
