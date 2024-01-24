#!/usr/bin/env python
# coding: utf-8

# In[101]:


import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense,Input,Multiply
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense


# In[102]:


import tensorflow as tf
from tensorflow.keras import models, layers,regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np


# In[117]:


EPOCHS = 35
BATCH_SIZE = 32
CHANNELS=3
DROPOUT_RATE = 0.35
L2_LAMBDA = 0.001


# In[118]:


IMAGE_SIZE, IMAGE_SIZE = 224, 224
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)


# In[119]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "E:\\Download\\MSTAR-10-Classes\\train",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[120]:


class_names = dataset.class_names
class_names


# In[121]:


for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# In[122]:


# Calculate the number of batches in the dataset
for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())


# In[123]:


# Calculate the number of batches for each set
train_batches = int(num_batches * 0.7)
val_batches = int(num_batches * 0.15)
test_batches = num_batches - train_batches - val_batches


# In[124]:


# Split the dataset into train, validation and test
train_ds = dataset.take(train_batches)
remaining_ds = dataset.skip(train_batches)
val_ds = remaining_ds.take(val_batches)
test_ds = remaining_ds.skip(val_batches)


# In[125]:


# Cache, Shuffle, and Prefetch the Dataset
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[126]:


# Add more diverse data augmentation
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomContrast(0.2)
])


# In[127]:


# Resize and rescale images
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255)
])


# In[128]:


model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, kernel_size=(3,3), activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA), input_shape=(img_width,img_height, CHANNELS)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(DROPOUT_RATE),
    layers.Conv2D(64, kernel_size=(3,3), activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(DROPOUT_RATE),
    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA)),
    layers.Dropout(DROPOUT_RATE),
    layers.Dense(len(class_names), activation='softmax'),
])


# In[129]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[130]:


# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)


# In[132]:


# Fit the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)


# In[133]:


# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print("\nTest Accuracy:", test_acc)


# In[136]:


# Evaluate the model on the training dataset
train_loss, train_acc = model.evaluate(train_ds, verbose=2)
print("\nTrain Accuracy:", train_acc)


# In[135]:


# Evaluate the model on the validation dataset
val_loss, val_acc = model.evaluate(val_ds, verbose=2)
print("\nValidation Accuracy:", val_acc)


# In[137]:



# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[138]:



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
# Calculate predictions on the test dataset
y_true = []
y_pred = []

for images_batch, labels_batch in test_ds:
    batch_predictions = model.predict(images_batch)
    batch_predictions = np.argmax(batch_predictions, axis=1)
    y_pred.extend(batch_predictions)
    y_true.extend(labels_batch.numpy())

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate and print the metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro',zero_division=1)
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[139]:


# Print the classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# In[140]:


# Plot the confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[141]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):

    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])


# In[100]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get the model's predictions on the test set
predictions = model.predict(test_ds)

# Convert the predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Get the true class labels
true_classes = []
for images, labels in test_ds:
    true_classes.extend(labels.numpy())
true_classes = np.array(true_classes)

# Compute the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




