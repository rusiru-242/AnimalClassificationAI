import os
import json
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from utils.dataset_loader import load_datasets

from config import (
    IMAGE_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    CLASS_NAMES_PATH,
    ACCURACY_GRAPH,
    LOSS_GRAPH,
    OUTPUT_DIR
)


# ==============================
# Create Directories
# ==============================

os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# Load Dataset
# ==============================

train_dataset, validation_dataset, class_names = load_datasets()

print("Reading one batch...")

for images, labels in train_dataset.take(1):
    print(images.shape)
    print(labels.shape)

print("Dataset OK")



NUM_CLASSES = len(class_names)

print("\nClasses:")
print(class_names)


# ==============================
# Preprocessing
# ==============================

train_dataset = train_dataset.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


validation_dataset = validation_dataset.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)



# ==============================
# Data Augmentation
# ==============================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.15),

    layers.RandomZoom(0.15),

    layers.RandomContrast(0.15),

    layers.RandomBrightness(0.2),

    layers.RandomTranslation(
        0.1,
        0.1
    )

])



# ==============================
# MobileNetV2 Base Model
# ==============================

base_model = MobileNetV2(

    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),

    include_top=False,

    weights="imagenet"

)


# Freeze pretrained layers

base_model.trainable = True


for layer in base_model.layers[:-50]:

    layer.trainable = False



# ==============================
# Build Functional Model
# ==============================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


x = data_augmentation(inputs)


x = base_model(
    x,
    training=False
)


x = layers.GlobalAveragePooling2D()(x)


x = layers.Dropout(0.5)(x)


x = layers.Dense(
    128,
    activation="relu"
)(x)


x = layers.Dropout(0.2)(x)


outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)



model = Model(
    inputs,
    outputs,
    name="SkinToneAI"
)



# ==============================
# Compile Model
# ==============================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)


model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




# ==============================
# Callbacks
# ==============================


early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.2,

    patience=3,

    min_lr=0.0000001

)


base_model.trainable = True


for layer in base_model.layers[:-30]:
    layer.trainable = False



checkpoint = ModelCheckpoint(

    filepath=MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)






# ==============================
# Train Model
# ==============================
print("\n==============================")
print("Testing first batch...")
print("==============================")

# images, labels = next(iter(train_dataset))

for images, labels in train_dataset.take(1):
    print("Images Shape:", images.shape)
    print("Labels Shape:", labels.shape)

print("Images Shape :", images.shape)
print("Labels Shape :", labels.shape)

print("First 10 Labels :", labels[:10].numpy())


print("Patience =", early_stop.patience)
print("Epochs:", EPOCHS)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[

    early_stop,

    #educe_lr,

    checkpoint

],
    verbose=1
)


# ==============================
# Save Class Names
# ==============================

with open(
    CLASS_NAMES_PATH,
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )



# ==============================
# Accuracy Graph
# ==============================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"]
)

plt.plot(
    history.history["val_accuracy"]
)

plt.title(
    "Model Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend(
    [
        "Training",
        "Validation"
    ]
)

plt.savefig(
    ACCURACY_GRAPH
)

plt.close()



# ==============================
# Loss Graph
# ==============================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"]
)

plt.plot(
    history.history["val_loss"]
)

plt.title(
    "Model Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend(
    [
        "Training",
        "Validation"
    ]
)

plt.savefig(
    LOSS_GRAPH
)

plt.close()



print("\n==============================")
print("Training Completed Successfully")
print("==============================")

print(
    "Model Saved:",
    MODEL_PATH
)

print(
    "Classes Saved:",
    CLASS_NAMES_PATH
)

print(
    "Accuracy Graph:",
    ACCURACY_GRAPH
)

print(
    "Loss Graph:",
    LOSS_GRAPH
)


plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("outputs/accuracy.png")
plt.close()


plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("outputs/loss.png")
plt.close()