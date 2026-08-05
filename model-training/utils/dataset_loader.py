import tensorflow as tf

from config import (
    DATASET_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    SEED
)


def load_datasets():

    print("Loading dataset...")


    # =========================
    # Training Dataset
    # =========================

    train_dataset = tf.keras.utils.image_dataset_from_directory(

        DATASET_PATH,

        validation_split=VALIDATION_SPLIT,

        subset="training",

        seed=SEED,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE

    )


    # =========================
    # Validation Dataset
    # =========================

    validation_dataset = tf.keras.utils.image_dataset_from_directory(

        DATASET_PATH,

        validation_split=VALIDATION_SPLIT,

        subset="validation",

        seed=SEED,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE

    )


    # =========================
    # Class Names
    # =========================

    class_names = train_dataset.class_names


    print("\nClasses:")
    print(class_names)


    # =========================
    # Performance Optimization
    # =========================

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = (
    train_dataset
    .shuffle(500)
    .prefetch(AUTOTUNE)
        )


    validation_dataset = (
    validation_dataset
    .prefetch(AUTOTUNE)
        )


    return (
        train_dataset,
        validation_dataset,
        class_names
    )