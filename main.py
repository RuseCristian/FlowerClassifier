import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk


matplotlib.use("TkAgg")


def plot_and_save_history(history):

    plt.figure()
    epochs = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs, history.history['loss'], 'r', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'r--', label='Validation loss')
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'b--', label='Validation accuracy')
    plt.title('Training and validation loss and accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig('combined_plot.png')


model_path = Path("flower_classifier_backup.keras")
if model_path.is_file():
    model = tf.keras.models.load_model('flower_classifier_backup22.keras')
    base_dir = 'flowers_train/'
    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_dir,
        seed=123,
        batch_size=32,
        image_size=(180, 180))
    flower_names = train_ds.class_names


    def classify_images(image_path):
        input_image = load_img(image_path, target_size=(180, 180))
        input_image_array = img_to_array(input_image)
        input_image_exp_dim = np.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        percentages = {flower_names[i]: result[i].numpy() * 100 for i in range(len(flower_names))}
        return percentages


    def load_image():
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = image.resize((180, 180), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo

            percentages = classify_images(file_path)
            result_text = "\n".join([f"{flower}: {percent:.2f}%" for flower, percent in percentages.items()])
            result_label.config(text=result_text)

            most_confident_flower = max(percentages, key=percentages.get)
            most_confident_value = percentages[most_confident_flower]
            confident_label.config(text=f"Most Confident: {most_confident_flower} ({most_confident_value:.2f}%)")

    root = tk.Tk()
    root.title("Flower Classifier")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    load_button = ttk.Button(frame, text="Load Image", command=load_image)
    load_button.pack()

    image_label = tk.Label(frame)
    image_label.pack()

    result_label = tk.Label(frame, text="", justify=tk.LEFT)
    result_label.pack()

    confident_label = tk.Label(frame, text="", justify=tk.LEFT, font=("Helvetica", 12, "bold"))
    confident_label.pack()

    root.mainloop()

else:
    train_dir = 'flowers_train/'
    test_dir = 'flowers_test'
    val_dir = 'flowers_val'

    img_size = 180
    batch_size = 32
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs available:")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("Amaraciune... Ai placa integrata, vai ce rusine.")


    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=[img_size, img_size],
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=532,
        image_size=[img_size, img_size],
        batch_size=batch_size
    )


    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=[img_size, img_size],
        batch_size=batch_size
    )

    print("\n\n\n")
    total_count = len(train_ds.file_paths) + len(val_ds.file_paths) + len(test_ds.file_paths)
    print("Training Set share of total dataset: " + str(round(len(train_ds.file_paths)/total_count, 3)) + "%")
    print("Validation Set share of total dataset: " + str(round(len(val_ds.file_paths)/total_count, 3)) + "%")
    print("Test Set share of total dataset: " + str(round(len(test_ds.file_paths)/total_count, 3)) + "%")


    flower_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal", input_shape=(img_size, img_size, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(flower_names))
    ])


    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=[callback]
    )


    test_loss, test_accuracy = model.evaluate(test_ds)
    plot_and_save_history(history)
    print(f'Test accuracy: {round(test_accuracy,3)}')


    y_true = []
    y_pred = []

    for images, labels in test_ds:
        predictions = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))

    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()

    print("Confusion Matrix:")
    print(cm)

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized Confusion Matrix:")
    print(cm_normalized)

    model.save('flower_classifier.keras')