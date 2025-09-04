import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import random
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, concatenate, Conv2DTranspose, Reshape, MaxPooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Dropout, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Пути к папкам trainX и trainY
trainX_path = os.getcwd() + "\\trainX"
trainY_path = os.getcwd() + "\\trainY"


# Функция для улучшения изображения
def enhance_image(image):
    # Применение фильтра Габора
    kernel_size = 2
    theta = np.pi / 4  # Направление фильтра
    sigma = 3
    lambda_ = 3
    gamma = 0.7
    psi = 0
    kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_, gamma, psi, ktype=cv2.CV_32F)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    # Бинаризация Оцу
    _, binary_image = cv2.threshold(filtered_image, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


# Функция для загрузки изображений из папок trainX и trainY
def load_images(trainX_path, trainY_path, num_examples):
    trainX_images = []
    trainY_images = []
    original_images = []
    desired_width = 128
    desired_height = 128
    count = 0
    for filename in os.listdir(trainX_path):
        count += 1
    if num_examples > count:
        num_examples = count - 1
    # Проход по файлам в папках trainX и trainY
    count = 0
    print("Загрузка данных..... подождите!")
    for filename in os.listdir(trainX_path):
        count += 1
        if count >= num_examples:
            print(count)
            break
        if filename.endswith('.BMP'):
            # Загрузка изображения из папки trainY
            imgX = cv2.imread(os.path.join(trainX_path, filename), cv2.IMREAD_GRAYSCALE)
            imgX = cv2.resize(imgX, (3 * desired_width, 3 * desired_height))
            imgX = enhance_image(imgX)
            imgX = cv2.resize(imgX, (desired_width, desired_height))  # Подгонка размера изображения по необходимости
            # trainX_images.append(imgX)
            # Сопоставление с изображениями из папки trainY
            prefix = filename[:-4]  # Удаление расширения файла (.BMP)
            # for suffix in ['CR', 'Obl', 'Zcut']:
            if filename.endswith('_CR.BMP'):
                imgY_filename = prefix[:-3] + '.BMP'
                # print(f"Файл {imgY_filename} заканчивается на '_CR.BMP'")
            elif filename.endswith('_Obl.BMP'):
                imgY_filename = prefix[:-4] + '.BMP'
                # print(f"Файл {imgY_filename} заканчивается на '_Obl.BMP'")
            elif filename.endswith('_Zcut.BMP'):
                imgY_filename = prefix[:-5] + '.BMP'
                # print(f"Файл {imgY_filename} заканчивается на '_Zcut.BMP'")
            imgY_path = os.path.join(trainY_path, imgY_filename)
            if os.path.exists(imgY_path):
                trainX_images.append(imgX)
                imgY = cv2.imread(imgY_path, cv2.IMREAD_GRAYSCALE)
                imgY = cv2.resize(imgY,
                                  (desired_width, desired_height))  # Подгонка размера изображения по необходимости
                original_images.append(imgY)
                imgY = cv2.resize(imgY, (3 * desired_width, 3 * desired_height))
                imgY = enhance_image(imgY)
                imgY = cv2.resize(imgY, (desired_width, desired_height))
                trainY_images.append(imgY)
                # original_images.append(imgY)
    print("Загрузка завершена!")
    return np.array(trainX_images), np.array(trainY_images), np.array(original_images)


def split_and_norm_dataset(X, Y):
    # Дополнительно: вывод размеров данных
    print("Размер X:", X.shape)
    print("Размер Y:", Y.shape)

    # Подготавливаем набор данных для обучения, разделяем данные на обучающую, тестовую и проверочную выборки
    split_сoeff = 0.8
    X_train = X[:int(split_сoeff * (len(X) - 1))].astype('float32') / 255.0
    Y_train = Y[:int(split_сoeff * (len(Y) - 1))].astype('float32') / 255.0

    X_valid = X[int(split_сoeff * (len(X) - 1)):int(0.9 * (len(X) - 1))].astype('float32') / 255.0
    Y_valid = Y[int(split_сoeff * (len(X) - 1)):int(0.9 * (len(X) - 1))].astype('float32') / 255.0

    X_test = X[int(0.9 * (len(X))):].astype('float32') / 255.0
    Y_test = Y[int(0.9 * (len(Y))):].astype('float32') / 255.0

    print(f"Размер обучающей выборки X_train: {X_train.shape}")
    print(f"Размер проверочной выборки X_valid: {X_valid.shape}")
    print(f"Размер тестовой выборки X_test: {X_test.shape}")
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


# Функция отображения случайных изображений
def plot_random_pair(X, Y, Z, txt1=" ", txt2=" ", txt3=" "):
    index = random.randint(0, len(trainX) - 1)  # Замените на нужный вам индекс
    # Отображение изображений
    plt.figure(figsize=(10, 5))

    # Изображение из X
    plt.subplot(1, 3, 1)
    plt.imshow(X[index], cmap='gray')
    plt.title(txt1)
    plt.axis('off')

    # Изображение из Y
    plt.subplot(1, 3, 2)
    plt.imshow(Y[index], cmap='gray')
    plt.title(txt2)
    plt.axis('off')

    # Изображения из Z
    plt.subplot(1, 3, 3)
    plt.imshow(Z[index], cmap='gray')
    plt.title(txt3)
    plt.axis('off')

    plt.show()

# Функция выводит на экран заданное количество строк из 3 столбцов для сравнения
def plot_images_from_dataset1(data1, data2, data3, num_examples=3, text1="", text2="", text3=""):
    # Количество примеров для отображения num_examples
    # Отображение изображений
    plt.figure(figsize=(5 * num_examples, 5))

    for i in range(num_examples):
        j = random.randint(0, len(data1) - 1)
        # Изображение из data1
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(data1[j], cmap='gray')
        plt.title(text1 + str(i + 1))
        plt.axis('off')

        # Изображение из data2
        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(data2[j], cmap='gray')
        plt.title(text2 + str(i + 1))
        plt.axis('off')

        # Изображение из data3
        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(data3[j], cmap='gray')
        plt.title(text3 + str(i + 1))
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Функция выводит на экран заданное количество строк из 3 столбцов для сравнения изображений
def plot_images(img1, img2, img3, text1="", text2="", text3=""):
    # Количество примеров для отображения num_examples
    # Отображение изображений
    plt.figure(figsize=(15, 5))

    for i in range(num_examples):
        # Изображение img1
        plt.subplot(num_examples, 3, 1)
        plt.imshow(img1, cmap='gray')
        plt.title(text1 + str(i + 1))
        plt.axis('off')

        # Изображение img2
        plt.subplot(num_examples, 3, 2)
        plt.imshow(img2, cmap='gray')
        plt.title(text2 + str(i + 1))
        plt.axis('off')

        # Изображение img3
        plt.subplot(num_examples, 3, 3)
        plt.imshow(img3, cmap='gray')
        plt.title(text3 + str(i + 1))
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Блок кодировщика
def encoder_block(input_layer, filters, kerner_size=3):
    conv = Conv2D(filters, kerner_size, activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv)
    conv = Conv2D(filters, kerner_size, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv)
    conv2 = Dropout(0.2)(conv2)
    # concat = tf.keras.layers.Concatenate()([conv1, conv2])
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    return conv2, pool


# Блок декодера
def decoder_block(input_layer, concat_layer, filters, kerner_size):
    up = Conv2D(filters, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(input_layer))
    merge = concatenate([concat_layer, up], axis=3)
    conv = Conv2D(filters, kerner_size, activation='relu', padding='same')(merge)
    conv = BatchNormalization()(conv)
    conv = Conv2D(filters, kerner_size, activation='relu', padding='same')(conv)
    conv = BatchNormalization()(conv)
    return conv


# Функция которая задаем модель с архитектурой U-NET
def build_unet(input_shape):
    # Входной слой
    inputs_layer = Input(input_shape)

    # Кодировщик
    encoder1, pool1 = encoder_block(inputs_layer, 64, 4)
    encoder2, pool2 = encoder_block(pool1, 128, 3)
    encoder3, pool3 = encoder_block(pool2, 256, 3)
    encoder4, pool4 = encoder_block(pool3, 512, 3)

    encoder5, _ = encoder_block(pool4, 1024)

    # Декодер
    decoder1 = decoder_block(encoder5, encoder4, 512, 3)
    decoder2 = decoder_block(decoder1, encoder3, 256, 3)
    decoder3 = decoder_block(decoder2, encoder2, 128, 3)
    decoder4 = decoder_block(decoder3, encoder1, 64, 4)

    # Выходной слой
    output_layer = Conv2D(1, 1, activation='sigmoid')(decoder4)

    # Создание модели U-Net
    model = Model(inputs=inputs_layer, outputs=output_layer)
    tf.keras.utils.plot_model(model, show_shapes=True)

    return model


# Функция для построения и сохранения графиков точности и потерь
def plot_training_and_save(history, filename):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Точность при обучении')
    plt.plot(epochs, val_acc, 'r', label='Точность при проверке')
    plt.title('Точность при обучении и проверки')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Потери при обучении')
    plt.plot(epochs, val_loss, 'r', label='Потери при проверки')
    plt.title('Потери при обучении и проверки')
    plt.legend()

    # Сохранение графиков в файл
    plt.savefig(filename)
    plt.close()


# Функция обучения сети
def train_model(trainX_path, trainY_path, EPOCHS=30, BATCHSIZE=32, num_images=15000):
    X, Y, original = load_images(trainX_path, trainY_path, num_images)
    X_train, Y_train, X_valid, Y_valid, _, _ = split_and_norm_dataset(X, Y)

    # Колбеки для сохранения лучшей модели, ранней остановки и динамического изменения learning rate
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    # Создаем нашу модель
    print("[INFO] building model...")
    input_shape = (128, 128, 1)
    opt = Adam(learning_rate=1e-3)
    model = build_unet(input_shape)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_valid, Y_valid),
        epochs=EPOCHS,
        steps_per_epoch=len(X_train) / BATCHSIZE,
        shuffle=True,
        batch_size=BATCHSIZE,
        callbacks=[checkpoint, early_stopping])

    # Построение графиков точности и потерь
    plot_training_and_save(history, "traning_graph.jpg")
    # Сохранение модели
    model.save("fun_model.h5")


# Функция предсказания для загруженной модели
def predict_image(model, image_path, save_path):
    # Загрузка изображения и предобработка
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (128, 128))
    image_resized = enhance_image(image_resized) / 255.0
    image_array = np.expand_dims(image_resized, axis=0)
    # Предсказание
    prediction = model.predict(image_array)

    # Инвертирование изображения
    prediction *= 255  # Масштабирование обратно в диапазон от 0 до 255
    prediction = prediction.astype(np.uint8)  # Приведение к типу uint8
    image = prediction[0, :, :, 0]
    # Отображение изображения
    cv2.imshow('Исправленное изображение', image)
    # Сохранение изображения
    print(f"Сохранение файла по пути: {save_path}")
    cv2.imwrite(save_path, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return prediction


def predict_from_data(model, img):
    image_array = np.expand_dims(img, axis=0)
    # Предсказание с использованием модели
    predicted_image = model.predict(image_array)
    return predicted_image


def plot_images_from_dataset(data1, data2, data3, num_examples=3, text1="", text2="", text3=""):
    # Количество примеров для отображения num_examples
    # Отображение изображений
    plt.figure(figsize=(5 * num_examples, 5))

    for i in range(num_examples):
        j = random.randint(0, len(data1) - 1)
        # Изображение из data1
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(data1[j][:, :], cmap='gray')
        plt.title(text1 + str(i + 1))
        plt.axis('off')

        # Изображение из data2
        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(data2[j][:, :], cmap='gray')
        plt.title(text2 + str(i + 1))
        plt.axis('off')

        # Изображение из data3
        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(data3[j][:, :, 0], cmap='gray')
        plt.title(text3 + str(i + 1))
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def main(args):
    # Если выбран режим обучения то выполняем обучение, лучшая модель сохраняется с именем best_model.h5
    if args.mode == 'train':
        train_model(trainX_path, trainY_path, 30, 128, 9000)
    #
    elif args.mode == 'repair':
        model = load_model(args.model_path)
        # Получаем имя файла из полного пути
        file_name = os.path.basename(args.image_path)
        # Формируем новый путь с префиксом "repair_" в имени файла
        save_path = os.path.join(os.path.dirname(args.image_path), "repair_" + file_name)
        prediction = predict_image(model, args.image_path, save_path)

    # если выбран режим тестирования берется 3 случайных примера из тестовой выборки
    elif args.mode == 'test':
        num_images = 15000
        orig = []
        damaged = []
        repair = []
        model = load_model(args.model_path)
        X, Y, original = load_images(trainX_path, trainY_path, num_images)
        _, _, _, _, X_test, Y_test = split_and_norm_dataset(X, Y)
        for i in range(3):
            j = random.randint(0, len(X_test) - 1)
            if i == 0:
                orig = np.expand_dims(Y_test[j], axis=0)
                damaged = np.expand_dims(X_test[j], axis=0)
                repair = predict_from_data(model, Y_test[j])
            else:
                arr = np.expand_dims(Y_test[j], axis=0)
                orig = np.concatenate((orig, arr), axis=0)
                arr = np.expand_dims(X_test[j], axis=0)
                damaged = np.concatenate((damaged, arr), axis=0)
                repair = np.concatenate((repair, predict_from_data(model, Y_test[j])), axis=0)
        print(
            f"orig shape: {np.array(orig).shape} damaged shape {np.array(damaged).shape}, repair.shape {np.array(repair).shape}")
        plot_images_from_dataset(orig, damaged, repair, 3, "оригинальный фильтрованный", "поврежденный исходное",
                                 "исправленный")
    else:
        print("Invalid mode. Please specify either 'train' or 'predict'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or predict using a model.')
    parser.add_argument('mode', choices=['test', 'show', 'train', 'repair'], help='Mode: train or predict')
    parser.add_argument('--model_path', default='best_model.h5', help='Путь к модели (в режиме предсказания)')
    parser.add_argument('--image_path', default='example.bmp', help='Путь к файлу изображений (в режиме предсказания)')
    args = parser.parse_args()
    main(args)