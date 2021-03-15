import math
import cv2
import os
import glob
import tensorflow as tf
import numpy as np
from PIL import Image

WIDTH = 300
CROPDIR = 'crop'
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE


def img_split(filename, save=True):
    # load image
    img = cv2.imread(filename, 0)

    # detect brightness and morphological improvement
    if img.mean() < 3:
        ret, binary = cv2.threshold(img, 0, 127, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, binary = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    erd = cv2.erode(binary, None, iterations=3)
    dil = cv2.dilate(erd, None, iterations=3)

    # edge detection by cv2.findContours
    contours = cv2.findContours(dil, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
    cntLst, rectLst = [], []  # cntLst for edge info, rectLst for position info (x, y, width, height)
    for i in contours:
        # remove the specks in images
        if cv2.contourArea(i) >= 10000:
            chgd = i.reshape(1, -1)[0]
            # get cell's position in image and delete incomplete cells near edge
            rect = cv2.boundingRect(i)
            if rect[0] < 24 or rect[1] < 24 or rect[0] + rect[2] > 1000 or rect[1] + rect[3] > 1000:
                tmp = 0
                for j in chgd:
                    if j < 24 or j > 1000:
                        tmp += 1
                if tmp >= 15:
                    continue
            cntLst.append(i)
            rectLst.append(rect)

    if save:
        # mkdir for crop image
        if not os.path.exists(CROPDIR):
            os.mkdir(CROPDIR)

        for num, i in enumerate(rectLst):
            center_x, center_y = int(i[1] + i[3] / 2), int(i[0] + i[2] / 2)
            if center_x < WIDTH / 2:
                center_x = WIDTH / 2
            if center_x > 1024 - WIDTH / 2:
                center_x = 1024 - WIDTH / 2
            if center_y < WIDTH / 2:
                center_y = WIDTH / 2
            if center_y > 1024 - WIDTH / 2:
                center_y = 1024 - WIDTH / 2

            ROI = img[int(center_x - WIDTH / 2):int(center_x + WIDTH / 2), int(center_y - WIDTH / 2):int(center_y + WIDTH / 2)]
            cv2.imwrite(f'{CROPDIR}/{filename}-{num}.png', ROI)
    return cntLst, rectLst


def _load_image(path):
    """
        convert image path to tensor for meachine learning
    :param:
        (list)path: for imge path
    :return:
        (tensor)image_tensor: a tensor for the image
    """
    image_raw = tf.io.read_file(path)
    image_tensor = tf.image.decode_png(image_raw, channels=1, dtype=tf.dtypes.uint16)
    image_tensor = tf.image.resize(image_tensor, [300, 300])
    return image_tensor


def model_predict(filename):
    # get all crop image path
    all_image_path0 = list(glob.glob(f"{CROPDIR}/{filename}*.png"))
    all_image_path0 = [str(i) for i in all_image_path0]

    # load image into tensor
    path_ds0 = tf.data.Dataset.from_tensor_slices(all_image_path0)
    image_ds0 = path_ds0.map(_load_image)

    # a fake label just for data format
    label_ds0 = tf.data.Dataset.from_tensor_slices(tf.cast([0 for i in all_image_path0], tf.int64))

    # zip dataset from image tensor and label for dataset format
    image_label_ds0 = tf.data.Dataset.zip((image_ds0, label_ds0))

    # set BATCH_SIZE and PREFETCH
    image_label_ds0 = image_label_ds0.batch(BATCH_SIZE)
    image_label_ds0 = image_label_ds0.prefetch(buffer_size=AUTOTUNE)

    # load model from program file and predict
    model = tf.keras.models.load_model('model/model_1.h5')
    y = [i[0] for i in model.predict(image_label_ds0)]
    x = [math.log(1 / (1 - i)) for i in y]
    res = [1 / (1 + math.exp(-0.5 * i)) for i in x]
    return y


def _transfer_16bit_to_8bit(image_16bit):
    # image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit


def output_vis(filename, cntLst, rectLst, res, output="."):
    img = cv2.imread(filename, -1)
    # img = _equalizeHist(img)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    res2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(res2, cntLst, -1, (0, 0, 65535), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX

    assert len(res) == len(rectLst)
    for num, i in enumerate(rectLst):
        center_x, center_y = int(i[1] + i[3] / 2), int(i[0] + i[2] / 2)
        left_up_x, left_up_y = int(center_x - WIDTH / 2), int(center_y - WIDTH / 2)
        right_down_x, right_down_y = int(center_x + WIDTH / 2), int(center_y + WIDTH / 2)
        cv2.rectangle(res2, (left_up_y, left_up_x), (left_up_y + 180, left_up_x + 40), (0, 65535, 0), -1)
        cv2.putText(res2, f'{num}: {res[num]:2.4f}', (left_up_y + 10, left_up_x + 30), font, 1, (0, 0, 65535), 2,
                    cv2.LINE_AA)
        cv2.rectangle(res2, (left_up_y, left_up_x), (right_down_y, right_down_x), (0, 65535, 0), 2)
    res3 = _transfer_16bit_to_8bit(res2)
    cv2.imwrite(f'{output}/{filename}-output.png', res3)


def open_image(filename):
    image = Image.open(f'{filename}-output.png')
    image.show()


def test(filename, output=".", open=True):
    cnt, rect = img_split(filename)
    result = model_predict(filename)
    output_vis(filename, cnt, rect, result, output=output)
    if open:
        open_image(filename)


if __name__ == '__main__':
    test("190716_U2OSPFA_H2BGFP_DAPI_008_R3D_D3D-0011.tif")

    # Batch_process
    # file_name = "20190716_U2OSPFA_H2BGFP_DAPI/*.tif"
    # files = glob.glob(file_name)
    # for i in list(files):
    #     test(i, output="output", open=False)

