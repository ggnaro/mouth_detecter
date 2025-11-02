import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import numpy as np
import PIL
from PIL import ImageOps
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import tensorflow_addons as tfa # need for Ranger optimizer if model training used
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as ss
from retinaface import RetinaFace
from openvino.inference_engine import IECore
import sys

import matplotlib
matplotlib.use('TkAgg',force=True)
print("Switched to:", matplotlib.get_backend())

class openvino_model:
    def laoding_model_path(model_path: str):
        ie = IECore()
        model_name = "segmentation"
        model_xml = os.path.join(model_path, f"{model_name}.xml")
        model_bin = os.path.join(model_path, f"{model_name}.bin")
        print(f"Loading model {model_name} ... ")
        sys.stdout.flush()
        net = ie.read_network(model=model_xml, weights=model_bin)
        print("done!")

        return ie, net

    def inference(ie, net, img_i_orig):
        # layout_shape = net.input_info["input_1"].input_data.shape  # OpenVino model is always with NCHW layout , refer to "input_data.layout"
        # h = layout_shape[2]
        # w = layout_shape[3]
        # img_i = cv2.resize(img_i_orig, (w, h))
        # img_i = img_i.transpose((2, 0, 1))  # image is with HWC, converting into CHW

        img_i = img_i_orig.transpose((2, 0, 1))
        img_i = np.expand_dims(img_i, axis=0)

        data = {}
        data["input_1"] = img_i
        net.input_info["input_1"].precision = "U8"

        output_info = net.outputs["StatefulPartitionedCall/model/up_sampling2d_10/resize/ResizeBilinear"]
        output_info.precision = "FP32"

        sys.stdout.flush()
        exec_net = ie.load_network(network=net, device_name="CPU")
        res = exec_net.infer(inputs=data)
        pre_image = res["StatefulPartitionedCall/model/up_sampling2d_10/resize/ResizeBilinear"][0]

        preimage_transpose = pre_image.transpose((1, 2, 0))  # image is with CHW, converting into HWC for show

        return preimage_transpose, res


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def subimage(image, center, theta, width, height):

   '''
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = 0 if int(center[0] - width/2) < 0 else int(center[0] - width/2)
   y = 0 if int(center[1] - height/2) < 0 else int(center[1] - height/2)

   crop_image = image[ y:y+height, x:x+width]

   return crop_image


def subimage_2(image, center, theta, width, height):

   '''
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = 0 if int(center[0] - width/2) < 0 else int(center[0] - width/2)
   y = 0 if int(center[1] - height/2) < 0 else int(center[1] - height/2)

   x_2 = int(x + width/4)
   y_2 = int(y + height/4)
   height_2 = int(height/2)
   width_2 = int(width/2)

   crop_image = image[ y_2:y_2+height_2, x_2:x_2+width_2]

   return crop_image


def calculate_mouthinside(rect_mouth):
    mouth_area = len(np.column_stack(np.where(rect_mouth > 0)))

    return mouth_area, mouth_area, mouth_area

def calculate_rect(rect_mouth):
    shape = rect_mouth.shape
    area = shape[0] * shape[1]
    lips_area = len(np.column_stack(np.where(rect_mouth > 0)))

    return 1-((lips_area+ 0.0001) / (area+ 0.0001)), area, lips_area

def load_model_inference_singleimage(model_path, img_path, model_limit_imgsize:Tuple[int, int]=(64,64)):
    model = tf.keras.models.load_model(model_path)
    img = load_img(img_path, target_size=model_limit_imgsize)
    img_array = img_to_array(img)
    img_array_dim = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array_dim)

    return predictions


def load_model_inference_singleimage_readed(model_path, img_readen, model_limit_imgsize:Tuple[int, int]=(64,64)):
    model = tf.keras.models.load_model(model_path)
    img_readen2 = cv2.resize(img_readen, model_limit_imgsize)

    # image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    img_array = img_to_array(img_readen2)
    img_array_dim = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array_dim)

    return predictions


def openness_mouth_inside(outputs_detected):
    # outputs_detected = outputs_detected.astype(np.uint8)
    global ration, mask_cropped, rectarea, targetarea
    gray = cv2.cvtColor(outputs_detected, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(outputs_detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ration, mask_cropped, rectarea, targetarea, outputs_detected = None, None, None, None, None
        # ration, mask_cropped, outputs_detected = None, None, None
        pass

    else:
        range_left = []
        range_top = []
        range_right = []
        range_bottom = []
        c_area_boxinfo_dict = {"idx": [],
                               "area": [],
                               "boxinfo": [],
                               "angles": []
                               }

        for j in range(0, len(contours)):    ### contours starting
            x, y, w, h = cv2.boundingRect(contours[j])
            range_left.append(x)
            range_top.append(y)
            range_right.append(x + w)
            range_bottom.append(y + h)

            rect = cv2.minAreaRect(contours[j])
            angle = rect[2]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            c_area_boxinfo_dict["idx"].append(j)
            height = int(rect[1][1])
            width = int(rect[1][0])
            area = height * width

            c_area_boxinfo_dict["area"].append(area)
            c_area_boxinfo_dict["boxinfo"].append(box)
            c_area_boxinfo_dict["angles"].append(angle)

        if len(c_area_boxinfo_dict["idx"]) > 1:
            boxinfo_NO1 = c_area_boxinfo_dict["boxinfo"][
                c_area_boxinfo_dict["area"].index(max(c_area_boxinfo_dict["area"]))]
            angle_1 = c_area_boxinfo_dict["angles"][c_area_boxinfo_dict["area"].index(max(c_area_boxinfo_dict["area"]))]
            copy_of_area = c_area_boxinfo_dict["area"][:]
            copy_of_area.remove(max(copy_of_area))
            boxinfo_NO2 = c_area_boxinfo_dict["boxinfo"][c_area_boxinfo_dict["area"].index(max(copy_of_area))]
            angle_2 = c_area_boxinfo_dict["angles"][c_area_boxinfo_dict["area"].index(max(c_area_boxinfo_dict["area"]))]

            if max(c_area_boxinfo_dict["area"]) / (max(copy_of_area) + 0.001) > 5:
                height_r = max(range_bottom) - min(range_top)
                width_r = max(range_right) - min(range_left)

                x = int(min(range_left))
                y = int(min(range_top))
                heightr = int(height_r)
                widthr = int(width_r)

                x_2 = int(min(range_left) + width_r / 4)
                y_2 = int(min(range_top) + height_r / 4)
                heightr_2 = int(height_r / 2)
                widthr_2 = int(width_r / 2)



                # mask_cropped = outputs_detected[min(range_top): min(range_top) + height_r, min(range_left): min(range_left) + width_r]
                mask_cropped = outputs_detected[y: y + heightr, x: x + widthr]

                # mask_cropped = outputs_detected[y_2: y_2 + heightr_2,x_2: x_2 + widthr_2]
                # cv2.imwrite("./testcropped.png", mask_cropped)
                cv2.rectangle(outputs_detected, (min(range_left), min(range_top)),
                              (max(range_right), max(range_bottom)), (128, 0, 128), 1)
                ration, rectarea, targetarea = calculate_mouthinside(mask_cropped)

            elif max(c_area_boxinfo_dict["area"]) / (max(copy_of_area) + 0.001) < 5 and max(c_area_boxinfo_dict["area"]) / (max(copy_of_area) + 0.001) >1:

                height_r = max(range_bottom) - min(range_top)
                width_r = max(range_right) - min(range_left)

                x = int(min(range_left))
                y = int(min(range_top))
                heightr = int(height_r)
                widthr = int(width_r)

                x_2 = int(min(range_left) + width_r / 4)
                y_2 = int(min(range_top) + height_r / 4)
                heightr_2 = int(height_r / 2)
                widthr_2 = int(width_r / 2)

            # mask_cropped = outputs_detected[min(range_top): min(range_top) + height_r, min(range_left): min(range_left) + width_r]
                mask_cropped = outputs_detected[y: y + heightr,x: x + widthr]
                # mask_cropped = outputs_detected[y_2: y_2 + heightr_2, x_2: x_2 + widthr_2]

                cv2.rectangle(outputs_detected, (min(range_left), min(range_top)),
                              (max(range_right), max(range_bottom)), (0, 255, 255), 1)
                ration, rectarea, targetarea = calculate_mouthinside(mask_cropped)

            else:
                # cv2.rectangle(outputs_detected, (min(range_left), min(range_top)), (max(range_right), max(range_bottom)),(255, 0, 255), 2)
                height_r = max(range_bottom) - min(range_top)
                width_r = max(range_right) - min(range_left)

                x = int(min(range_left))
                y = int(min(range_top))
                heightr = int(height_r)
                widthr = int(width_r)

                x_2 = int(min(range_left) + width_r / 4)
                y_2 = int(min(range_top) + height_r / 4)
                heightr_2 = int(height_r / 2)
                widthr_2 = int(width_r / 2)

                # mask_cropped = outputs_detected[min(range_top): min(range_top) + height_r, min(range_left): min(range_left) + width_r]
                mask_cropped = outputs_detected[y: y+ heightr , x: x + widthr]
                # mask_cropped = outputs_detected[y_2: y_2 + heightr_2,x_2: x_2 + widthr_2]

                cv2.rectangle(outputs_detected, (min(range_left), min(range_top)),
                              (max(range_right), max(range_bottom)), (255, 0, 255), 1)
                ration, rectarea, targetarea = calculate_mouthinside(mask_cropped)


        else:
            if max(c_area_boxinfo_dict["area"]) < 2:
                ration, mask_cropped, rectarea, targetarea = None, None, None, None
                print("NO RESULTS", j)
                pass

            else:

                boxinfo_NO1 = c_area_boxinfo_dict["boxinfo"][
                    c_area_boxinfo_dict["area"].index(max(c_area_boxinfo_dict["area"]))]

                angle_1 = c_area_boxinfo_dict["angles"][
                    c_area_boxinfo_dict["area"].index(max(c_area_boxinfo_dict["area"]))]
                idx = c_area_boxinfo_dict["area"].index(max(c_area_boxinfo_dict["area"]))
                rect = cv2.minAreaRect(contours[idx])
                height = int(rect[1][1])
                width = int(rect[1][0])
                # cv2.drawContours(outputs_detected, [boxinfo_NO1], 0, (0, 255, 0), 2)

                center = rect[0]
                mask_cropped = subimage(image=outputs_detected, center=center, theta=angle_1, width=width, height=height)
                cv2.drawContours(outputs_detected, [boxinfo_NO1], 0, (0, 255, 0), 1)

                ration, rectarea, targetarea = calculate_mouthinside(mask_cropped)

    return ration, rectarea, targetarea, mask_cropped, outputs_detected


def inference_single_image_areapixel_mouthinside_readed(image_dir, modelformat="TF", activation_func = "sigmoid"):

    global prediction, overall_pixels, openness, rectarea, targetarea, cropped_mouth, outputs_detected

    if modelformat == "TF":
        prediction = load_model_inference_singleimage_readed("./model/mouthinside/tensorflow", image_dir, (64, 64))
    elif modelformat == "OP":
        iecore, net = openvino_model.laoding_model_path("./model/mouthinside/openvino")
        resized_img = cv2.resize(image_dir, (64, 64))
        prediction, data = openvino_model.inference(iecore, net, resized_img)

    # pixels_intervals_picpath = "D:\\segmentation_mouth\\microsoft_seg_0630\\speculate_results\\pixels_intervals\\"

    if activation_func == "sigmoid":
        x1 = np.zeros((64, 64), dtype=np.uint8)
        x3 = np.zeros((64, 64), dtype=np.uint8)
        temp = np.zeros((64, 64), dtype=np.uint8)

        target_predict = sigmoid(prediction[:, :, 1])
        target_predict[target_predict > 0.4] = 1
        target_predict[target_predict <= 0.4] = 0  ### PROBABILITY

        temp1 = np.maximum(temp, target_predict)  # output activation & filter those minus, below 0
        sigmoid_pre_uint8 = cv2.normalize(temp1, None, 0, 255, cv2.NORM_MINMAX,
                                          cv2.CV_8U)  # transer those activated value into normalized [0,255] intervals
        pretend_img = cv2.merge((sigmoid_pre_uint8, x1, x3))  # combined 3 channels, so that digest into cvtocolor

        sigmoid_pre_uint8[sigmoid_pre_uint8 > 0] = 255
        overall_pixels = len(np.column_stack(np.where(sigmoid_pre_uint8 == 255)))

        openness, rectarea, targetarea, cropped_mouth, outputs_detected = openness_mouth_inside(pretend_img)

    elif activation_func == "linear":
        x1 = np.zeros((64, 64), dtype=np.uint8)
        x3 = np.zeros((64, 64), dtype=np.uint8)
        temp = np.zeros((64, 64), dtype=np.uint8)

        temp1 = np.maximum(temp, prediction[0][:, :, 1])

        test = cv2.normalize(prediction[0][:, :, 1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # hist_1 = cv2.calcHist([test], [0], None, [256], [1, 256])
        # plt.plot(hist_1)
        # plt.xlim([1, 256])
        # # plt.show()
        # # plt.pause(2)
        # plt.savefig(pixels_intervals_picpath + "hist_1-" + os.path.basename(image_dir))
        # plt.close()

        # temp1[temp1 < 2] = 0
        temp1[temp1 > 0] = 255
        temp1_uint8 = temp1.astype(np.uint8)

        pretend_img = cv2.merge((temp1_uint8, x1, x3))
        overall_pixels = len(np.column_stack(np.where(temp1_uint8 == 255)))

        openness, rectarea, targetarea, cropped_mouth, outputs_detected = openness_mouth_inside(pretend_img)

    elif activation_func == "check":

        x1 = np.zeros((64, 64), dtype=np.uint8)
        x3 = np.zeros((64, 64), dtype=np.uint8)
        temp = np.zeros((64, 64), dtype=np.uint8)

        original_output = prediction[0][:, :, 1]

        temp1 = np.maximum(temp, prediction[0][:, :, 1])


    elif activation_func == "CDF":

        temp = np.zeros((64, 64), dtype=np.float32)
        cdf_pre = np.maximum(temp, ss.norm.cdf(prediction[0][:, :, 1]))
        cdf_pre_uint8 = cv2.normalize(cdf_pre, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        copy_cdf = cdf_pre_uint8.copy()
        copy_cdf[copy_cdf > 0] = 1
        overall_pixels = len(np.column_stack(np.where(copy_cdf == 1)))

        openness, rectarea, targetarea, cropped_mouth, outputs_detected = openness_mouth_inside(cdf_pre_uint8)

    return prediction, overall_pixels, openness, rectarea, targetarea, cropped_mouth, outputs_detected


def inference_whole_image_retina_openvino(img_path):
    ori_img = mpimg.imread(img_path)
    frame = ori_img.copy()
    none_image = np.zeros_like(ori_img)
    frame_seg = none_image.copy()

    #retinaface = RetinaFace()
    #resp = retinaface.predict(frame)
    resp = RetinaFace.detect_faces(img_path)
    if resp:
        for face in resp:
            bbox = resp[face]['facial_area']
            # bbox = []
            # x = face['x1']
            # bbox.append(x)
            # y = face['y1']
            # bbox.append(y)
            # x2 = face['x2']
            # w = face['x2'] - face['x1']
            # bbox.append(w)
            # y2 = face['y2']
            # h = face['y2'] - face['y1']
            # bbox.append(h)
            print(bbox)
            w = bbox[1] - bbox[0]
            h = bbox[3] - bbox[2]
            x = bbox[0]
            y = bbox[2]
            x2 = bbox[1]
            y2 = bbox[3]
            # crop_img = frame[y:y2, x:x2]
            crop_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # crop_img = frame[np.int(bbox[1]):np.int(bbox[3]), np.int(bbox[0]):np.int(bbox[2])]

            noneimage = np.zeros_like(crop_img)

            predictions, overall_pixels, openness, rectarea, targetarea, cropped_mouth, outputs_detected = \
                inference_single_image_areapixel_mouthinside_readed(crop_img, modelformat="OP", activation_func="sigmoid")

            if openness == None or cropped_mouth.any() == False or outputs_detected.any() == False:
                openness = 0
                cropped_mouth = noneimage
                outputs_detected = noneimage

            outputs_detected_onpic = cv2.resize(outputs_detected, ( w, h ))
            temp = np.zeros((64, 64), dtype=np.uint8)

            target_predict = sigmoid(predictions[:, :, 1])
            target_predict[target_predict > 0.4] = 1
            target_predict[target_predict <= 0.4] = 0  ### PROBABILITY

            temp1 = np.maximum(temp, target_predict)  # output activation & filter those minus, below 0
            sigmoid_pre_uint8 = cv2.normalize(temp1, None, 0, 255, cv2.NORM_MINMAX,
                                              cv2.CV_8U)

            sigmoid_pre_uint8[sigmoid_pre_uint8 > 0] = 255
            overall_pixels = len(np.column_stack(np.where(sigmoid_pre_uint8 == 255)))

            label = "closed"

            if overall_pixels > 40: ### pixels threshold
                label = "open"

            if label == "closed":
                frame_seg[y:y2, x:x2] = outputs_detected_onpic
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 150), 10)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 150), 5)
            elif label == "open":
                frame_seg[y:y2, x:x2] = outputs_detected_onpic
                cv2.rectangle(frame, (x, y), (x2, y2), (150, 0, 0), 10)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 0, 0), 5)
    else:
        pass

    fig, ax = plt.subplots(1, 2, figsize=(24,12))
    ax[0].imshow(frame.astype('uint8'))
    ax[0].set_title("Inference ROI results")
    ax[1].imshow(frame_seg.astype('uint8'))
    # plt.imshow((out * 255)..astype(np.uint8))  out.astype('uint8')
    ax[1].set_title("segmentation filtered")


if __name__ == "__main__":

    inference_whole_image_retina_openvino("./test.png")

