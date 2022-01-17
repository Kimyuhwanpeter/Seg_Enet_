# -*- coding:utf-8 -*-
from Seg_Enet_model import *
from PFB_measurement_related import Measurement
from random import shuffle, random
from canny_edge import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/BoniRob/train.txt",

                           "val_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/BoniRob/val.txt",

                           "test_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/BoniRob/test.txt",
                           
                           "label_path": "/yuwhan/yuwhan/Dataset/Segmentation/BoniRob/raw_aug_gray_mask/",
                           
                           "image_path": "/yuwhan/yuwhan/Dataset/Segmentation/BoniRob/raw_aug_rgb_img/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/136/136",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 200,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 4,

                           "sample_images": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/Seg_Enet/BoniRob/sample_images",

                           "save_checkpoint": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/Seg_Enet/BoniRob/checkpoint",

                           "save_print": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/Seg_Enet/BoniRob/train_out.txt",

                           "test_images": "D:/[1]DB/[5]4th_paper_DB/crop_weed/related_work/SegNet/rice_seedling_weed/test_images",

                           "train": True})


optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
color_map = np.array([[255, 0, 0], [0, 0, 255], [0,0,0]], dtype=np.uint8)

def tr_func(image_list, label_list):


    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [384, 512])
    no_img = img
    IR = img[:, :, 0]/255.
    IG = img[:, :, 1]/255.
    IB = img[:, :, 2]/255.
    
	# Excess Green
    IExG = 2*IG - IR - IB
    IExG = tf.expand_dims(IExG, -1)
	# Excess Red
    IExR = 1.4*IR - IG
    IExR = tf.expand_dims(IExR, -1)
	# Color Index of Vegetation Extraction
    ICIVE = 0.881*IG - 0.441*IR - 0.385*IB - 18.78745
    ICIVE = tf.expand_dims(ICIVE, -1)
	# Normalized Difference Index
    INDI = (IG - IR)/(IG + IR)
    INDI = tf.expand_dims(INDI, -1)
    # HSV Color Space
    image_hsv = tf.image.rgb_to_hsv(img)
    IHUE,ISAT,IVAL = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]
    IHUE,ISAT,IVAL = tf.expand_dims(IHUE, -1), tf.expand_dims(ISAT, -1), tf.expand_dims(IVAL, -1)
    # Laplacian on IExG
    IExG_lap = laplacian(tf.pad(IExG, [[2,2],[2,2],[0,0]], constant_values=0))
    IExG_lap = IExG_lap[0, :, :, :]
    # Sobel in x & y directions on IExG
    IExG_sobel = tf.image.sobel_edges(IExG[tf.newaxis, :, :, :])
    IExG_sobelx = IExG_sobel[0, :, :, :, 1]
    IExG_sobely = IExG_sobel[0, :, :, :, 0]
    # Canny Edge Detector on IExG
    IExG_canny = TF_Canny(IExG[tf.newaxis, :, :, :], return_raw_edges=True)
    IExG_canny = tf.expand_dims(IExG_canny, -1)
    img = tf.concat((IB[:, :, tf.newaxis],IG[:, :, tf.newaxis],IR[:, :, tf.newaxis],IExG,IExR,ICIVE,INDI,IHUE,ISAT,IVAL,IExG_sobelx,IExG_sobely,IExG_lap,IExG_canny), -1)

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [384, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [384, 512])
    IR = img[:, :, 0]/255.
    IG = img[:, :, 1]/255.
    IB = img[:, :, 2]/255.
    
	# Excess Green
    IExG = 2*IG - IR - IB
    IExG = tf.expand_dims(IExG, -1)
	# Excess Red
    IExR = 1.4*IR - IG
    IExR = tf.expand_dims(IExR, -1)
	# Color Index of Vegetation Extraction
    ICIVE = 0.881*IG - 0.441*IR - 0.385*IB - 18.78745
    ICIVE = tf.expand_dims(ICIVE, -1)
	# Normalized Difference Index
    INDI = (IG - IR)/(IG + IR)
    INDI = tf.expand_dims(INDI, -1)
    # HSV Color Space
    image_hsv = tf.image.rgb_to_hsv(img)
    IHUE,ISAT,IVAL = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]
    IHUE,ISAT,IVAL = tf.expand_dims(IHUE, -1), tf.expand_dims(ISAT, -1), tf.expand_dims(IVAL, -1)
    # Laplacian on IExG
    IExG_lap = laplacian(tf.pad(IExG, [[2,2],[2,2],[0,0]], constant_values=0))
    IExG_lap = IExG_lap[0, :, :, :]
    # Sobel in x & y directions on IExG
    IExG_sobel = tf.image.sobel_edges(IExG[tf.newaxis, :, :, :])
    IExG_sobelx = IExG_sobel[0, :, :, :, 1]
    IExG_sobely = IExG_sobel[0, :, :, :, 0]
    # Canny Edge Detector on IExG
    IExG_canny = TF_Canny(IExG[tf.newaxis, :, :, :], return_raw_edges=True)
    IExG_canny = tf.expand_dims(IExG_canny, -1)
    img = tf.concat((IB[:, :, tf.newaxis],IG[:, :, tf.newaxis],IR[:, :, tf.newaxis],IExG,IExR,ICIVE,INDI,IHUE,ISAT,IVAL,IExG_sobelx,IExG_sobely,IExG_lap,IExG_canny), -1)

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [384, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def test_func2(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [384, 512])
    IR = img[:, :, 0]/255.
    IG = img[:, :, 1]/255.
    IB = img[:, :, 2]/255.
    
	# Excess Green
    IExG = 2*IG - IR - IB
    IExG = tf.expand_dims(IExG, -1)
	# Excess Red
    IExR = 1.4*IR - IG
    IExR = tf.expand_dims(IExR, -1)
	# Color Index of Vegetation Extraction
    ICIVE = 0.881*IG - 0.441*IR - 0.385*IB - 18.78745
    ICIVE = tf.expand_dims(ICIVE, -1)
	# Normalized Difference Index
    INDI = (IG - IR)/(IG + IR)
    INDI = tf.expand_dims(INDI, -1)
    # HSV Color Space
    image_hsv = tf.image.rgb_to_hsv(img)
    IHUE,ISAT,IVAL = image_hsv[:, :, 0], image_hsv[:, :, 1], image_hsv[:, :, 2]
    IHUE,ISAT,IVAL = tf.expand_dims(IHUE, -1), tf.expand_dims(ISAT, -1), tf.expand_dims(IVAL, -1)
    # Laplacian on IExG
    IExG_lap = laplacian(tf.pad(IExG, [[2,2],[2,2],[0,0]], constant_values=0))
    IExG_lap = IExG_lap[0, :, :, :]
    # Sobel in x & y directions on IExG
    IExG_sobel = tf.image.sobel_edges(IExG[tf.newaxis, :, :, :])
    print(IExG_sobel)
    IExG_sobelx = IExG_sobel[0, :, :, :, 1]
    IExG_sobely = IExG_sobel[0, :, :, :, 0]
    # Canny Edge Detector on IExG
    IExG_canny = TF_Canny(IExG[tf.newaxis, :, :, :], return_raw_edges=True)
    IExG_canny = tf.expand_dims(IExG_canny, -1)
    img = tf.concat((IB[:, :, tf.newaxis],IG[:, :, tf.newaxis],IR[:, :, tf.newaxis],IExG,IExR,ICIVE,INDI,IHUE,ISAT,IVAL,IExG_sobelx,IExG_sobely,IExG_lap,IExG_canny), -1)

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [384, 512], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, temp_img, lab

def run_model(model, images, training=True):
    return model(images, training=training)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

@tf.function
def cal_loss(model, images, batch_labels):

    with tf.GradientTape() as tape:

        batch_labels = tf.reshape(batch_labels, [-1,])

        logits = run_model(model, images, True)
        logits = tf.reshape(logits, [-1, FLAGS.total_classes])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(batch_labels, logits)

        # print(tf.keras.losses.BinaryCrossentropy(from_logits=True)(crop_labels, crop_logits))
        # print(tf.keras.losses.BinaryCrossentropy(from_logits=True)(weed_labels, weed_logits))
       
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def main():
    tf.keras.backend.clear_session()
    model = Seg_Enet()

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        #elif isinstance(layer, tf.keras.layers.Conv2D):
        #    layer.kernel_regularizer = tf.keras.regularizers.l2(0.0005)

    model.summary()
    
    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0
        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        val_list = np.loadtxt(FLAGS.val_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.image_path + data for data in train_list]
        val_img_dataset = [FLAGS.image_path + data for data in val_list]
        test_img_dataset = [FLAGS.image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.label_path + data for data in train_list]
        val_lab_dataset = [FLAGS.label_path + data for data in val_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        val_ge = tf.data.Dataset.from_tensor_slices((val_img_dataset, val_lab_dataset))
        val_ge = val_ge.map(test_func)
        val_ge = val_ge.batch(1)
        val_ge = val_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx): 
                batch_images, print_images, batch_labels = next(tr_iter)  
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == FLAGS.ignore_label, 2, batch_labels)    # 2 is void
                batch_labels = np.where(batch_labels == 255, 0, batch_labels)
                batch_labels = np.where(batch_labels == 128, 1, batch_labels)

                # crop_labels = np.where(batch_labels == 0, 1, 0)
                # weed_labels = np.where(batch_labels == 1, 1, 0)

                # crop_labels = np.squeeze(crop_labels, -1)
                # weed_labels = np.squeeze(weed_labels, -1)

                loss = cal_loss(model, batch_images, batch_labels)  # loss?????? ???߰? ???? test iou?̴? ?κ??ڵ? ???ľ???!
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:

                    logits = run_model(model, batch_images, False)
                    for i in range(FLAGS.batch_size):
                        logit = logits[i]
                        logit = tf.nn.softmax(logit, -1)
                        predict_image = tf.argmax(logit, -1)
                        
                        label = batch_labels[i]
                        label = np.squeeze(label, -1)
                        
                        pred_mask_color = color_map[predict_image]  # ?????׸?ó?? ?Ұ?!
                        
                        label = np.expand_dims(label, -1)
                        label = np.concatenate((label, label, label), -1)
                        label_mask_color = np.zeros([384, 512, 3], dtype=np.uint8)
                        label_mask_color = np.where(label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                        label_mask_color = np.where(label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                        temp_img = predict_image
                        temp_img = np.expand_dims(temp_img, -1)
                        temp_img2 = temp_img
                        temp_img = np.concatenate((temp_img, temp_img, temp_img), -1)
                        image = np.concatenate((temp_img2, temp_img2, temp_img2), -1)
                        pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), print_images[i], image)
                        pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping /= 255.

                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_warping_predict.png", pred_mask_warping)
                    

                count += 1

            tr_iter = iter(train_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model, batch_image, False)
                    logits = tf.nn.softmax(logits, -1)
                    predict_image = tf.argmax(logits, -1)

                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_image,
                                        label=batch_label, 
                                        shape=[384*512, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_ = Measurement(predict=predict_image,
                                            label=batch_label,
                                            shape=[384*512, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=predict_image,
                                            label=batch_label,
                                            shape=[384*512, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("=================================================================================================================================================")
            print("Epoch: %3d, train mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), train F1_score = %.4f, train sensitivity = %.4f, train TDR = %.4f" % (epoch, miou / len(train_img_dataset),
                                                                                                                                                 crop_iou / len(train_img_dataset),
                                                                                                                                                 weed_iou / len(train_img_dataset),
                                                                                                                                                  f1_score / len(train_img_dataset),
                                                                                                                                                  sensitivity / len(train_img_dataset),
                                                                                                                                                  tdr / len(train_img_dataset)))
            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train mIoU: ")
            output_text.write("%.4f" % (miou / len(train_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(train_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(train_img_dataset)))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score / len(train_img_dataset)))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(train_img_dataset)))
            output_text.write(", train TDR: ")
            output_text.write("%.4f" % (tdr / len(train_img_dataset)))
            output_text.write("\n")

            val_iter = iter(val_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            model_ = Seg_Enet(batch_size=1)
            model_.set_weights(model.get_weights())
            for i in range(len(val_img_dataset)):
                batch_images, batch_labels = next(val_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model_, batch_image, False)
                    logits = tf.nn.softmax(logits, -1)
                    predict_image = tf.argmax(logits, -1)

                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_image,
                                        label=batch_label, 
                                        shape=[384*512, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_ = Measurement(predict=predict_image,
                                            label=batch_label,
                                            shape=[384*512, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=predict_image,
                                            label=batch_label,
                                            shape=[384*512, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("Epoch: %3d, val mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), val F1_score = %.4f, val sensitivity = %.4f, val TDR = %.4f" % (epoch, miou / len(val_img_dataset),
                                                                                                                                         crop_iou / len(val_img_dataset),
                                                                                                                                         weed_iou / len(val_img_dataset),
                                                                                                                                         f1_score / len(val_img_dataset),
                                                                                                                                         sensitivity / len(val_img_dataset),
                                                                                                                                         tdr / len(val_img_dataset)))
            output_text.write("val mIoU: ")
            output_text.write("%.4f" % (miou / len(val_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(val_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(val_img_dataset)))
            output_text.write(", val F1_score: ")
            output_text.write("%.4f" % (f1_score / len(val_img_dataset)))
            output_text.write(", val sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(val_img_dataset)))
            output_text.write(", val TDR: ")
            output_text.write("%.4f" % (tdr / len(val_img_dataset)))
            output_text.write("\n")

            test_iter = iter(test_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            model_ = Seg_Enet(batch_size=1)
            model_.set_weights(model.get_weights())
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model_, batch_image, False)
                    logits = tf.nn.softmax(logits, -1)
                    predict_image = tf.argmax(logits, -1)

                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_image,
                                        label=batch_label, 
                                        shape=[384*512, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_ = Measurement(predict=predict_image,
                                            label=batch_label,
                                            shape=[384*512, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=predict_image,
                                            label=batch_label,
                                            shape=[384*512, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("Epoch: %3d, test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity = %.4f, test TDR = %.4f" % (epoch, miou / len(test_img_dataset),
                                                                                                                                             crop_iou / len(test_img_dataset),
                                                                                                                                             weed_iou / len(test_img_dataset),
                                                                                                                                             f1_score / len(test_img_dataset),
                                                                                                                                             sensitivity / len(test_img_dataset),
                                                                                                                                             tdr / len(test_img_dataset)))
            print("=================================================================================================================================================")
            output_text.write("test mIoU: ")
            output_text.write("%.4f" % (miou / len(test_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(test_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(test_img_dataset)))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score / len(test_img_dataset)))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(test_img_dataset)))
            output_text.write(", test TDR: ")
            output_text.write("%.4f" % (tdr / len(test_img_dataset)))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/Crop_weed_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)
    else:
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        test_img_dataset = [FLAGS.image_path + data for data in test_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func2)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_iter = iter(test_ge)
        miou = 0.
        f1_score = 0.
        tdr = 0.
        sensitivity = 0.
        crop_iou = 0.
        weed_iou = 0.
        pre_ = 0.
        TP_, TN_, FP_, FN_ = 0, 0, 0, 0
        for i in range(len(test_img_dataset)):
            batch_images, nomral_img, batch_labels = next(test_iter)
            batch_labels = tf.squeeze(batch_labels, -1)
            for j in range(1):
                batch_image = tf.expand_dims(batch_images[j], 0)
                logits = run_model(model, batch_image, False) # type?? batch label?? ???? type???? ?????־?????

                    
                logits = tf.nn.softmax(logits, -1)
                predict_image = tf.argmax(logits, -1)

                batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                batch_label = np.where(batch_label == 255, 0, batch_label)
                batch_label = np.where(batch_label == 128, 1, batch_label)

                output_confusion, temp_output_3D = Measurement(predict=predict_image,
                                        label=batch_label,
                                        shape=[384*512, ],
                                        total_classes=FLAGS.total_classes).show_confusion()
                output_confusion_ = output_confusion

                pred_mask_color = color_map[predict_image]  # ?????׸?ó?? ?Ұ?!
                batch_label = np.expand_dims(batch_label, -1)
                batch_label = np.concatenate((batch_label, batch_label, batch_label), -1)
                label_mask_color = np.zeros([384*512, 3], dtype=np.uint8)
                label_mask_color = np.where(batch_label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                label_mask_color = np.where(batch_label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                predict_image = np.expand_dims(predict_image, -1)
                temp_img = np.concatenate((predict_image, predict_image, predict_image), -1)
                image = np.concatenate((predict_image, predict_image, predict_image), -1)
                pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), nomral_img[j], image)
                pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                pred_mask_warping /= 255.

                output_confusion_warp = np.where(temp_output_3D == np.array([0,0,0], dtype=np.uint8), nomral_img[j], temp_output_3D)
                output_confusion_warp = np.where(temp_output_3D == np.array([10,10,10], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), output_confusion_warp)
                output_confusion_warp = np.where(temp_output_3D == np.array([20,20,20], dtype=np.uint8), np.array([0, 255, 255], dtype=np.uint8), output_confusion_warp)
                output_confusion_warp = np.where(temp_output_3D == np.array([30,30,30], dtype=np.uint8), np.array([255, 0, 255], dtype=np.uint8), output_confusion_warp)
                output_confusion_warp = np.where(temp_output_3D == np.array([40,40,40], dtype=np.uint8), np.array([255, 255, 0], dtype=np.uint8), output_confusion_warp)
                output_confusion_warp = np.array(output_confusion_warp, dtype=np.int32)
                output_confusion_warp = output_confusion_warp / 255

                name = test_img_dataset[i].split("/")[-1].split(".")[0]
                plt.imsave(FLAGS.test_images + "/" + name + "_label.png", label_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict.png", pred_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict_warp.png", pred_mask_warping[0])
                plt.imsave(FLAGS.test_images + "/" + name + "_output_confusion.png", output_confusion)
                plt.imsave(FLAGS.test_images + "/" + name + "output_confusion_warp.png", output_confusion_warp)

if __name__ == "__main__":
    main()
