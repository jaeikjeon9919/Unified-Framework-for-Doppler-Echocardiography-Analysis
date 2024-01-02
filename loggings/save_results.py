import os
import cv2
import json
import numpy as np
import typing as T
from PIL import Image
from torchvision.transforms import functional as F
from .uncertainty_handler import UncertaintyHandler
from ..utils.util import mkdirs, save_png, save_npy, Denormalization


def save_results(input_batch, output, step_result_params):
    inference_result_dir = os.path.join(step_result_params["ckpt_dir"], "inference_result")
    cfg = step_result_params["cfg"]
    num_classes = len(output["prediction"][0])
    patient_name = input_batch["patient_name"][0]
    dcm_frame_name = input_batch["dcm_frame_name"][0]
    data_type = input_batch["data_type"][0]

    png_dir = os.path.join(inference_result_dir, data_type, patient_name)
    npy_dir = os.path.join(inference_result_dir, data_type, patient_name, "npy")
    mkdirs([png_dir, npy_dir])
    png_dir = os.path.join(png_dir, dcm_frame_name)
    npy_dir = os.path.join(npy_dir, dcm_frame_name)

    uncertainty_handler = UncertaintyHandler(output["prediction"].exp().cpu())
    try:
        setattr(uncertainty_handler, "MC_predictions", output["MC_predictions"].exp().cpu())
    except:
        pass

    model_prediction = output["prediction"][0].argmax(0).cpu().numpy()

    if cfg.logging.inference_analysis.resize_to_original:
        input_dir = input_batch["input_dir"][0]
        json_dir = input_batch["json_dir"][0]
        frame_num = input_batch["frame_num"][0]
        selected_classes = input_batch["selected_class"]
        selected_classes = [i[0] for i in selected_classes]
        with open(json_dir) as json_file:
            label = json.load(json_file)

        input_image = cv2.imread(input_dir)[:, :, :3]
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        model_prediction = np.array(
            F.resize(
                Image.fromarray(model_prediction.astype(np.uint8)),
                input_image.shape,
                interpolation=F.InterpolationMode.NEAREST,
            )
        )
        label = build_mask_from_contours(label, frame_num, selected_classes, input_image.shape)
    else:
        if cfg.data.datatype != "video":
            input_image = (Denormalization(input_batch["input"][0, 0, :, :].cpu().numpy()) * 255).astype(dtype=np.uint8)
        else:
            input_image = (Denormalization(input_batch["input"][0, 0, :, :].cpu().numpy())).astype(dtype=np.uint8)
            input_image = input_image[0, :, :]
        label = input_batch["label"][0].cpu().numpy()

    input_image = np.expand_dims(input_image, axis=2)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)

    # input image
    if cfg.logging.inference_analysis.input:
        save_png(png_dir + "_input.png", input_image)

    # ground-truth mask
    save_png(png_dir + "_label.png", label)
    save_npy(npy_dir + "_label.npy", label)

    # model output
    if cfg.logging.inference_analysis.output:
        # model_prediction = output["prediction"][0].argmax(0).cpu().numpy()
        save_png(png_dir + "_output.png", model_prediction)
        save_npy(npy_dir + "_output.npy", model_prediction)

    # draw contour of the model outputZ
    input_image_with_contour = draw_contour(input_image.copy(), label, model_prediction, num_classes=num_classes)
    save_png(png_dir + "_input_image_contour.png", input_image_with_contour)

    # draw contour of the only mode output
    input_image_with_contour = draw_contour_prediction(input_image.copy(), label, num_classes=num_classes)
    save_png(png_dir + "_input_image_label_contour.png", input_image_with_contour)

    # draw contour of the only mode output
    input_image_with_contour = draw_contour_prediction(input_image.copy(), model_prediction, num_classes=num_classes)
    save_png(png_dir + "_input_image_prediction_contour.png", input_image_with_contour)

    # probability map of model output
    if cfg.logging.inference_analysis.probability_map:
        save_png(png_dir + "_probability_map.png", output["prediction"][0, 1].cpu().numpy())

    # entropy
    if cfg.logging.inference_analysis.entropy:
        entropy = uncertainty_handler.get_entropy()
        save_png(png_dir + "_entropy.png", entropy)

    # BALD
    if cfg.logging.inference_analysis.BALD:
        BALD = uncertainty_handler.get_BALD()
        save_png(png_dir + "_BALD.png", BALD)

    # separate epistemic and aleatoric
    if cfg.logging.inference_analysis.epistemic:
        aleatoric, epistemic = uncertainty_handler.separate_uncertainty()
        tr_epistemic, tr_aleatoric = epistemic.sum(0), aleatoric.sum(0)
        save_png(png_dir + "_epistemic.png", tr_epistemic), save_png(png_dir + "_aleatoric.png", tr_aleatoric)


def build_mask_from_contours(label: T.Dict, frame_num: str, selected_classes: T.List, d_size: T.Tuple) -> np.ndarray:
    mask = np.zeros(d_size, dtype=np.uint8)
    for i, class_label in enumerate(selected_classes):
        class_contour = label["Frame"][str(frame_num)][class_label]
        if len(class_contour) != 1:
            for indv_class_contour in class_contour:
                cv2.drawContours(mask, [np.array(indv_class_contour)], -1, (i + 1), -1)
        else:
            cv2.drawContours(mask, np.array(class_contour), -1, (i + 1), -1)
    return mask


def draw_contour(
    input_image, model_prediction, label, num_classes, pred_contour_col=(255, 100, 0), gt_contour_col=(0, 100, 255),
):
    for i in range(1, num_classes):
        contours_pred, _ = cv2.findContours(
            (model_prediction == i).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
        )
        contours_label, _ = cv2.findContours((label == i).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contours_pred:
            for contour in contours_pred:
                cv2.drawContours(input_image, [contour], 0, pred_contour_col, 1)
        for contour in contours_label:
            cv2.drawContours(input_image, [contour], 0, gt_contour_col, 1)
    return input_image


def draw_contour_prediction(input_image, model_prediction, num_classes, line_thickness=1, color_codes=(255, 100, 0)):
    for i in range(1, num_classes):
        contours_pred, _ = cv2.findContours(
            (model_prediction == i).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours_pred:
            for contour in contours_pred:
                cv2.drawContours(input_image, [contour], 0, color_codes, line_thickness)
    return input_image
