def segmentation_eval_metrics(
    output, label, num_classes, view, dice_coeff_by_class, iou_by_class, important_class_dictionary, epsilon=1e-6,
):
    total_dice_coeff = 0
    total_iou = 0

    for index in range(1, num_classes):
        pred_inds = output.argmax(1) == index
        target_inds = label == index

        intersection_area = (pred_inds[target_inds]).long().sum().item()
        sum_of_area = pred_inds.sum() + target_inds.sum()
        union_area = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_area
        dice_coeff = 2 * float(intersection_area) / float(sum_of_area + epsilon)
        iou = float(intersection_area) / float(union_area + epsilon)

        dice_coeff_by_class[str(important_class_dictionary[view][index - 1])].append(dice_coeff)
        iou_by_class[str(important_class_dictionary[view][index - 1])].append(iou)

        total_dice_coeff += dice_coeff
        total_iou += iou

    mean_dice_coeff = total_dice_coeff / (num_classes - 1)  # exclude background
    mean_iou = total_iou / (num_classes - 1)  # exclude background
    return mean_dice_coeff, mean_iou
