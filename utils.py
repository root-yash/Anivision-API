import torch
def intersection_over_union(boxes_preds, boxes_labels):
    """
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    Returns:
        tensor: Intersection over union for all examples
    """
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def ctob(predictions, anchor):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.

    INPUT:
    predictions: tensor of size (N ,Batch_size,A, S, S, num_classes+5)
                 N: No of Anchor Block
                 A: Anchor in anchor block
                 S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes

    OUTPUT:
    converted_bboxes: the converted boxes of sizes (Batch_size,N*num_anchors*S*S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    num_anchors = 3
    converted_bboxes = []
    for i, prediction in enumerate(predictions):
        anchors = anchor[i].reshape(-1, 1, 1, 2)
        S = prediction.shape[2]
        BATCH_SIZE = prediction.shape[0]
        box_prediction = prediction[..., 1:5]
        w_h = box_prediction[..., 2:4] * anchors
        scores = prediction[..., 0:1]
        best_class = torch.argmax(prediction[..., 5:], dim=-1).unsqueeze(-1)
        cell_indices = (
            torch.arange(S)
                .repeat(prediction.shape[0], 3, S, 1)
                .unsqueeze(-1).to(prediction.device)
        )
        x = 1 / S * (box_prediction[..., 0:1] + cell_indices)
        y = 1 / S * (box_prediction[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
        temp = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
        if len(converted_bboxes) == 0:
            converted_bboxes = temp
        else:
            converted_bboxes = torch.cat((converted_bboxes, temp), dim=1)
    return converted_bboxes

def nms(bboxes,class_dict, threshold, iou_threshold, iou_threshold_o=1.0):
    class_list = list(class_dict)
    bboxes = bboxes[torch.argmax(bboxes[...,1])]
    bboxes_after_nms = []
    if bboxes[1] > threshold:
        chosen_box = bboxes.tolist()
        bboxes_after_nms.append(class_list[int(chosen_box[0])])
        bboxes_after_nms+=chosen_box[2:]
    return bboxes_after_nms

