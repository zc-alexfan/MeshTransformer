import torch


def xyxy2xywh(boxes):
    assert isinstance(boxes, (torch.Tensor))
    assert boxes.size(1) == 4
    """
    xyxy: top left and bottom right corner coordinates.
    xywh: center coordinates of the box and full width/height.
    """
    x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    xywh = torch.stack((x, y, w, h), dim=1)
    return xywh


def xywh2xyxy(boxes):
    assert isinstance(boxes, (torch.Tensor))
    assert boxes.size(1) == 4
    """
    xyxy: top left and bottom right corner coordinates.
    xywh: center coordinates of the box and full width/height.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = x - 0.5 * w
    x2 = x + 0.5 * w
    y1 = y - 0.5 * h
    y2 = y + 0.5 * h
    xyxy = torch.stack((x1, y1, x2, y2), dim=1)
    return xyxy


def encode(gt_boxes, proposals):
    """
    Both are in form xyxy with the top-left and bottom-right pixel coordinates
    """
    assert isinstance(gt_boxes, torch.Tensor)
    assert isinstance(proposals, torch.Tensor)
    assert proposals.size() == gt_boxes.size()
    assert gt_boxes.size(1) == 4
    gt_boxes = xyxy2xywh(gt_boxes)  # xywh
    proposals = xyxy2xywh(proposals)

    gx = gt_boxes[:, 0]
    gy = gt_boxes[:, 1]
    gw = gt_boxes[:, 2]
    gh = gt_boxes[:, 3]

    px = proposals[:, 0]
    py = proposals[:, 1]
    pw = proposals[:, 2]
    ph = proposals[:, 3]

    # avoid having negative or zero width/height
    pw = torch.clamp(pw, 1, float("inf"))
    ph = torch.clamp(ph, 1, float("inf"))
    gw = torch.clamp(gw, 1, float("inf"))
    gh = torch.clamp(gh, 1, float("inf"))

    tx = (gx - px) / pw
    ty = (gy - py) / ph
    tw = torch.log(gw / pw)
    th = torch.log(gh / ph)
    t_target = torch.stack((tx, ty, tw, th), dim=1)
    return t_target


def decode(proposals, t_target):
    """
    proposals: xyxy with top-left and bottom-right pixel coordinates.
    t_target: either predicted from the regressor or from the `encode` function
    """
    assert isinstance(t_target, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(proposals, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert proposals.size() == t_target.size()
    assert t_target.size(1) == 4

    proposals = xyxy2xywh(proposals)
    px = proposals[:, 0]
    py = proposals[:, 1]
    pw = proposals[:, 2]
    ph = proposals[:, 3]

    tx = t_target[:, 0]
    ty = t_target[:, 1]
    tw = t_target[:, 2]
    th = t_target[:, 3]

    gx = pw * tx + px
    gy = ph * ty + py
    gw = pw * torch.exp(tw)
    gh = ph * torch.exp(th)

    pred_boxes = torch.stack((gx, gy, gw, gh), dim=1)
    pred_boxes = xywh2xyxy(pred_boxes)  # convert back to xyxy form
    return pred_boxes


def intersect(pred_bbox, target_bbox):
    """
    Compute the intersection area between pred_bbox and target_bbox.
    pred_bbox: (num_box, 4) in xyxy
    target_bbox: (num_box, 4) in xyxy
    Output: (num_box)
    """
    assert target_bbox.size() == pred_bbox.size()
    assert len(pred_bbox.size()) == 2
    assert pred_bbox.size(1) == 4

    x1, y1, x2, y2 = torch.chunk(pred_bbox, 4, dim=1)
    a1, b1, a2, b2 = torch.chunk(target_bbox, 4, dim=1)
    x_overlap = torch.max(
        torch.FloatTensor([0.0]).to(pred_bbox.device),
        torch.min(x2, a2) - torch.max(x1, a1),
    )
    y_overlap = torch.max(
        torch.FloatTensor([0.0]).to(pred_bbox.device),
        torch.min(y2, b2) - torch.max(y1, b1),
    )
    inter_area = (x_overlap * y_overlap).view(-1)
    return inter_area


def union(pred_bbox, target_bbox, intersect_area):
    """
    Compute the union area between pred_bbox and target_bbox.
    pred_bbox: (num_box, 4) in xyxy
    target_bbox: (num_box, 4) in xyxy
    Output: (num_box)
    """
    assert target_bbox.size() == pred_bbox.size()
    assert len(pred_bbox.size()) == 2
    assert pred_bbox.size(1) == 4
    assert target_bbox.size(0) == intersect_area.size(0)
    assert len(intersect_area.size()) == 1
    x1, y1, x2, y2 = torch.chunk(pred_bbox, 4, dim=1)
    a1, b1, a2, b2 = torch.chunk(target_bbox, 4, dim=1)
    area_a = (x2 - x1) * (y2 - y1)
    area_b = (a2 - a1) * (b2 - b1)
    area_a = area_a.view(-1)
    area_b = area_b.view(-1)
    return area_a + area_b - intersect_area


def box_iou(pred_bbox, target_bbox):
    """
    Compute the IoU between pred_bbox and target_bbox.
    pred_bbox: (num_box, 4) in xyxy
    target_bbox: (num_box, 4) in xyxy
    Output: (num_box)
    """
    assert target_bbox.size() == pred_bbox.size()
    assert len(pred_bbox.size()) == 2
    assert pred_bbox.size(1) == 4
    top = intersect(pred_bbox, target_bbox)
    bottom = union(pred_bbox, target_bbox, top)
    iou = top / bottom
    return iou


def bound_boxes(bboxes, w_list, h_list):
    """
    Bound the boxes (xyxy) within the image size.
    bboxes: (num_box, 4)
    wlist: (num_box)
    hlist: (num_box)
    Return: (num_box, 4)
    """
    assert isinstance(bboxes, torch.Tensor)
    assert isinstance(w_list, torch.Tensor)
    assert isinstance(h_list, torch.Tensor)
    assert len(bboxes.shape) == 2
    assert len(w_list.shape) == 1
    assert bboxes.shape[1] == 4
    assert bboxes.shape[0] == w_list.shape[0]
    assert w_list.shape == h_list.shape

    # avoid boundary problem
    mask = torch.ones(bboxes.size()).to(w_list.device)
    bboxes = torch.max(bboxes, mask)
    bboxes[:, 0] = torch.min(bboxes[:, 0], w_list)
    bboxes[:, 1] = torch.min(bboxes[:, 1], h_list)
    bboxes[:, 2] = torch.min(bboxes[:, 2], w_list)
    bboxes[:, 3] = torch.min(bboxes[:, 3], h_list)
    return bboxes
