import torch

def diou_loss(preds, targets, eps=1e-7):
    # Конвертируем рамки в формат [x1, y1, x2, y2]
    preds_x1 = preds[..., 0] - preds[..., 2] / 2
    preds_y1 = preds[..., 1] - preds[..., 3] / 2
    preds_x2 = preds[..., 0] + preds[..., 2] / 2
    preds_y2 = preds[..., 1] + preds[..., 3] / 2
    
    targets_x1 = targets[..., 0] - targets[..., 2] / 2
    targets_y1 = targets[..., 1] - targets[..., 3] / 2
    targets_x2 = targets[..., 0] + targets[..., 2] / 2
    targets_y2 = targets[..., 1] + targets[..., 3] / 2

    # Вычисляем IoU
    # Найдем координаты области пересечения
    inter_x1 = torch.max(preds_x1, targets_x1)
    inter_y1 = torch.max(preds_y1, targets_y1)
    inter_x2 = torch.min(preds_x2, targets_x2)
    inter_y2 = torch.min(preds_y2, targets_y2)

    # Площадь пересечения. Используйте clamp для избежания отрицательных значений.
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Площади предсказанных и истинных рамок
    preds_area = preds[..., 2] * preds[..., 3]
    targets_area = targets[..., 2] * targets[..., 3]
    
    # Площадь объединения
    union_area = preds_area + targets_area - inter_area + eps
    
    # Итоговый IoU
    iou = inter_area / union_area

    # Вычисляем штраф за расстояние (Distance Penalty)
    # Квадрат расстояния между центрами рамок (rho^2)
    center_dist_sq = (preds[..., 0] - targets[..., 0])**2 + (preds[..., 1] - targets[..., 1])**2

    # Найдём координаты углов наименьшей рамки, которая охватывает обе (enclosing box)
    enclose_x1 = torch.min(preds_x1, targets_x1)
    enclose_y1 = torch.min(preds_y1, targets_y1)
    enclose_x2 = torch.max(preds_x2, targets_x2)
    enclose_y2 = torch.max(preds_y2, targets_y2)

    # Квадрат диагонали охватывающей рамки (c^2). Добавляем eps для стабильности
    enclose_diag_sq = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + eps
    
    # Штраф за расстояние (отношение квадрата расстояния между центрами к квадрату диагонали охватывающей рамки)
    distance_penalty = center_dist_sq / enclose_diag_sq

    # Собираем всё вместе
    diou = iou - distance_penalty
    
    # DIoU Loss определяется как 1 - DIoU
    loss = 1.0 - diou
    
    return loss

if __name__ == '__main__':
    # 1: Рамки полностью совпадают, loss должен быть 0
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.3]])
    loss = diou_loss(boxes, boxes)
    assert abs(loss.item() - 0.0) < 1e-5, f"loss должен быть ~0, а он {loss.item()}"

    # 2: Рамки не пересекаются, loss должен быть > 1 из-за distance_penalty
    preds_2 = torch.tensor([[0.2, 0.2, 0.1, 0.1]])
    targets_2 = torch.tensor([[0.8, 0.8, 0.1, 0.1]])
    loss_2 = diou_loss(preds_2, targets_2)
    assert loss_2.item() > 1.0, f"loss должен быть > 1, а он {loss_2.item()}"

    # 3: Рамки с одним центром, но разного размера. Loss должен быть > 0 и < 1
    preds_3 = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    targets_3 = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    loss_3 = diou_loss(preds_3, targets_3)
    assert 0.0 < loss_3.item() < 1.0, f"loss должен быть между 0 и 1, а он {loss_3.item()}"

    # 4: Функция должна корректно работать с батчем из нескольких рамок
    preds_4 = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],  # Совпадающая
        [0.2, 0.2, 0.1, 0.1]   # Непересекающаяся
    ])
    targets_4 = torch.tensor([
        [0.5, 0.5, 0.2, 0.2],
        [0.8, 0.8, 0.1, 0.1]
    ])
    losses_4 = diou_loss(preds_4, targets_4)
    assert losses_4.shape == (2,), f"форма результата должна быть (2,), а она {losses_4.shape}"
    assert abs(losses_4[0].item() - 0.0) < 1e-5, f"loss для первой пары должен быть ~0, а он {losses_4[0].item()}"
    assert losses_4[1].item() > 1.0, f"loss для второй пары должен быть > 1, а он {losses_4[1].item()}"
