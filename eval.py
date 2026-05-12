import torch
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score
)


def evaluate_model(model, loader, device, threshold=0.5):

    model.eval()

    all_preds = []
    all_masks = []

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            preds = torch.sigmoid(outputs)

            preds = (preds > threshold).float()

            preds = preds.cpu().numpy().flatten()
            masks = masks.cpu().numpy().flatten()

            all_preds.extend(preds)
            all_masks.extend(masks)

    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)

    precision = precision_score(
        all_masks,
        all_preds,
        zero_division=0
    )

    recall = recall_score(
        all_masks,
        all_preds,
        zero_division=0
    )

    f1 = f1_score(
        all_masks,
        all_preds,
        zero_division=0
    )

    iou = jaccard_score(
        all_masks,
        all_preds,
        zero_division=0
    )

    print("\n===== EVALUATION RESULTS =====")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"IoU Score : {iou:.4f}")