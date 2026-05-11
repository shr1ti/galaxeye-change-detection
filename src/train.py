import torch
from tqdm import tqdm


def train_one_epoch(

    model,
    loader,
    criterion,
    optimizer,
    device

):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(loader)

    for images, masks in progress_bar:

        images = images.to(device)

        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_description(
            f"Loss: {loss.item():.4f}"
        )

    epoch_loss = running_loss / len(loader)

    return epoch_loss