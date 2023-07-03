import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.Cityscapes import CityscapesDataset
from model.UNET import UNET

# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_imgs,
# )

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 100  # 1024 originally
IMAGE_WIDTH = 200  # 2048 originally
LOAD_MODEL = False
DATA_ROOT = r"./training_data"


def main():
    # train_transform = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Rotate(limit=35, p=1.0),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.1),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    # val_transforms = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    print(f"Init model using {DEVICE=} ...")
    model = UNET(in_channels=3, out_channels=19).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Init dataset ...")
    dataset_train = CityscapesDataset(DATA_ROOT, split="train")
    dataset_val = CityscapesDataset(DATA_ROOT, split="val")
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # check_accuracy(val_loader, model, device=DEVICE)
    # scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        # loop = tqdm(train_loader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        for batch_idx, (image, label, _, _) in enumerate(train_loader):
            print("loading image and label ...")
            image = image.to(device=DEVICE)[:, :, :IMAGE_HEIGHT, :IMAGE_WIDTH]
            label = label.float().to(device=DEVICE)[:, :, :IMAGE_HEIGHT, :IMAGE_WIDTH]
            print(f"{image.shape=}, {label.shape=}")

            # forward
            # with torch.cuda.amp.autocast():
            print("forward")
            predictions = model(image)
            loss = loss_fn(predictions, label)

            # backward
            print("backward")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            # loop.set_postfix(loss=loss.item())
            print(f"[Batch {batch_idx:4d}] Loss: {loss.item():.4f}")

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # save_checkpoint(checkpoint)

        # # check accuracy
        # check_accuracy(val_loader, model, device=DEVICE)

        # # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )


if __name__ == "__main__":
    main()
