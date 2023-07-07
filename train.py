import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset.Cityscapes import CityscapesDataset
from model.UNET import UNET
from pathlib import Path
import numpy as np
import os
from dotenv import load_dotenv

# Hyperparameters etc.
load_dotenv()
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 1000
NUM_WORKERS = 2
IMAGE_SCALE = 0.25
LOAD_MODEL = False
DATA_ROOT = os.environ['CITYSCAPES_DATASET']


def main():
    print("Init dataset ...")
    dataset_train = CityscapesDataset(DATA_ROOT, split="train", scale=IMAGE_SCALE)
    # subset to test if it overfits, comment this for full scale training
    dataset_train = Subset(dataset_train, np.arange(5))
    dataset_val = CityscapesDataset(DATA_ROOT, split="val", scale=IMAGE_SCALE)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Init model using {DEVICE=} ...")
    model = UNET(in_channels=3, out_channels=19).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        # loop = tqdm(train_loader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        model.train()
        for batch_idx, (image, label, _, _) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.float().to(DEVICE)

            # forward
            output = model(image)
            loss = criterion(output, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            # loop.set_postfix(loss=loss.item())

            if batch_idx % 10 == 0:
                print(
                    f"[Batch {batch_idx:4d}/{len(train_loader)}]"
                    f" Loss: {loss.item():.4f}"
                )
        model.eval()

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        # save_checkpoint(checkpoint)

        # # check accuracy
        # check_accuracy(val_loader, model, device=DEVICE)

        # save model and some examples to a folder
        print("save snapshot")
        image, label, _, _ = dataset_train[0]
        image = image.to(DEVICE)
        output = model(image.unsqueeze(0)).squeeze().to("cpu")
        folder = Path("snapshot") / f"e{epoch:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        CityscapesDataset.plot_image(image, folder / "image.png")
        CityscapesDataset.plot_mask(label, folder / "label.png")
        CityscapesDataset.plot_output(output, folder / "output.png")

        print("")


if __name__ == "__main__":
    main()
