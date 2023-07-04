import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.Cityscapes import CityscapesDataset
from model.UNET import UNET

# Hyperparameters etc.
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_SCALE = 0.25
LOAD_MODEL = False
DATA_ROOT = r"./training_data"


def main():
    print("Init dataset ...")
    dataset_train = CityscapesDataset(DATA_ROOT, split="train", scale=IMAGE_SCALE)
    dataset_val = CityscapesDataset(DATA_ROOT, split="val")
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Init model using {DEVICE=} ...")
    model = UNET(in_channels=3, out_channels=19).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # if LOAD_MODEL:
    #     load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        # loop = tqdm(train_loader)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")

        for batch_idx, (image, label, _, _) in enumerate(train_loader):
            image = image.to(DEVICE)
            label = label.float().to(DEVICE)

            # forward
            predictions = model(image)
            loss = loss_fn(predictions, label)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update tqdm loop
            # loop.set_postfix(loss=loss.item())
            print(f"[Batch {batch_idx:4d}/{len(train_loader)}] Loss: {loss.item():.4f}")

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
