import os
import gc
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter

from dataset import MZIrisDataset, MZIrisDatasetMask
from network import SiameseResnet

from loss import ContrastiveLoss


def load_dataset(
    data_path: str,
    image_dir: str,
    batch_size: int,
    mask: bool,
    mask_dir: str,
    mask_inverse: bool,
    device,
):
    if mask:
        dataset = MZIrisDatasetMask(
            data_path,
            image_dir,
            mask_dir,
            mask_inverse,
        )
    else:
        dataset = MZIrisDataset(
            data_path,
            image_dir,
        )
    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[args.train_split, 1 - args.train_split],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        # collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
    )
    return train_dataloader, val_dataloader


def write_results(
    writer,
    event: str,  # "train" / "val"
    epoch: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    pos_dist: float,
    neg_dist: float,
    avg_loss: float,
) -> None:
    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = r = tp / (tp + fn)
    tnr = tn / (tn + fp)
    # fpr = fp / (tn + fp)
    p = tp / (tp + fp)
    f1 = 2 / (1 / p + 1 / r)
    avg_pd = pos_dist / (tp + fn)
    avg_nd = neg_dist / (tn + fp)

    print(f"{event.upper()} Accuracy (%): {acc*100}")
    print(f"{event.upper()} TPR (%): {tpr*100}")
    print(f"{event.upper()} TNR (%): {tnr*100}")
    print(f"{event.upper()} Precision (%): {p*100}")
    print(f"{event.upper()} Recall (%): {r*100}")
    print(f"{event.upper()} F1 score (%): {f1*100}")
    print(f"{event.upper()} average positive distance: {avg_pd}")
    print(f"{event.upper()} average negative distance: {avg_nd}")
    print(f"{event.upper()} batch average loss: {avg_loss}")

    writer.add_scalar(f"{event}/acc", acc, epoch)
    writer.add_scalar(f"{event}/tpr", tpr, epoch)
    writer.add_scalar(f"{event}/tnr", tnr, epoch)
    writer.add_scalar(f"{event}/precision", p, epoch)
    writer.add_scalar(f"{event}/recall", r, epoch)
    writer.add_scalar(f"{event}/f1", f1, epoch)
    writer.add_scalar(f"{event}/average_positive_distance", avg_pd, epoch)
    writer.add_scalar(f"{event}/average_negative_distance", avg_nd, epoch)
    writer.add_scalar(f"{event}/batch_average_loss", avg_loss, epoch)


def train1(
    dataloader: DataLoader,
    model: nn.Sequential,
    optimizer,
    loss_fn,
    thres: int,
    device,
):
    loss = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total_pos_dist = 0
    total_neg_dist = 0
    y_true = []
    y_pred = []

    for i, data in tqdm(
        enumerate(dataloader),
        unit="batch",
        total=len(dataloader),
    ):
        # Training
        sid_left, sid_right, left, right, labels = data
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        embeddings_left, embeddings_right = model(left, right)
        loss_ = loss_fn(embeddings_left, embeddings_right, labels)
        loss_.backward()
        optimizer.step()

        # Accuracy and distance
        batch_dist = nn.functional.pairwise_distance(
            embeddings_left,
            embeddings_right,
        )
        for i, dist in enumerate(batch_dist):
            dist = dist.item()
            y_true.append(labels[i].item())
            y_pred.append(1 if dist <= thres else 0)
            if labels[i] == 1:
                total_pos_dist += dist
                if dist <= thres:
                    tp += 1
                else:
                    fn += 1
            elif labels[i] == 0:
                total_neg_dist += dist
                if dist > thres:
                    tn += 1
                else:
                    fp += 1

        # Loss
        loss += loss_.item()

    gc.collect()

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pd": total_pos_dist,
        "nd": total_neg_dist,
        "loss": loss / len(dataloader),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def val1(
    dataloader: DataLoader,
    model: nn.Sequential,
    loss_fn,
    thres: int,
    device,
):
    loss = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total_pos_dist = 0
    total_neg_dist = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            left, right, labels = data
            embeddings_left, embeddings_right = model(left, right)
            loss_ = loss_fn(embeddings_left, embeddings_right, labels)
            batch_dist = nn.functional.pairwise_distance(
                embeddings_left,
                embeddings_right,
            )
            for i, dist in enumerate(batch_dist):
                y_true.append(labels[i])
                y_pred.append(1 if dist <= thres else 0)
                if labels[i] == 1:
                    total_pos_dist += dist
                    if dist <= thres:
                        tp += 1
                    else:
                        fn += 1
                else:
                    total_neg_dist += dist
                    if dist > thres:
                        tn += 1
                    else:
                        fp += 1
            loss += loss_.item()

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pd": total_pos_dist,
        "nd": total_neg_dist,
        "loss": loss / len(dataloader),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backbone",
        "-b",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
        ],
        help="Choose one model backbone from ['resnet18', 'resnet34', 'resnet50', 'resnet101']",
    )
    parser.add_argument(
        "--init-weight-path",
        "-w",
        type=str,
        default="",
        help="Path to the initial weight",
    )
    parser.add_argument("--data-path", "-d", required=True, help="Path to the dataset")
    parser.add_argument(
        "--image-dir",
        "-i",
        require=True,
        help="Path to the image directory",
    )
    parser.add_argument(
        "--train-split",
        "-ts",
        type=float,
        default=0.7,
        help="Train/val split ratio, defaults to 0.7",
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=32, help="Batch size, defaults to 32"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=20, help="Number of epochs, defaults to 20"
    )
    parser.add_argument(
        "--lr",
        "-l",
        type=float,
        default=0.0001,
        help="Adam learning rate, defaults to 0.0001",
    )
    parser.add_argument(
        "--thres",
        "-t",
        type=float,
        default=0.5,
        help="Distance threshold, defaults to 0.5",
    )
    parser.add_argument(
        "--suffix", "-s", type=str, default="", help="Suffix for output file/dir name"
    )
    parser.add_argument(
        "--mask",
        "-m",
        action="store_true",
        help="Mask images - need to specify mask directory",
    )
    parser.add_argument(
        "--mask-dir", "-md", type=str, default="", help="Path to the mask directory"
    )
    parser.add_argument(
        "--mask-inverse", "-mi", action="store_true", help="Mask inverse"
    )

    return parser.parse_args()


def check_args(args):
    if args.init_weight_path and not os.path.isfile(args.init_weight_path):
        print("ERROR: Initial weight path does not exist.")
        exit(1)

    if not os.path.isfile(args.data_path):
        print("ERROR: Data path does not exist.")
        exit(1)

    if not os.path.isdir(args.image_dir):
        print("ERROR: Image directory does not exist.")
        exit(1)

    if args.mask and not args.mask_dir:
        print("ERROR: Mask directory is required with masking option.")
        exit(1)

    if args.mask_dir and not os.path.isdir(args.mask_dir):
        print("ERROR: Mask directory does not exist.")
        exit(1)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    args = parse_arguments()
    check_args(args)
    print("Argument parsed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected device type: {device.type}")

    train_dataloader, val_dataloader = load_dataset(
        args.data_path,
        args.image_dir,
        args.batch_size,
        args.mask,
        args.mask_dir,
        args.mask_inverse,
        device,
    )
    print("Training and validation dataset loaded.")

    model = SiameseResnet(backbone=args.backbone)
    if args.init_weight_path:
        weights = torch.load(args.init_weight_path)
        model.load_state_dict(weights)
    model.to(device)
    print("Model loaded.")

    checkpoint_dir = (
        f"./checkpoints/{args.backbone}_{args.suffix}_{timestamp}"
        if args.suffix
        else f"./checkpoints/{args.backbone}_{timestamp}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Model checkpoints will be saved to {checkpoint_dir}.")

    optimizer = Adam(model.parameters(), lr=args.lr)
    print("Optimizer constructed.")

    loss_fn = ContrastiveLoss()
    print("Loss constructed.")

    result_path = (
        f"./runs/results_{args.backbone}_{args.suffix}_{timestamp}"
        if args.suffix
        else f"./runs/{args.backbone}_{timestamp}"
    )
    writer = SummaryWriter(result_path)
    writer.add_hparams(
        {
            "Backbone structure": args.backbone,
            "Train split": args.train_split,
            "Batch size": args.batch_size,
            "Optimizer": "Adam",
            "Learning rate": args.lr,
            "Distance threshold": args.thres,
            "Epochs": args.epochs,
        },
        {},
    )
    print(f"Results will be saved to {result_path}.")

    print("Training preperation completed :)")

    # Training loop
    best_loss = 999999999
    for epoch in range(args.epochs):
        print("#" * 20)
        print(f"Epoch {epoch + 1}")

        # Training
        model.train()
        print("Training...")
        train_results = train1(
            train_dataloader,
            model,
            optimizer,
            loss_fn,
            args.thres,
            device,
        )
        write_results(
            writer,
            "train",
            epoch,
            train_results["tp"],
            train_results["tn"],
            train_results["fp"],
            train_results["fn"],
            train_results["pd"],
            train_results["tp"],
            train_results["loss"],
            # train_results["y_true"],
            # train_results["y_pred"],
        )

        # Validation
        model.eval()
        print("Validating...")
        val_results = val1(
            val_dataloader,
            model,
            loss_fn,
            args.thres,
            device,
        )
        write_results(
            writer,
            "val",
            epoch,
            val_results["tp"],
            val_results["tn"],
            val_results["fp"],
            val_results["fn"],
            val_results["pd"],
            val_results["tp"],
            val_results["loss"],
            # val_results["y_true"],
            # val_results["y_pred"],
        )

        # Save train vs validation loss
        writer.add_scalars(
            "train_vs_val_loss",
            {
                "training": train_results["loss"],
                "validation": val_results["loss"],
            },
            epoch,
        )
        writer.flush()
        print("Results saved")

        # Save model checkpoint
        checkpoint_name = f"epoch{epoch}"
        if val_results["loss"] < best_loss:
            checkpoint_name = checkpoint_name + "_best"
            best_loss = val_results["loss"]
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print("Checkpoint saved.")

        print()

    writer.close()
