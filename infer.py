import os
import gc
import argparse
from tqdm import tqdm
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

from dataset import MZIrisDataset, MZIrisDatasetMask
from network import SiameseResnet
from loss import ContrastiveLoss


def load_dataset(
    data_path: str,
    image_dir: str,
    batch_size: int,
    mask: bool,
    mask_dir: str,
    mask_inv: bool,
):
    if mask:
        test_dataset = MZIrisDatasetMask(
            data_path,
            image_dir,
            mask_dir,
            mask_inv,
        )
    else:
        test_dataset = MZIrisDataset(
            data_path,
            image_dir,
        )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def load_model(backbone, weight_path):
    model = SiameseResnet(backbone=backbone)
    weights = torch.load(weight_path)
    model.add_hook()
    model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(weights)
    model.eval()
    return model


def plot_conf_matrix(y_true, y_pred):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, ax=ax)
    return fig


def write_results(
    writer,
    model,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    pos_dist: float,
    neg_dist: float,
    avg_dist: float,
    avg_loss: float,
    y_true: list,
    y_pred: list,
    cams,
) -> None:
    acc = (tp + tn) / (tp + tn + fp + fn)
    tpr = r = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    p = tp / (tp + fp)
    f1 = 2 / (1 / p + 1 / r)
    avg_pd = pos_dist / (tp + fn)
    avg_nd = neg_dist / (tn + fp)

    writer.add_scalar("acc", acc)
    writer.add_scalar("tpr", tpr)
    writer.add_scalar("fpr", fpr)
    writer.add_scalar("tnr", tnr)
    writer.add_scalar("precision", p)
    writer.add_scalar("recall", r)
    writer.add_scalar("f1", f1)
    writer.add_scalar("average_positive_distance", avg_pd)
    writer.add_scalar("average_negative_distance", avg_nd)
    writer.add_scalar("average_distance", avg_dist)
    writer.add_scalar("average_loss", avg_loss)
    writer.add_figure("conf_matrix", plot_conf_matrix(y_true, y_pred))
    writer.add_image("cams", cams)


def save(
    save_image_dir: str,
    left_sid_batch: str,
    right_sid_batch: str,
    left_batch,
    right_batch,
    label: str,
    i: int,
):
    for image_tensor, image_name in zip(left_batch, left_sid_batch):
        image = to_pil_image(image_tensor)
        image.save(
            f"./test_res_images/{save_image_dir}/{label}/left_{i}_{image_name}.png"
        )
    for image_tensor, image_name in zip(right_batch, right_sid_batch):
        image = to_pil_image(image_tensor)
        image.save(
            f"./test_res_images/{save_image_dir}/{label}/right_{i}_{image_name}.png"
        )


def infer(
    dataloader: DataLoader,
    model: nn.Sequential,
    loss_fn,
    thres: int,
    save_image: bool,
    save_image_dir: str,
    device,
):
    losses = []
    dists = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total_pos_dist = 0
    total_neg_dist = 0
    y_true = []
    y_pred = []
    cams = []

    with torch.no_grad():
        for i, data in tqdm(
            enumerate(dataloader),
            unit="pair",
            total=len(dataloader),
        ):
            sid_left, sid_right, left, right, label = data
            left = left.to(device)
            right = right.to(device)
            label = label.to(device)
            embedding_left, embedding_right = model(left, right)
            losses.append(loss_fn(embedding_left, embedding_right, label).item())
            dists.append(
                nn.functional.pairwise_distance(
                    embedding_left,
                    embedding_right,
                ).item()
            )
            y_true.append(label.item())
            y_pred.append(1 if dists[-1] <= thres else 0)

            if label == 1:
                total_pos_dist += dists[-1]
                if dists[-1] <= thres:
                    tp += 1
                    if save_image:
                        save(save_image_dir, sid_left, sid_right, left, right, "tp", i)
                else:
                    fn += 1
                    if save_image:
                        save(save_image_dir, sid_left, sid_right, left, right, "fn", i)
            else:
                total_neg_dist += dists[-1]
                if dists[-1] > thres:
                    tn += 1
                    if save_image:
                        save(save_image_dir, sid_left, sid_right, left, right, "tn", i)
                else:
                    fp += 1
                    if save_image:
                        save(save_image_dir, sid_left, sid_right, left, right, "fp", i)

            pred = torch.max(embedding_left + embedding_right, dim=1)[1]
            bz, nc, h, w = model.module.features.shape
            before_dot = model.module.features.reshape((bz, nc, h * w))
            cams_ = []
            params = list(model.module.fc.parameters())[0]
            for ids, bd in enumerate(before_dot):
                weight = params[pred[ids]]
                cam = torch.matmul(weight, bd)
                cam_img = cam.reshape(h, w)
                cam_img = cam_img - torch.min(cam_img)
                cam_img = cam_img / torch.max(cam_img)
                cams_.append(cam_img)
            cams_ = torch.stack(cams_)
            cams_tot = torch.sum(cams_, dim=0)
            cams_tot = torch.unsqueeze(cams_tot.cpu().detach(), 0)
            cams.append(cams_tot)

    cams = torch.sum(torch.stack(cams), dim=0)
    cams = cams / torch.max(cams)

    gc.collect()

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pd": total_pos_dist,
        "nd": total_neg_dist,
        "dist": sum(dists) / len(dataloader),
        "loss": sum(losses) / (i + 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "cams": cams,
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
        "--weight-path",
        "-w",
        type=str,
        required=True,
        help="Path to the initial weight",
    )
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--image-dir",
        "-i",
        type=str,
        required=True,
        help="Path to the image directory",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=float,
        default=1,
        help="Batch size, defaults to 1",
    )
    parser.add_argument(
        "--thres",
        "-t",
        type=float,
        default=0.5,
        help="Distance threshold, defaults to 0.5",
    )
    parser.add_argument(
        "--suffix",
        "-s",
        type=str,
        default="",
        help="Suffix for output file/dir name",
    )
    parser.add_argument(
        "--mask",
        "-m",
        action="store_true",
        help="Mask images - need to specify mask directory",
    )
    parser.add_argument(
        "--mask-dir",
        "-md",
        type=str,
        default="",
        help="Path to the mask directory",
    )
    parser.add_argument(
        "--mask-inv",
        "-mi",
        action="store_true",
        help="Inversely mask images",
    )
    parser.add_argument(
        "--save-image",
        "-si",
        action="store_true",
        help="Save inferred images",
    )

    return parser.parse_args()


def check_args(args):
    if not os.path.isfile(args.init_weight_path):
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
    time_stamp = datetime.now().strftime("%m%d_%H%M%S")

    args = parse_arguments()
    check_args(args)
    print("Argument parsed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Detected device type: {device.type}")

    infer_dataloader = load_dataset(
        args.data_path,
        args.image_dir,
        args.batch_size,
        args.mask,
        args.mask_dir,
        args.mask_inv,
    )
    print("Inference dataset loaded.")

    model = load_model(args.backbone, args.weight_path)
    print("Model loaded.")

    result_path = f"tests/results_{args.backbone}_{time_stamp}"
    if args.suffix:
        result_path = f"tests/results_{args.backbone}_{args.suffix}_{time_stamp}"
    writer = SummaryWriter(result_path)
    writer.add_hparams(
        {"Weight path": args.weight_path, "Distance threshold": args.thres},
        {},
    )
    print(f"Results will be saved to {result_path}.")

    print("Inference preperation completed :)")

    save_image_dir = "1fc"
    if args.mask:
        save_image_dir = "1fc-mask"
    if args.mask_inv:
        save_image_dir = "1fc-mask-inv"
    infer_results = infer(
        infer_dataloader,
        model,
        ContrastiveLoss(),
        args.thres,
        args.save_image,
        save_image_dir,
        device,
    )
    print("Inference completed!")

    write_results(
        writer,
        model,
        infer_results["tp"],
        infer_results["tn"],
        infer_results["fp"],
        infer_results["fn"],
        infer_results["pd"],
        infer_results["nd"],
        infer_results["dist"],
        infer_results["loss"],
        infer_results["y_true"],
        infer_results["y_pred"],
        infer_results["cams"],
    )
    writer.flush()
    writer.close()
