# https://github.com/QuocThangNguyen/deep-metric-learning-tsinghua-dogs/blob/master/src/scripts/visualize_tsne.py
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects
import seaborn as sns

import argparse
import os
from multiprocessing import cpu_count
import time
from pprint import pformat
import logging
import sys
from typing import Dict, Any, List, Tuple
from networks.mobilenetv3 import SupConMobileNetV3Large
from util import *
import torch.backends.cudnn as cudnn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',
    datefmt='%y-%b-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

plt.rcParams['figure.figsize'] = (32, 32)
plt.rcParams['figure.dpi'] = 150

def set_random_seed(seed: int) -> None:
    """
    Set random seed for package random, numpy and pytorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_scatter(X: np.ndarray, y: np.ndarray, idx_to_class: Dict[int, str]):
    """
    Render a scatter plot with as many as unique colors as the number of classes.
    Source: https://www.datacamp.com/community/tutorials/introduction-t-sne
    Args:
        X: 2-D array output of t-sne algorithm
        y: 1-D array containing the labels of the dataset
        idx_to_class: dictionary mapping from class's index to class's name
    Return:
        Tuple
    """
    y = np.asarray(y)
    # Choose a color palette with seaborn
    n_classes: int = len(idx_to_class)
    palette: np.ndarray = np.array(sns.color_palette("hls", n_classes))

    # Create a scatter plot
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    scatter = ax.scatter(X[:, 0], X[:, 1], lw=0, s=40, c=palette[y.astype(np.int32)])
    ax.axis("tight")
    ax.axis("off")

    # Add the labels for each sample corresponding to the label
    for i in range(n_classes):
        # Position of each label at median of data points
        x_text, y_text = np.median(X[y == i, :], axis=0)
        text = ax.text(x_text, y_text, str(i), fontsize=20)
        text.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])

    # Add legends for each class
    annotations: List = []
    for i in range(n_classes):
        circle = Line2D([0], [0], marker='o', color=palette[i], label=f"{i}: {idx_to_class[i]}")
        annotations.append(circle)
    plt.legend(handles=annotations, loc="best")

    return figure, ax, scatter


def set_model(args):
    model = SupConMobileNetV3Large(name='mobilenetv3_large')
    ckpt = torch.load(args["checkpoint_path"], map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}

    if torch.cuda.is_available():
        #model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True
        model.load_state_dict(state_dict)
        
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing embeddings with T-SNE")

    parser.add_argument(
        "-c", "--checkpoint_path",
        type=str,
        default ="",      # checkpoint.pth.tar
        help="Path to model's checkpoint."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="output/",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--embedding_layer",
        type=str,
        default="encoder",
        help="Which embedding to visualization( encoder, component_embedding or cls_embedding)"
    )
    parser.add_argument(
        "-n", "--n_images",
        type=int,
        default=2000,
        help="Number of random images to visualize"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1,
        help="Random seed"
    )
    parser.add_argument('--relabel', action='store_true', help='relabel dataset')
    args: Dict[str, Any] = vars(parser.parse_args())

    set_random_seed(args["random_seed"])

    # Create output directory if not exists
    if not os.path.isdir(args["output_dir"]):
        os.makedirs(args["output_dir"])
        logging.info(f"Created output directory {args['output_dir']}")

    # Initialize device
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Initialized device {device}")

    # Load model's checkpoint
    loc = 'cuda:0'
    
    checkpoint_path: str = args["checkpoint_path"]
    checkpoint: Dict[str, Any] = torch.load(checkpoint_path, map_location="cuda:0")
    logging.info(f"Loaded checkpoint at {args['checkpoint_path']}")

    # Intialize model
    model = set_model(args)

    # Initialize dataset and dataloader
    
    training_loader, validation_loader, test_loader = CreateTSNEdataset_regroup_fruit_8(args["random_seed"])
#     input_size ,num_classes ,train_com_loader, train_cls_loader, val_loader, val_com_loader, test_loader, train_com_df, val_df, test_df = get_fruit_8(opt.seed)
    
#     val_loader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=opt.batch_size, shuffle=False,
#         num_workers=opt.num_workers, pin_memory=True)

    # Calculate embeddings from images in reference set
    start = time.time()
    embeddings_train, labels_train, image_paths_train, name_list_train, full_name_list_train, tsne_label_list_train = get_features_trained_weight(model, training_loader, embedding_layer=args["embedding_layer"], tsne=True)

    embeddings_val, labels_val, image_paths_val, name_list_val, full_name_list_val, tsne_label_list_val = get_features_trained_weight(model, validation_loader, embedding_layer=args["embedding_layer"], tsne=True)
    
    embeddings_test, labels_test, image_paths_test, name_list_test, full_name_list_test, tsne_label_list_test = get_features_trained_weight(model, test_loader, embedding_layer=args["embedding_layer"], tsne=True)
    
    end = time.time()
    logging.info(f"Calculated {len(embeddings_train)+len(embeddings_val)+len(embeddings_test)} embeddings: {end - start} second")

    # Init T-SNE
    tsne = TSNE(n_components=2, random_state=12345, perplexity=35, learning_rate=200, n_iter=2000, n_jobs=-1)
    
    # Train + Val set
#     embeddings = np.concatenate((embeddings_train, embeddings_val), axis=0)
#     labels = labels_train + labels_val
#     image_paths = image_paths_train + image_paths_val
#     name_list = name_list_train + name_list_val
#     full_name_list = full_name_list_train + full_name_list_val
#     tsne_label_list = tsne_label_list_train + tsne_label_list_val

#     X_transformed = tsne.fit_transform(embeddings)
    
#     idx_to_class: Dict[int, str] = {
#         #idx: class_name.split("-")[-1]
#         idx: class_name
#         for idx, class_name in zip(
#             training_loader.dataset.dataframe['component_name'].values.tolist()+validation_loader.dataset.dataframe['component_name'].values.tolist(), 
#             training_loader.dataset.dataframe['component_full_name'].values.tolist()+validation_loader.dataset.dataframe['component_full_name'].values.tolist())
#     }
#     if args["relabel"]:
#         for idx, num in enumerate(set(name_list)):
#             if idx != num:
#                 item = np.where(np.asarray(name_list) == num)[0]
#                 for i in item:
#                     name_list[i] = idx
#         for idx, key in enumerate(sorted(list(idx_to_class))):
#             if idx != key:
#                 idx_to_class[idx] = idx_to_class.pop(key)

#     plot_scatter(X_transformed, name_list, idx_to_class)
#     tnse_points_path: str = os.path.join(args["output_dir"], "{}/{}_train+val_HBE+_component_tsne{}.pdf".format(args["random_seed"], args["embedding_layer"], args["checkpoint_path"].split('/')[-1]))
#     plt.savefig(tnse_points_path, dpi=150)
#     idx_to_class = {0: 'good', 1: 'bad'}
#     plot_scatter(X_transformed, labels, idx_to_class)
#     tnse_points_path: str = os.path.join(args["output_dir"], "{}/{}_train+val_HBE+_good_bad_tsne_{}.pdf".format(args["random_seed"], args["embedding_layer"], args["checkpoint_path"].split('/')[-1]))
#     plt.savefig(tnse_points_path, dpi=150)    
#     logging.info(f"Plot is saved at: {tnse_points_path}")
    
    
    # Train set
    embeddings = embeddings_train
    labels = labels_train
    image_paths = image_paths_train
    name_list = name_list_train
    full_name_list = full_name_list_train
    tsne_label_list = tsne_label_list_train

    X_transformed = tsne.fit_transform(embeddings)

    # Visualize T-SNE in points
    idx_to_class: Dict[int, str] = {
        #idx: class_name.split("-")[-1]
        idx: class_name
        for idx, class_name in zip(
            training_loader.dataset.dataframe['component_name'].values.tolist(), 
            training_loader.dataset.dataframe['component_full_name'].values.tolist())
    }
    if args["relabel"]:
        for idx, num in enumerate(set(name_list)):
            if idx != num:
                item = np.where(np.asarray(name_list) == num)[0]
                for i in item:
                    name_list[i] = idx
        for idx, key in enumerate(sorted(list(idx_to_class))):
            if idx != key:
                idx_to_class[idx] = idx_to_class.pop(key)
        
    plot_scatter(X_transformed, name_list, idx_to_class)
    tnse_points_path: str = os.path.join(args["output_dir"], "{}/{}_train_HBE+_component_tsne{}.pdf".format(args["random_seed"], args["embedding_layer"], args["checkpoint_path"].split('/')[-1]))
    plt.savefig(tnse_points_path, dpi=150)
    
    
    idx_to_class = {0: 'good', 1: 'bad'}
    plot_scatter(X_transformed, labels, idx_to_class)
    tnse_points_path: str = os.path.join(args["output_dir"], "{}/{}_train_HBE+_good_bad_tsne_{}.pdf".format(args["random_seed"], args["embedding_layer"], args["checkpoint_path"].split('/')[-1]))
    plt.savefig(tnse_points_path, dpi=150)
    
    logging.info(f"Plot is saved at: {tnse_points_path}")
    
    # Val set
    embeddings = embeddings_val
    labels = labels_val
    image_paths = image_paths_val
    name_list = name_list_val
    full_name_list = full_name_list_val
    tsne_label_list = tsne_label_list_val

    X_transformed = tsne.fit_transform(embeddings)

    # Visualize T-SNE in points
    idx_to_class: Dict[int, str] = {
        #idx: class_name.split("-")[-1]
        idx: class_name
        for idx, class_name in zip(
            validation_loader.dataset.dataframe['component_name'].values.tolist(), 
            validation_loader.dataset.dataframe['component_full_name'].values.tolist())
    }
    if args["relabel"]:
        for idx, num in enumerate(set(name_list)):
            if idx != num:
                item = np.where(np.asarray(name_list) == num)[0]
                for i in item:
                    name_list[i] = idx
        for idx, key in enumerate(sorted(list(idx_to_class))):
            if idx != key:
                idx_to_class[idx] = idx_to_class.pop(key)
        
    plot_scatter(X_transformed, name_list, idx_to_class)
    tnse_points_path: str = os.path.join(args["output_dir"], "{}/{}_val_HBE+_component_tsne{}.pdf".format(args["random_seed"], args["embedding_layer"], args["checkpoint_path"].split('/')[-1]))
    plt.savefig(tnse_points_path, dpi=150)
    idx_to_class = {0: 'good', 1: 'bad'}
    plot_scatter(X_transformed, labels, idx_to_class)
    tnse_points_path: str = os.path.join(args["output_dir"], "{}/{}_val_HBE+_good_bad_tsne_{}.pdf".format(args["random_seed"], args["embedding_layer"], args["checkpoint_path"].split('/')[-1]))
    plt.savefig(tnse_points_path, dpi=150)
    
    logging.info(f"Plot is saved at: {tnse_points_path}")
    
    # Test set
    embeddings = embeddings_test
    labels = labels_test
    image_paths = image_paths_test
    name_list = name_list_test
    full_name_list = full_name_list_test
    tsne_label_list = tsne_label_list_test

    X_transformed = tsne.fit_transform(embeddings)

    # Visualize T-SNE in points
    idx_to_class: Dict[int, str] = {
        #idx: class_name.split("-")[-1]
        idx: class_name
        for idx, class_name in zip(
            test_loader.dataset.dataframe['component_name'].values.tolist(), 
            test_loader.dataset.dataframe['component_full_name'].values.tolist())
    }
    if args["relabel"]:
        for idx, num in enumerate(set(name_list)):
            if idx != num:
                item = np.where(np.asarray(name_list) == num)[0]
                for i in item:
                    name_list[i] = idx
        for idx, key in enumerate(sorted(list(idx_to_class))):
            if idx != key:
                idx_to_class[idx] = idx_to_class.pop(key)
        
    plot_scatter(X_transformed, name_list, idx_to_class)
    tnse_points_path: str = os.path.join(args["output_dir"], "{}/{}_test_HBE+_component_tsne{}.pdf".format(args["random_seed"], args["embedding_layer"], args["checkpoint_path"].split('/')[-1]))
    plt.savefig(tnse_points_path, dpi=150)
    idx_to_class = {0: 'good', 1: 'bad'}
    plot_scatter(X_transformed, labels, idx_to_class)
    tnse_points_path: str = os.path.join(args["output_dir"], "{}/{}_test_HBE+_good_bad_tsne_{}.pdf".format(args["random_seed"], args["embedding_layer"], args["checkpoint_path"].split('/')[-1]))
    plt.savefig(tnse_points_path, dpi=150)    
    logging.info(f"Plot is saved at: {tnse_points_path}")