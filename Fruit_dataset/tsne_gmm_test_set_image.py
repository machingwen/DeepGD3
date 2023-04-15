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

def compute_plot_coordinates(image,x,y,image_centers_area_size,offset):

    image_height, image_width, _ = image.shape
    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset
    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    xmin = center_x - int(image_width / 2)
    ymin = center_y - int(image_height / 2)
    xmax = xmin + image_width
    ymax = ymin + image_height

    return xmin, ymin, xmax, ymax    

def plot_scatter_all(args, X, y, X_test, y_test):    
    tx = X[:, 0]
    ty = X[:, 1]
    
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    for idx, num in enumerate(set(y)):
        if idx != num:
            item = np.where(np.asarray(y) == num)[0]
            for i in item:
                y[i] = idx
    
    colors_per_class = {}
    for i in set(y):
        colors_per_class[i] = [random.randrange(0, 255) for i in range(3)]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
    
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        if idx == 0:
            ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=380, edgecolors="k", linewidths=3) # good new component
        else:
            ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=380, edgecolors="k", linewidths=3) # bad new component
    
    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    tnse_points_path= os.path.join(args["output_dir"], "{}/{}_train+test_stage1_component_HBE+_component_tsne.pdf".format(args["random_seed"], args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)
    
def plot_scatter(args, X, y, X_test, y_test):    
    tx = X[:, 0]
    ty = X[:, 1]
    tx_test = X_test[:, 0]
    ty_test = X_test[:, 1]
    
    figure = plt.figure()
    ax = plt.subplot(aspect="equal")
    
    colors_per_class = {}
    colors_per_class[0] = [0, 0, 255]
    colors_per_class[1] = [255, 0, 0]
    
    for label in colors_per_class:
        indices = [i for i, l in enumerate(y) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        ax.scatter(current_tx, current_ty, lw=0, s=40, c=color, label=label)
        
    colors_per_class = {}
    colors_per_class[len(set(y))] = [0, 0, 255]
    colors_per_class[len(set(y))+1] = [0, 255, 0]
    for idx, label in enumerate(colors_per_class):
        indices = [i for i, l in enumerate(y_test) if l == label]
        current_tx = np.take(tx_test, indices)
        current_ty = np.take(ty_test, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255        
        if idx == 0:
            ax.scatter(current_tx, current_ty, marker="o", c=color, label=label, s=380, edgecolors="k", linewidths=3) # good new component
        else:
            ax.scatter(current_tx, current_ty, marker="X", c=color, label=label, s=380, edgecolors="k", linewidths=3) # bad new component
    
    ax.axis("tight")
    ax.axis("off")
    ax.legend(loc='best')
    tnse_points_path= os.path.join(args["output_dir"], "{}/{}_train+test_stage1_component_HBE+_class_tsne.pdf".format(args["random_seed"], args["embedding_layer"]))
    plt.savefig(tnse_points_path, dpi=150)

    
def set_model(args):
    model = SupConMobileNetV3Large(name='mobilenetv3_large')
    ckpt = torch.load(args["checkpoint_path"], map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
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
        default="shared_embedding",
        help="Which embedding to visualization( encoder or shared_embedding)"
    )
    parser.add_argument(
        "--name",
        type=int,
        default=15,
        help="Test component name"
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
    
    training_loader, _, test_loader = CreateTSNEdataset_regroup_fruit_8(args["random_seed"])
    #test_loader = CreateTSNEdatasetRelabel_OneComponent(args["random_seed"], componentName=args["name"])

    # Calculate embeddings from images in reference set
    start = time.time()
    embeddings_train, labels_train, _, name_list_train, _, _ = get_features_trained_weight(model, training_loader, embedding_layer=args["embedding_layer"], tsne=True)
    embeddings_test, labels_test, _, name_list_test, _, _ = get_features_trained_weight(model, test_loader, embedding_layer=args["embedding_layer"], tsne=True)   
    end = time.time()
    logging.info(f"Calculated {len(embeddings_train)+len(embeddings_test)} embeddings: {end - start} second")
    
    
    # Train + Test set
    embeddings = np.concatenate((embeddings_train, embeddings_test), axis=0)
    
    component_labels_test_mapping = []
    for lb in labels_test:
        if lb == 0:
            component_labels_test_mapping.append(len(set(name_list_train)))
        if lb == 1:
            component_labels_test_mapping.append(len(set(name_list_train))+1)
    
    class_labels_test_mapping = []
    for lb in labels_test:
        if lb == 0:
            class_labels_test_mapping.append(len(set(labels_train)))
        if lb == 1:
            class_labels_test_mapping.append(len(set(labels_train))+1)
    
    # Init T-SNE
    tsne = TSNE(n_components=2, random_state=12345, perplexity=35, learning_rate=200, n_iter=2000, n_jobs=-1)
    X_transformed = tsne.fit_transform(embeddings)
    
    tsne_train = X_transformed[0:len(embeddings_train)]
    tsne_test = X_transformed[len(embeddings_train)::]    
    plot_scatter_all(args, tsne_train, name_list_train, tsne_test, component_labels_test_mapping)
    
    labels = labels_train+labels_test
    plot_scatter(args, tsne_train, labels_train, tsne_test, class_labels_test_mapping)