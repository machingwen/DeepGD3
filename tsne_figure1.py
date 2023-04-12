# https://github.com/QuocThangNguyen/deep-metric-learning-tsinghua-dogs/blob/master/src/scripts/visualize_tsne.py
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
# from tsnecuda import TSNE
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

def scale_to_01_range(x: np.ndarray) -> float:
    """
    Scale and move the coordinates so they fit [0; 1] range
    """
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def resize_image(image: np.ndarray, max_image_size: int) -> np.ndarray:
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_border(image: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    height, width, _ = image.shape
    # get the color corresponding to image class
    image = cv2.rectangle(image, (0, 0), (width - 1, height - 1), color, thickness=10)
    return image


def compute_plot_coordinates(image: np.ndarray,
                             x: int,
                             y: int,
                             image_centers_area_size: int,
                             offset: int
                             ) -> Tuple[int, int, int, int]:

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


def plot_tsne_images(X: np.ndarray,
                     image_paths: List[str],
                     labels: List[int],
                     plot_size=2000,
                     max_image_size=224
                     ) -> np.ndarray:

    n_classes: int = np.unique(labels).max() + 1
    palette: np.ndarray = (np.array(sns.color_palette("hls", n_classes)) * 255).astype(np.int32)

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    x_scaled: np.ndarray = X[:, 0]
    y_scaled: np.ndarray = X[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    x_scaled = scale_to_01_range(x_scaled)
    y_scaled = scale_to_01_range(y_scaled)

    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset: int = max_image_size // 2
    image_centers_area_size: int = plot_size - 2 * offset

    plot: np.ndarray = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in zip(image_paths, labels, x_scaled, y_scaled):
        image: np.ndarray = cv2.imread(image_path)
        # scale the image to put it to the plot
        image = resize_image(image, max_image_size)
        # draw a rectangle with a color corresponding to the image class
        color = palette[label].tolist()
        image = draw_border(image, color)
        # compute the coordinates of the image on the scaled plot visualization
        xmin, ymin, xmax, ymax = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)
        # put the image to its TSNE coordinates using numpy subarray indices
        plot[ymin:ymax, xmin:xmax, :] = image

    return plot


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
    parser.add_argument(
        "--componentName",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument('--mlp', action='store_true', help='Multiple FC for embedding layer')
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
    test_loader_good, test_loader_bad = CreateTSNEdatasetRelabel_OneComponent(args["random_seed"], args["componentName"])

    start = time.time()
    
    embeddings_good, labels_good, image_paths_good, name_list_good, full_name_list_good, tsne_label_list_good = get_features_trained_weight(model, test_loader_good, embedding_layer=args["embedding_layer"], tsne=True)
    
    embeddings_bad, labels_bad, image_paths_bad, name_list_bad, full_name_list_bad, tsne_label_list_bad = get_features_trained_weight(model, test_loader_bad, embedding_layer=args["embedding_layer"], tsne=True)
    
    end = time.time()
    try:
        logging.info(f"Calculated {len(embeddings_good)+len(embeddings_bad)} embeddings: {end - start} second")
    except:
        logging.info(f"Calculated {len(embeddings_good)} embeddings: {end - start} second")

    # Init T-SNE
    tsne = TSNE(n_components=2, random_state=12345, perplexity=35, learning_rate=200, n_iter=2000, n_jobs=-1)
    
    try:
        embeddings = np.concatenate((embeddings_good, embeddings_bad), axis=0)
        labels = labels_good + labels_bad
        image_paths = image_paths_good + image_paths_bad
        name_list = name_list_good + name_list_bad
        full_name_list = full_name_list_good + full_name_list_bad
        tsne_label_list = tsne_label_list_good + tsne_label_list_bad
    except:
        embeddings = embeddings_good
        labels = labels_good
        image_paths = image_paths_good
        name_list = name_list_good
        full_name_list = full_name_list_good
        tsne_label_list = tsne_label_list_good
        
    X_transformed = tsne.fit_transform(embeddings)
    # Visualize T-SNE in points
    idx_to_class: Dict[int, str] = {
        #idx: class_name.split("-")[-1]
        idx: class_name
        for idx, class_name in zip(
            test_loader_good.dataset.dataframe['component_name'].values.tolist()+test_loader_bad.dataset.dataframe['component_name'].values.tolist(), 
            test_loader_good.dataset.dataframe['component_full_name'].values.tolist()+test_loader_bad.dataset.dataframe['component_full_name'].values.tolist())
    }
    for idx, num in enumerate(set(name_list)):
        if idx != num:
            item = np.where(np.asarray(name_list) == num)[0]
            for i in item:
                name_list[i] = idx
    for idx, key in enumerate(sorted(list(idx_to_class))):
        if idx != key:
            idx_to_class[idx] = idx_to_class.pop(key)  
            
    idx_to_class = {0: 'good', 1: 'bad'}
    plot_scatter(X_transformed, labels, idx_to_class)
    tsne_points_path = os.path.join(args["output_dir"], "{}/{}_HybridExpert+_component_{}_good_bad_tsne_points.pdf".format(args["random_seed"], args["embedding_layer"], args["componentName"]))
    plt.savefig(tsne_points_path, dpi=150)    
    logging.info(f"Plot is saved at: {tsne_points_path}")