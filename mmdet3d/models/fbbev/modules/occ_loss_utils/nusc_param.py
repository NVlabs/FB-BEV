import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# nusc_class_frequencies = np.array([57330862, 25985376, 1561108, 28862014, 196106643, 15920504,
#                 2158753, 26539491, 4004729, 34838681, 75173306, 2255027978, 50959399, 646022466, 869055679,
#                 1446141335, 1724391378, 2242961742295])

# nusc_class_frequencies = np.array([2242961742295, 25985376, 1561108, 28862014, 196106643, 15920504,
#                 2158753, 26539491, 4004729, 34838681, 75173306, 2255027978, 50959399, 646022466, 869055679,
#                 1446141335, 1724391378])

nusc_class_frequencies = np.array([
 944004,
 1897170,
 152386,
 2391677,
 16957802,
 724139,
 189027,
 2074468,
 413451,
 2384460,
 5916653,
 175883646,
 4275424,
 51393615,
 61411620,
 105975596,
 116424404,
 1892500630
 ])


# nusc_class_names = [
#     "noise",
#     "barrier",
#     "bicycle",
#     "bus",
#     "car",
#     "construction",
#     "motorcycle",
#     "pedestrian",
#     "trafficcone",
#     "trailer",
#     "truck",
#     "driveable_surface",
#     "other",
#     "sidewalk",
#     "terrain",
#     "mannade",
#     "vegetation",
#     "free",
# ]

nusc_class_names = [
    "empty", # 0
    "barrier", # 1
    "bicycle", # 2 
    "bus", # 3 
    "car", # 4
    "construction", # 5
    "motorcycle", # 6
    "pedestrian", # 7
    "trafficcone", # 8
    "trailer", # 9
    "truck", # 10
    "driveable_surface", # 11
    "other", # 12
    "sidewalk", # 13
    "terrain", # 14
    "mannade", # 15 
    "vegetation", # 16
]

# classname_to_color = {  # RGB.
#     0: (0, 0, 0),  # Black. noise
#     1: (112, 128, 144),  # Slategrey barrier
#     2: (220, 20, 60),  # Crimson bicycle
#     3: (255, 127, 80),  # Orangered bus
#     4: (255, 158, 0),  # Orange car
#     5: (233, 150, 70),  # Darksalmon construction
#     6: (255, 61, 99),  # Red motorcycle
#     7: (0, 0, 230),  # Blue pedestrian
#     8: (47, 79, 79),  # Darkslategrey trafficcone
#     9: (255, 140, 0),  # Darkorange trailer
#     10: (255, 99, 71),  # Tomato truck
#     11: (0, 207, 191),  # nuTonomy green driveable_surface
#     12: (175, 0, 75),  # flat other
#     13: (75, 0, 75),  # sidewalk
#     14: (112, 180, 60),  # terrain
#     15: (222, 184, 135),  # Burlywood mannade
#     16: (0, 175, 0),  # Green vegetation
# }
classname_to_color = {  # RGB.
    # 0: (0, 0, 0),  # Black. noise
    1: (112, 128, 144),  # Slategrey barrier
    2: (220, 20, 60),  # Crimson bicycle
    3: (255, 127, 80),  # Orangered bus
    4: (255, 158, 0),  # Orange car
    5: (233, 150, 70),  # Darksalmon construction
    6: (255, 61, 99),  # Red motorcycle
    7: (0, 0, 230),  # Blue pedestrian
    8: (47, 79, 79),  # Darkslategrey trafficcone
    9: (255, 140, 0),  # Darkorange trailer
    10: (255, 99, 71),  # Tomato truck
    11: (0, 207, 191),  # nuTonomy green driveable_surface
    12: (175, 0, 75),  # flat other
    13: (75, 0, 75),  # sidewalk
    14: (112, 180, 60),  # terrain
    15: (222, 184, 135),  # Burlywood mannade
    16: (0, 175, 0),  # Green vegetation
}

def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss
