import open3d as o3d
import numpy as np

def query_points_from_voxels(pred, gt, img_metas):
    # pred, [tensor of shape (num_class, x, y, z)]: predicted classes
    # gt, [tensor of shape (batch, num_points)]: target points with semantic labels
    
    # logits to pred cls_id
    pred = np.argmax(pred.detach().cpu().numpy(), axis=0)
    gt_ = gt.detach().cpu().numpy()
    
    pred_fore_mask = pred > 0
    if pred_fore_mask.sum() == 0:
        return None
    
    # select foreground 3d voxel vertex
    x = np.linspace(0, pred.shape[0] - 1, pred.shape[0])
    y = np.linspace(0, pred.shape[1] - 1, pred.shape[1])
    z = np.linspace(0, pred.shape[2] - 1, pred.shape[2])
    X, Y, Z = np.meshgrid(x, y, z,  indexing='ij')
    vv = np.stack([X, Y, Z], axis=-1)
    
    # foreground predictions & coordinates
    pred = pred[pred_fore_mask]
    vv = vv[pred_fore_mask]
    
    vv[:, 0] = (vv[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
    vv[:, 1] = (vv[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
    vv[:, 2] = (vv[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vv)
    
    # for every lidar point, search its nearest *foreground* voxel vertex as the semantic prediction
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    indices = []
    for vert in gt_[:, :3]:
        _, inds, _ = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
    
    pred_valid = pred[np.array(indices)]
    
    return pred_valid
    