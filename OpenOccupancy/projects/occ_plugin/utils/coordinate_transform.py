
import torch

def coarse_to_fine_coordinates(coarse_cor, ratio, topk=30000):
    """
    Args:
        coarse_cor (torch.Tensor): [3, N]"""

    fine_cor = coarse_cor * ratio
    fine_cor = fine_cor[None].repeat(ratio**3, 1, 1)  # [8, 3, N]

    device = fine_cor.device
    value = torch.meshgrid([torch.arange(ratio).to(device), torch.arange(ratio).to(device), torch.arange(ratio).to(device)])
    value = torch.stack(value, dim=3).reshape(-1, 3)

    fine_cor = fine_cor + value[:,:,None]

    if fine_cor.shape[-1] < topk:
        return fine_cor.permute(1,0,2).reshape(3,-1)
    else:
        fine_cor = fine_cor[:,:,torch.randperm(fine_cor.shape[-1])[:topk]]
        return fine_cor.permute(1,0,2).reshape(3,-1)



def project_points_on_img(points, rots, trans, intrins, post_rots, post_trans, bda_mat, pts_range,
                        W_img, H_img, W_occ, H_occ, D_occ):
    with torch.no_grad():
        voxel_size = ((pts_range[3:] - pts_range[:3]) / torch.tensor([W_occ-1, H_occ-1, D_occ-1])).to(points.device)
        points = points * voxel_size[None, None] + pts_range[:3][None, None].to(points.device)

        # project 3D point cloud (after bev-aug) onto multi-view images for corresponding 2D coordinates
        inv_bda = bda_mat.inverse()
        points = (inv_bda @ points.unsqueeze(-1)).squeeze(-1)
        
        # from lidar to camera
        points = points.view(-1, 1, 3)
        points = points - trans.view(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / (points_d + 1e-5)
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[..., :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)

        points_uv[..., 0] = (points_uv[..., 0] / (W_img-1) - 0.5) * 2
        points_uv[..., 1] = (points_uv[..., 1] / (H_img-1) - 0.5) * 2

        mask = (points_d[..., 0] > 1e-5) \
            & (points_uv[..., 0] > -1) & (points_uv[..., 0] < 1) \
            & (points_uv[..., 1] > -1) & (points_uv[..., 1] < 1)
    
    return points_uv.permute(2,1,0,3), mask