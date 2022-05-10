import torch
import numpy as np
from torch.nn import functional as F
import cv2

"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""


def rot_aa(aa, rot):
    """Rotate axis angle parameters."""
    # pose parameters
    R = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ]
    )
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def rigid_tf_torch_batch(points, R, T):
    """
    Performs rigid transformation to incoming points but batched
    Q = (points*R.T) + T
    points: (batch, num, 3)
    R: (batch, 3, 3)
    T: (batch, 3, 1)
    out: (batch, num, 3)
    """
    points_out = torch.bmm(R, points.permute(0, 2, 1)) + T
    points_out = points_out.permute(0, 2, 1)
    return points_out


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(batch_size, 3, 3)
    return rotMat


def batch_aa2rot(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    assert len(axisang.shape) == 2
    assert axisang.shape[1] == 3
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def solve_rigid_tf_np(A: np.ndarray, B: np.ndarray):
    """
    “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. , May 1987
    Input: expects Nx3 matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector

    This function should be a fix for compute_rigid_tf when the det == -1
    """

    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def batch_solve_rigid_tf(A, B):
    """
    “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. , May 1987
    Input: expects BxNx3 matrix of points
    Returns R,t
    R = Bx3x3 rotation matrix
    t = Bx3x1 column vector
    """

    assert A.shape == B.shape
    A = A.permute(0, 2, 1)
    B = B.permute(0, 2, 1)

    batch, num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    _, num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = torch.mean(A, dim=2)
    centroid_B = torch.mean(B, dim=2)

    # ensure centroids are 3x1
    centroid_A = centroid_A.view(batch, -1, 1)
    centroid_B = centroid_B.view(batch, -1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = torch.bmm(Am, Bm.permute(0, 2, 1))

    # find rotation
    U, S, Vt = torch.svd(H)
    Vt = Vt.permute(0, 2, 1)
    R = torch.bmm(Vt.permute(0, 2, 1), U.permute(0, 2, 1))

    # special reflection case
    neg_idx = torch.det(R) < 0
    if neg_idx.sum() > 0:
        raise Exception(
            f"some rotation matrices are not orthogonal; make sure implementation is correct for such case: {neg_idx}"
        )
    Vt[neg_idx, 2, :] *= -1
    R[neg_idx, :, :] = torch.bmm(
        Vt[neg_idx].permute(0, 2, 1), U[neg_idx].permute(0, 2, 1)
    )

    t = torch.bmm(-R, centroid_A) + centroid_B
    return R, t


def rigid_tf_np(points, R, T):
    """
    Performs rigid transformation to incoming points
    Q = (points*R.T) + T
    points: (num, 3)
    R: (3, 3)
    T: (1, 3)

    out: (num, 3)
    """

    assert isinstance(points, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert isinstance(T, np.ndarray)
    assert len(points.shape) == 2
    assert points.shape[1] == 3
    assert R.shape == (3, 3)
    assert T.shape == (1, 3)
    points_new = np.matmul(R, points.T).T + T
    return points_new


def batch_rot2aa(Rs):
    assert len(Rs.shape) == 3
    assert Rs.shape[1] == Rs.shape[2]
    assert Rs.shape[1] == 3

    """
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);

        out_theta = std::acos(c);

        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """
    cos = 0.5 * (torch.stack([torch.trace(x) for x in Rs]) - 1)
    cos = torch.clamp(cos, -1, 1)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(x):
    rotmat = x.reshape(-1, 3, 3)
    rot6d = rotmat[:, :, :2].reshape(x.shape[0], -1)
    return rot6d


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = (
            torch.tensor([0, 0, 1], dtype=torch.float32, device=rotation_matrix.device)
            .reshape(1, 3, 1)
            .expand(rot_mat.shape[0], -1, -1)
        )
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix))
        )

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape
            )
        )
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape
            )
        )

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q


def perspective_to_weak_perspective_torch(
    perspective_camera,
    focal_length,
    img_res,
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    tx = perspective_camera[:, 0]
    ty = perspective_camera[:, 1]
    tz = perspective_camera[:, 2]

    weak_perspective_camera = torch.stack(
        [2 * focal_length / (img_res * tz + 1e-9), tx, ty],
        dim=-1,
    )
    return weak_perspective_camera


def convert_perspective_to_weak_perspective(
    perspective_camera,
    focal_length,
    img_res,
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    weak_perspective_camera = torch.stack(
        [
            2 * focal_length / (img_res * perspective_camera[:, 2] + 1e-9),
            perspective_camera[:, 0],
            perspective_camera[:, 1],
        ],
        dim=-1,
    )
    return weak_perspective_camera


def convert_weak_perspective_to_perspective(
    weak_perspective_camera, focal_length, img_res
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    # if isinstance(focal_length, torch.Tensor):
    #     focal_length = focal_length[:, 0]

    perspective_camera = torch.stack(
        [
            weak_perspective_camera[:, 1],
            weak_perspective_camera[:, 2],
            2 * focal_length / (img_res * weak_perspective_camera[:, 0] + 1e-9),
        ],
        dim=-1,
    )
    return perspective_camera


def get_default_cam_t(f, img_res):
    cam = torch.tensor([[5.0, 0.0, 0.0]])
    return convert_weak_perspective_to_perspective(cam, f, img_res)


def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

    return projected_points[:, :, :-1]


def weak_perspective_projection(
    points, rotation, weak_cam_params, focal_length, camera_center, img_res
):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length
    K[:, 1, 1] = focal_length
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    translation = convert_weak_perspective_to_perspective(
        weak_cam_params, focal_length, img_res
    )

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)

    return projected_points[:, :, :-1]


def estimate_translation_np(S, joints_2d, joints_conf, focal_length, img_size):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length

    f = np.array([focal_length[0], focal_length[1]])
    # optical center
    center = np.array([img_size[1] / 2.0, img_size[0] / 2.0])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(f, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array(
        [
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O - np.reshape(joints_2d, -1),
        ]
    ).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(
    S,
    joints_2d,
    focal_length,
    img_size,
    use_all_joints=False,
    rotation=None,
    pad_2d=False,
):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """
    if pad_2d:
        batch, num_pts = joints_2d.shape[:2]
        joints_2d_pad = torch.ones((batch, num_pts, 3))
        joints_2d_pad[:, :, :2] = joints_2d
        joints_2d_pad = joints_2d_pad.to(joints_2d.device)
        joints_2d = joints_2d_pad

    device = S.device

    if rotation is not None:
        S = torch.einsum("bij,bkj->bki", rotation, S)

    # Use only joints 25:49 (GT joints)
    if use_all_joints:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(
            S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size
        )
    return torch.from_numpy(trans).to(device)


def estimate_translation_cam(
    S, joints_2d, focal_length, img_size, use_all_joints=False, rotation=None
):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    def estimate_translation_np(S, joints_2d, joints_conf, focal_length, img_size):
        """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
        Input:
            S: (25, 3) 3D joint locations
            joints: (25, 3) 2D joint locations and confidence
        Returns:
            (3,) camera translation vector
        """

        num_joints = S.shape[0]
        # focal length
        f = np.array([focal_length[0], focal_length[1]])
        # optical center
        center = np.array([img_size[0] / 2.0, img_size[1] / 2.0])

        # transformations
        Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
        XY = np.reshape(S[:, 0:2], -1)
        O = np.tile(center, num_joints)
        F = np.tile(f, num_joints)
        weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

        # least squares
        Q = np.array(
            [
                F * np.tile(np.array([1, 0]), num_joints),
                F * np.tile(np.array([0, 1]), num_joints),
                O - np.reshape(joints_2d, -1),
            ]
        ).T
        c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

        # weighted least squares
        W = np.diagflat(weight2)
        Q = np.dot(W, Q)
        c = np.dot(W, c)

        # square matrix
        A = np.dot(Q.T, Q)
        b = np.dot(Q.T, c)

        # solution
        trans = np.linalg.solve(A, b)

        return trans

    device = S.device

    if rotation is not None:
        S = torch.einsum("bij,bkj->bki", rotation, S)

    # Use only joints 25:49 (GT joints)
    if use_all_joints:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(
            S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size
        )
    return torch.from_numpy(trans).to(device)


def get_coord_maps(size=56):
    xx_ones = torch.ones([1, size], dtype=torch.int32)
    xx_ones = xx_ones.unsqueeze(-1)

    xx_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    xx_range = xx_range.unsqueeze(1)

    xx_channel = torch.matmul(xx_ones, xx_range)
    xx_channel = xx_channel.unsqueeze(-1)

    yy_ones = torch.ones([1, size], dtype=torch.int32)
    yy_ones = yy_ones.unsqueeze(1)

    yy_range = torch.arange(size, dtype=torch.int32).unsqueeze(0)
    yy_range = yy_range.unsqueeze(-1)

    yy_channel = torch.matmul(yy_range, yy_ones)
    yy_channel = yy_channel.unsqueeze(-1)

    xx_channel = xx_channel.permute(0, 3, 1, 2)
    yy_channel = yy_channel.permute(0, 3, 1, 2)

    xx_channel = xx_channel.float() / (size - 1)
    yy_channel = yy_channel.float() / (size - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    out = torch.cat([xx_channel, yy_channel], dim=1)
    return out


def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)),
        axis=2,
    )

    return r_mat


def to_sphere(u, v):
    theta = 2 * np.pi * u
    phi = np.arccos(1 - 2 * v)
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    s = np.stack([cx, cy, cz])
    return s


def sample_on_sphere(range_u=(0, 1), range_v=(0, 1)):
    u = np.random.uniform(*range_u)
    v = np.random.uniform(*range_v)
    return to_sphere(u, v)


def sample_pose_on_sphere(range_v=(0, 1), range_u=(0, 1), radius=1, up=[0, 1, 0]):
    # sample location on unit sphere
    loc = sample_on_sphere(range_u, range_v)

    # sample radius if necessary
    if isinstance(radius, tuple):
        radius = np.random.uniform(*radius)

    loc = loc * radius
    R = look_at(loc, up=np.array(up))[0]

    RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
    RT = torch.Tensor(RT.astype(np.float32))
    return RT


def rectify_pose(camera_r, body_aa, rotate_x=False):
    body_r = batch_rodrigues(body_aa).reshape(-1, 3, 3)

    if rotate_x:
        rotate_x = torch.tensor([[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]])
        body_r = body_r @ rotate_x

    final_r = camera_r @ body_r
    body_aa = batch_rot2aa(final_r)
    return body_aa


def batch_euler2matrix(r):
    return quaternion_to_rotation_matrix(euler_to_quaternion(r))


def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1, 2))[..., :4].to(r.device)
    quaternion[..., 0] += cx * cy * cz - sx * sy * sz
    quaternion[..., 1] += cx * sy * sz + cy * cz * sx
    quaternion[..., 2] += cx * cz * sy - sx * cy * sz
    quaternion[..., 3] += cx * cy * sz + sx * cz * sy
    return quaternion


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def euler_angles_from_rotmat(R):
    """
    computer euler angles for rotation around x, y, z axis
    from rotation amtrix
    R: 4x4 rotation matrix
    https://www.gregslabaugh.net/publications/euler.pdf
    """
    r21 = np.round(R[:, 2, 0].item(), 4)
    if abs(r21) != 1:
        y_angle1 = -1 * torch.asin(R[:, 2, 0])
        y_angle2 = math.pi + torch.asin(R[:, 2, 0])
        cy1, cy2 = torch.cos(y_angle1), torch.cos(y_angle2)

        x_angle1 = torch.atan2(R[:, 2, 1] / cy1, R[:, 2, 2] / cy1)
        x_angle2 = torch.atan2(R[:, 2, 1] / cy2, R[:, 2, 2] / cy2)
        z_angle1 = torch.atan2(R[:, 1, 0] / cy1, R[:, 0, 0] / cy1)
        z_angle2 = torch.atan2(R[:, 1, 0] / cy2, R[:, 0, 0] / cy2)

        s1 = (x_angle1, y_angle1, z_angle1)
        s2 = (x_angle2, y_angle2, z_angle2)
        s = (s1, s2)

    else:
        z_angle = torch.tensor([0], device=R.device).float()
        if r21 == -1:
            y_angle = torch.tensor([math.pi / 2], device=R.device).float()
            x_angle = z_angle + torch.atan2(R[:, 0, 1], R[:, 0, 2])
        else:
            y_angle = -torch.tensor([math.pi / 2], device=R.device).float()
            x_angle = -z_angle + torch.atan2(-R[:, 0, 1], R[:, 0, 2])
        s = ((x_angle, y_angle, z_angle),)
    return s


def quaternion_raw_multiply(a, b):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_invert(quaternion):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    return quaternion * quaternion.new_tensor([1, -1, -1, -1])


def quaternion_apply(quaternion, point):
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def estimate_translation_k_np(S, joints_2d, joints_conf, K):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """
    num_joints = S.shape[0]
    # focal length

    focal = np.array([K[0, 0], K[1, 1]])
    # optical center
    center = np.array([K[0, 2], K[1, 2]])

    # transformations
    Z = np.reshape(np.tile(S[:, 2], (2, 1)).T, -1)
    XY = np.reshape(S[:, 0:2], -1)
    O = np.tile(center, num_joints)
    F = np.tile(focal, num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf), (2, 1)).T, -1)

    # least squares
    Q = np.array(
        [
            F * np.tile(np.array([1, 0]), num_joints),
            F * np.tile(np.array([0, 1]), num_joints),
            O - np.reshape(joints_2d, -1),
        ]
    ).T
    c = (np.reshape(joints_2d, -1) - O) * Z - F * XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W, Q)
    c = np.dot(W, c)

    # square matrix
    A = np.dot(Q.T, Q)
    b = np.dot(Q.T, c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation_k(
    S,
    joints_2d,
    K,
    use_all_joints=False,
    rotation=None,
    pad_2d=False,
):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """
    if pad_2d:
        batch, num_pts = joints_2d.shape[:2]
        joints_2d_pad = torch.ones((batch, num_pts, 3))
        joints_2d_pad[:, :, :2] = joints_2d
        joints_2d_pad = joints_2d_pad.to(joints_2d.device)
        joints_2d = joints_2d_pad

    device = S.device

    if rotation is not None:
        S = torch.einsum("bij,bkj->bki", rotation, S)

    # Use only joints 25:49 (GT joints)
    if use_all_joints:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        K_i = K[i]
        trans[i] = estimate_translation_k_np(S_i, joints_i, conf_i, K_i)
    return torch.from_numpy(trans).to(device)


def weak_perspective_to_perspective_torch(
    weak_perspective_camera, focal_length, img_res
):
    # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz]
    # in 3D given the bounding box size
    # This camera translation can be used in a full perspective projection
    s = weak_perspective_camera[:, 0]
    tx = weak_perspective_camera[:, 1]
    ty = weak_perspective_camera[:, 2]
    perspective_camera = torch.stack(
        [
            tx,
            ty,
            2 * focal_length / (img_res * s + 1e-9),
        ],
        dim=-1,
    )
    return perspective_camera


def to_homo(x):
    assert isinstance(x, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x.shape[1] == 3
    assert len(x.shape) == 2
    batch_size = x.shape[0]
    x_homo = torch.ones(batch_size, 4, device=x.device)
    x_homo[:, :3] = x.clone()
    return x_homo


def to_homo_batch(x):
    assert isinstance(x, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x.shape[2] == 3
    assert len(x.shape) == 3
    batch_size = x.shape[0]
    num_pts = x.shape[1]
    x_homo = torch.ones(batch_size, num_pts, 4, device=x.device)
    x_homo[:, :, :3] = x.clone()
    return x_homo


def to_xyz(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[1] == 4
    assert len(x_homo.shape) == 2
    batch_size = x_homo.shape[0]
    x = torch.ones(batch_size, 3, device=x_homo.device)
    x = x_homo[:, :3] / x_homo[:, 3:4]
    return x


def to_xyz_batch(x_homo):
    """
    Input: (B, N, 4)
    Ouput: (B, N, 3)
    """
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 4
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 3, device=x_homo.device)
    x = x_homo[:, :, :3] / x_homo[:, :, 3:4]
    return x


def to_xy(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[1] == 3
    assert len(x_homo.shape) == 2
    batch_size = x_homo.shape[0]
    x = torch.ones(batch_size, 2, device=x_homo.device)
    x = x_homo[:, :2] / x_homo[:, 2:3]
    return x


def to_xy_batch(x_homo):
    assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert x_homo.shape[2] == 3
    assert len(x_homo.shape) == 3
    batch_size = x_homo.shape[0]
    num_pts = x_homo.shape[1]
    x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
    x = x_homo[:, :, :2] / x_homo[:, :, 2:3]
    return x


def transform_points(world2cam_mat, pts):
    """
    Map points from one coord to another based on the 4x4 matrix.
    e.g., map points from world to camera coord.
    pts: (N, 3), in METERS!!
    world2cam_mat: (4, 4)
    Output: points in cam coord (N, 3)
    We follow this convention:
    | R T |   |pt|
    | 0 1 | * | 1|
    i.e. we rotate first then translate as T is the camera translation not position.
    """
    assert isinstance(pts, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(world2cam_mat, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert world2cam_mat.shape == (4, 4)
    assert len(pts.shape) == 2
    assert pts.shape[1] == 3
    pts_homo = to_homo(pts)

    # mocap to cam
    pts_cam_homo = torch.matmul(world2cam_mat, pts_homo.T).T
    pts_cam = to_xyz(pts_cam_homo)

    assert pts_cam.shape[1] == 3
    return pts_cam


def transform_points_batch(world2cam_mat, pts):
    """
    Map points from one coord to another based on the 4x4 matrix.
    e.g., map points from world to camera coord.
    pts: (B, N, 3), in METERS!!
    world2cam_mat: (B, 4, 4)
    Output: points in cam coord (B, N, 3)
    We follow this convention:
    | R T |   |pt|
    | 0 1 | * | 1|
    i.e. we rotate first then translate as T is the camera translation not position.
    """
    assert isinstance(pts, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(world2cam_mat, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert world2cam_mat.shape[1:] == (4, 4)
    assert len(pts.shape) == 3
    assert pts.shape[2] == 3
    batch_size = pts.shape[0]
    pts_homo = to_homo_batch(pts)

    # mocap to cam
    pts_cam_homo = torch.bmm(world2cam_mat, pts_homo.permute(0, 2, 1)).permute(0, 2, 1)
    pts_cam = to_xyz_batch(pts_cam_homo)

    assert pts_cam.shape[2] == 3
    return pts_cam


def project2d_batch(K, pts_cam):
    """
    K: (B, 3, 3)
    pts_cam: (B, N, 3)
    """

    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape[1:] == (3, 3)
    assert pts_cam.shape[2] == 3
    assert len(pts_cam.shape) == 3
    pts2d_homo = torch.bmm(K, pts_cam.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)
    return pts2d


def project2d(K, pts_cam):
    assert isinstance(K, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert isinstance(pts_cam, (torch.FloatTensor, torch.cuda.FloatTensor))
    assert K.shape == (3, 3)
    assert pts_cam.shape[1] == 3
    assert len(pts_cam.shape) == 2
    pts2d_homo = torch.matmul(K, pts_cam.T).T
    pts2d = to_xy(pts2d_homo)
    return pts2d
