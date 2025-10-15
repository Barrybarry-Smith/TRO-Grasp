import torch
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.transforms import axis_angle_to_matrix
from theseus.geometry.so3 import SO3

def matrix_to_euler(matrix):
    device = matrix.device
    # forward_kinematics() requires intrinsic euler ('XYZ')
    euler = Rotation.from_matrix(matrix.cpu().numpy()).as_euler('XYZ')
    return torch.tensor(euler, dtype=torch.float32, device=device)

def euler_to_matrix(euler):
    device = euler.device
    matrix = Rotation.from_euler('XYZ', euler.cpu().numpy()).as_matrix()
    return torch.tensor(matrix, dtype=torch.float32, device=device)

def matrix_to_rot6d(matrix):
    return matrix.T.reshape(9)[:6]

def rot6d_to_matrix(rot6d):
    x = normalize(rot6d[..., 0:3])
    y = normalize(rot6d[..., 3:6])
    a = normalize(x + y)
    b = normalize(x - y)
    x = normalize(a + b)
    y = normalize(a - b)
    z = normalize(torch.cross(x, y, dim=-1))
    matrix = torch.stack([x, y, z], dim=-2).mT
    return matrix

def euler_to_rot6d(euler):
    matrix = euler_to_matrix(euler)
    return matrix_to_rot6d(matrix)

def rot6d_to_euler(rot6d):
    matrix = rot6d_to_matrix(rot6d)
    return matrix_to_euler(matrix)

def axisangle_to_matrix(axis, angle):
    (x, y, z), c, s = axis, torch.cos(angle), torch.sin(angle)
    return torch.tensor([
        [(1 - c) * x * x + c, (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
        [(1 - c) * x * y + s * z, (1 - c) * y * y + c, (1 - c) * y * z - s * x],
        [(1 - c) * x * z - s * y, (1 - c) * y * z + s * x, (1 - c) * z * z + c]
    ])

def euler_to_quaternion(euler):
    device = euler.device
    quaternion = Rotation.from_euler('XYZ', euler.cpu().numpy()).as_quat()
    return torch.tensor(quaternion, dtype=torch.float32, device=device)

def normalize(v):
    return v / torch.norm(v, dim=-1, keepdim=True)

def q_euler_to_q_rot6d(q_euler):
    return torch.cat([q_euler[..., :3], euler_to_rot6d(q_euler[..., 3:6]), q_euler[..., 6:]], dim=-1)

def q_rot6d_to_q_euler(q_rot6d):
    return torch.cat([q_rot6d[..., :3], rot6d_to_euler(q_rot6d[..., 3:9]), q_rot6d[..., 9:]], dim=-1)

def inv_se3(T):

    T_inv = torch.zeros_like(T)
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    
    R_T = R.transpose(-2, -1)
    t_inv = -(R_T @ t.unsqueeze(-1)).squeeze(-1)

    T_inv[:, :3, :3] = R_T
    T_inv[:, :3, 3] = t_inv
    T_inv[:, 3, 3] = 1.0

    return T_inv

def batch_inv_se3(T):

    B, L, = T.shape[:2]
    T = T.view(B * L, 4, 4)
    T_inv = inv_se3(T)
    return T_inv.view(B, L, 4, 4)


def matrix_to_vector(T: torch.Tensor) -> torch.Tensor:
    assert T.shape[-2:] == (4, 4), f"Expected [...,4,4], got {T.shape}"
    device, dtype = T.device, T.dtype
    leading = T.shape[:-2]
    R = T[..., :3, :3]              # [..., 3, 3]
    t = T[..., :3, 3]               # [..., 3]

    rvec = matrix_to_axis_angle(R.reshape(-1, 3, 3))  # [*, 3]
    rvec = rvec.view(*leading, 3)                     # [..., 3]
    pose = torch.cat([t, rvec], dim=-1)
    return pose

def vector_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    assert pose.shape[-1] == 6, f"Expected [...,6], got {pose.shape}"
    device, dtype = pose.device, pose.dtype
    leading = pose.shape[:-1]

    t = pose[..., :3].reshape(-1, 3)     # [*, 3]
    rvec = pose[..., 3:].reshape(-1, 3)  # [*, 3]

    R = axis_angle_to_matrix(rvec)                     # [*, 3, 3]
    R = R.view(*leading, 3, 3)                         # [..., 3, 3]
    T = torch.eye(4, device=device, dtype=dtype).expand(*leading, 4, 4).clone()  # [..., 4, 4]
    T[..., :3, :3] = R
    T[..., :3, 3]  = t.view(*leading, 3)
    return T


def compute_relative_se3(T_1, T_2):
    """
        Input: T_1: [L, 4, 4], T_2: [P, 4, 4]
        Return: T_rel: [L, P, 4, 4]
    """
    T_1_inv = inv_se3(T_1)               # [L, 4, 4]
    T_1_inv = T_1_inv[:, None, :, :]     # [L, 1, 4, 4]
    T_2 = T_2[None, :, :, :]             # [1, P, 4, 4]
    T_rel = torch.matmul(T_1_inv, T_2)   # [L, P, 4, 4]
    return T_rel

def compute_batch_relative_se3(T_1, T_2):
    """
        Input: T_1: [B, L, 4, 4], T_2: [B, P, 4, 4]
        Return: T_rel: [B, L, P, 4, 4]
    """
    T_1_inv = batch_inv_se3(T_1)         # [B, L, 4, 4]
    T_1_inv = T_1_inv[:, :, None, :, :]  # [B, L, 1, 4, 4]
    T_2 = T_2[:, None, :, :, :]          # [B, 1, P, 4, 4]
    T_rel = torch.matmul(T_1_inv, T_2)
    return T_rel

def se3_to_lie(se3: torch.Tensor) -> torch.Tensor:
    trans = se3[..., :3, 3]
    rot = se3[..., :3, :3]
    log_rot = SO3(tensor=rot.reshape(-1, 3, 3)).log_map()
    log_rot = log_rot.view(*rot.shape[:-2], 3)
    return torch.cat([trans, log_rot], dim=-1)

def lie_to_se3(lie: torch.Tensor) -> torch.Tensor:
    origin_shape = lie.shape[:-1]
    trans = lie[..., :3].reshape(-1, 3)
    log_rot = lie[..., 3:].reshape(-1, 3)
    rot = SO3().exp_map(log_rot).to_matrix()
    
    N = trans.shape[0]
    se3 = torch.eye(4, device=lie.device, dtype=lie.dtype)
    se3 = se3.unsqueeze(0).repeat(N, 1, 1)
    se3[:, :3, :3] = rot
    se3[:, :3, 3] = trans

    return se3.reshape(*origin_shape, 4, 4)


def diffuse_normalize(x, x_mean, x_std, eps=1e-8):
    return (x - x_mean[None, None, :]) / (x_std[None, None, :] + eps)

def rotation_matrix_geodesic_loss(R_pred, R_gt, eps=1e-8):
    R_rel = torch.matmul(R_pred, R_gt.transpose(-1, -2))  # [B, 3, 3]
    tr = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    c = (tr - 1.0) * 0.5
    skew = R_rel - R_rel.transpose(-1, -2)
    s = 0.5 * torch.linalg.norm(skew, dim=(-2, -1))
    theta = torch.atan2(s, c.clamp_min(-1 + eps))
    return theta.mean()

if __name__ == '__main__':
    """ Test correctness of above functions, no need to compare euler angle due to singularity. """
    test_euler = torch.rand(3) * 2 * torch.pi

    test_matrix = euler_to_matrix(test_euler)
    test_euler_prime = matrix_to_euler(test_matrix)
    test_matrix_prime = euler_to_matrix(test_euler_prime)
    assert torch.allclose(test_matrix, test_matrix_prime), \
        f"Original Matrix: {test_matrix}, Converted Matrix: {test_matrix_prime}"

    test_rot6d = matrix_to_rot6d(test_matrix)
    test_matrix_prime = rot6d_to_matrix(test_rot6d)
    assert torch.allclose(test_matrix, test_matrix_prime),\
        f"Original Matrix: {test_matrix}, Converted Matrix: {test_matrix_prime}"

    test_rot6d_prime = matrix_to_rot6d(test_matrix_prime)
    assert torch.allclose(test_rot6d, test_rot6d_prime), \
        f"Original Rot6D: {test_rot6d}, Converted Rot6D: {test_rot6d_prime}"

    test_euler_prime = rot6d_to_euler(test_rot6d)
    test_rot6d_prime = euler_to_rot6d(test_euler_prime)
    assert torch.allclose(test_rot6d, test_rot6d_prime), \
        f"Original Rot6D: {test_rot6d}, Converted Rot6D: {test_rot6d_prime}"

    print("All Tests Passed！")
