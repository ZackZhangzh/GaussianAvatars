# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz


class MeshGaussianModel(GaussianModel):
    """
    Gaussian Model bound to a custom mesh (non-FLAME).
    
    Key differences from FlameGaussianModel:
    - No parametric deformation (shape/expression)
    - Single timestep only (timestep=0)
    - Rigid transformation only (rotation + translation)
    - Canonical space = input mesh centered at origin
    
    Usage:
        gaussians = MeshGaussianModel('path/to/mesh.obj', sh_degree=3)
        gaussians.select_mesh_by_timestep(0)
    """
    
    def __init__(self, mesh_path: str, sh_degree: int, 
                 not_optimize_mesh_transform: bool = False):
        """
        Initialize MeshGaussianModel with custom mesh.
        
        Args:
            mesh_path: Path to .obj file
            sh_degree: Spherical harmonics degree
            not_optimize_mesh_transform: If True, freeze rotation/translation during training
        """
        super().__init__(sh_degree)
        
        self.mesh_path = Path(mesh_path)
        self.not_optimize_mesh_transform = not_optimize_mesh_transform
        self.mesh_param = None
        
        # Load mesh and initialize canonical space
        self._load_custom_mesh()
        
        # Initialize binding (1 point per face, same as FLAME)
        if self.binding is None:
            num_faces = len(self.faces_static)
            self.binding = torch.arange(num_faces).cuda()
            self.binding_counter = torch.ones(num_faces, dtype=torch.int32).cuda()
            print(f"[MeshGaussianModel] Initialized binding: "
                  f"{num_faces} faces, 1 point/face")
    
    def _load_custom_mesh(self):
        """
        Load .obj file and initialize canonical space.
        
        Coordinate system:
            Input mesh (arbitrary units) 
                -> Convert to meters
                -> Center to origin
                -> Canonical space
        """
        from utils.lbs import load_mesh_from_file
        
        # Load mesh vertices and faces
        verts_np, faces_np = load_mesh_from_file(self.mesh_path)
        
        # Unit conversion: assume input is mm, convert to meters (FLAME units)
        # Auto-detect: if max coordinate > 10, likely in mm
        if np.abs(verts_np).max() > 10.0:
            print(f"[MeshGaussianModel] Auto-detected mm units, converting to meters")
            verts_np = verts_np / 1000.0
        
        # Center to origin (canonical space, same as FLAME)
        verts_centered = verts_np - verts_np.mean(axis=0)
        
        # Store as canonical reference
        self.verts_canonical = torch.from_numpy(verts_centered).float().cuda()[None, ...]
        self.faces_static = torch.from_numpy(faces_np).long().cuda()
        
        # Initialize rigid transform parameters (identity transform)
        self.mesh_param = {
            'rotation': torch.zeros([1, 3]).float().cuda(),      # Euler XYZ (radians)
            'translation': torch.zeros([1, 3]).float().cuda(),   # Translation (meters)
        }
        
        self.num_timesteps = 1  # Single timestep only
        
        # Initialize mesh properties at identity transform
        self.update_mesh_properties(self.verts_canonical, self.verts_canonical)
        
        print(f"[MeshGaussianModel] Loaded mesh: {verts_np.shape[0]} verts, "
              f"{faces_np.shape[0]} faces")
        print(f"[MeshGaussianModel] Canonical bbox: "
              f"[{verts_centered.min():.3f}, {verts_centered.max():.3f}] meters")
    
    def select_mesh_by_timestep(self, timestep, original=False):
        """
        Update mesh properties for current timestep.
        Required by training loop (train.py L119) and remote viewer (train.py L84).
        
        For MeshGaussianModel: always use timestep=0 (single timestep mode).
        
        Args:
            timestep: Timestep index (must be 0)
            original: Unused for MeshGaussianModel (for FLAME compatibility)
                     FlameGaussianModel uses this to toggle original vs optimized mesh.
                     MeshGaussianModel has no such distinction, so this is ignored.
        """
        if timestep != 0:
            print(f"[MeshGaussianModel] Warning: requested timestep={timestep}, "
                  f"but only timestep=0 is supported. Using timestep=0.")
            timestep = 0
        
        # Note: 'original' parameter ignored for custom mesh
        # (no original/optimized mesh distinction like FLAME)
        
        # Apply rigid transformation: verts = R @ verts_cano + t
        R = self._euler_to_rotation_matrix(self.mesh_param['rotation'][0])
        t = self.mesh_param['translation'][0]
        
        verts_transformed = torch.matmul(
            self.verts_canonical[0], R.T  # [N, 3] @ [3, 3]
        ) + t  # [N, 3]
        verts_transformed = verts_transformed.unsqueeze(0)  # [1, N, 3]
        
        # Update face properties (centers, orientations, scales)
        self.update_mesh_properties(verts_transformed, self.verts_canonical)
    
    def update_mesh_properties(self, verts, verts_cano):
        """
        Compute face-level properties for Gaussian binding.
        
        This method is 100% copied from FlameGaussianModel.update_mesh_properties (L137-154).
        It computes:
            - face_center: Center of each triangle
            - face_orien_mat/quat: Orientation of each face
            - face_scaling: Scale factor for each face
        
        Args:
            verts: Transformed vertices [1, N, 3]
            verts_cano: Canonical vertices [1, N, 3]
        """
        faces = self.faces_static
        triangles = verts[:, faces]  # [1, F, 3, 3]
        
        # Face centers
        self.face_center = triangles.mean(dim=-2).squeeze(0)  # [F, 3]
        
        # Face orientation and scaling
        self.face_orien_mat, self.face_scaling = compute_face_orientation(
            verts.squeeze(0), faces.squeeze(0), return_scale=True
        )
        self.face_orien_quat = quat_xyzw_to_wxyz(
            rotmat_to_unitquat(self.face_orien_mat)
        )
        
        # For mesh rendering (if needed)
        self.verts = verts
        self.faces = faces
        
        # Canonical reference (for regularization)
        self.verts_cano = verts_cano
    
    def _euler_to_rotation_matrix(self, euler_angles):
        """
        Convert Euler angles (XYZ convention) to rotation matrix.
        
        IMPORTANT: Uses torch.stack() instead of torch.tensor() to preserve gradients.
        
        Args:
            euler_angles: [3] tensor (rx, ry, rz in radians)
        
        Returns:
            R: [3, 3] rotation matrix (differentiable w.r.t. euler_angles)
        """
        rx, ry, rz = euler_angles[0], euler_angles[1], euler_angles[2]
        
        # Use zeros/ones with same device/dtype (preserves computation graph)
        zero = torch.zeros(1, device=euler_angles.device, dtype=euler_angles.dtype).squeeze()
        one = torch.ones(1, device=euler_angles.device, dtype=euler_angles.dtype).squeeze()
        
        cos_rx, sin_rx = torch.cos(rx), torch.sin(rx)
        cos_ry, sin_ry = torch.cos(ry), torch.sin(ry)
        cos_rz, sin_rz = torch.cos(rz), torch.sin(rz)
        
        # Build rotation matrices using stack (preserves gradients!)
        Rx = torch.stack([
            torch.stack([one, zero, zero]),
            torch.stack([zero, cos_rx, -sin_rx]),
            torch.stack([zero, sin_rx, cos_rx])
        ])
        
        Ry = torch.stack([
            torch.stack([cos_ry, zero, sin_ry]),
            torch.stack([zero, one, zero]),
            torch.stack([-sin_ry, zero, cos_ry])
        ])
        
        Rz = torch.stack([
            torch.stack([cos_rz, -sin_rz, zero]),
            torch.stack([sin_rz, cos_rz, zero]),
            torch.stack([zero, zero, one])
        ])
        
        # Combined rotation: R = Rz @ Ry @ Rx (XYZ convention)
        R = Rz @ Ry @ Rx
        return R
    
    def training_setup(self, training_args):
        """
        Setup optimizer for training.
        
        By default, mesh transform (rotation/translation) is optimized.
        Can be disabled via --not_optimize_mesh_transform flag.
        
        Uses separate learning rates for rotation and translation:
            - mesh_pose_lr (default 1e-5): for rotation
            - mesh_trans_lr (default 1e-6): for translation
        
        Args:
            training_args: Optimization parameters
        """
        super().training_setup(training_args)
        
        # Check if mesh transform optimization is disabled
        if self.not_optimize_mesh_transform:
            print("[MeshGaussianModel] Mesh transform frozen (rotation/translation fixed)")
            return
        
        # Enable mesh optimization with split learning rates (mirroring FLAME)
        self.mesh_param['rotation'].requires_grad = True
        self.mesh_param['translation'].requires_grad = True
        
        # Get learning rates from training_args (parallel to flame_pose_lr/flame_trans_lr)
        mesh_pose_lr = getattr(training_args, 'mesh_pose_lr', 1e-5)
        mesh_trans_lr = getattr(training_args, 'mesh_trans_lr', 1e-6)
        
        # Add rotation parameter group (similar to FLAME pose params)
        param_rotation = {
            'params': [self.mesh_param['rotation']],
            'lr': mesh_pose_lr,
            'name': 'mesh_rotation'
        }
        self.optimizer.add_param_group(param_rotation)
        
        # Add translation parameter group (similar to FLAME trans params)
        param_translation = {
            'params': [self.mesh_param['translation']],
            'lr': mesh_trans_lr,
            'name': 'mesh_translation'
        }
        self.optimizer.add_param_group(param_translation)
        
        print(f"[MeshGaussianModel] Mesh optimization enabled: "
              f"rotation_lr={mesh_pose_lr}, translation_lr={mesh_trans_lr}")
    
    def save_ply(self, path):
        """
        Save Gaussian points and mesh parameters.
        
        Args:
            path: Path to save point_cloud.ply
        """
        super().save_ply(path)
        
        # Save mesh transform parameters
        npz_path = Path(path).parent / "mesh_param.npz"
        mesh_param = {k: v.cpu().numpy() for k, v in self.mesh_param.items()}
        np.savez(str(npz_path), **mesh_param)
        print(f"[MeshGaussianModel] Saved mesh params to {npz_path}")
    
    def load_ply(self, path, **kwargs):
        """
        Load Gaussian points and mesh parameters.
        
        Args:
            path: Path to point_cloud.ply
            **kwargs: Additional arguments (has_target, motion_path, etc.)
        """
        super().load_ply(path)
        
        # Load mesh transform parameters if available
        npz_path = Path(path).parent / "mesh_param.npz"
        if npz_path.exists():
            mesh_param = np.load(str(npz_path))
            self.mesh_param = {
                k: torch.from_numpy(v).cuda() 
                for k, v in mesh_param.items()
            }
            print(f"[MeshGaussianModel] Loaded mesh params from {npz_path}")
        else:
            print(f"[MeshGaussianModel] Warning: mesh_param.npz not found, "
                  f"using default identity transform")
