# Code heavily inspired by https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py.
# Please consider citing their work if you find this code useful. The code is subject to the license available via
# https://github.com/vchoutas/smplx/edit/master/LICENSE

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
try:
    from pytorch3d.io import load_obj
except ImportError:
    from utils.pytorch3d_load_obj import load_obj

FLAME_MESH_PATH = "flame_model/assets/flame/head_template_mesh.obj"
# FLAME_MESH_PATH = "/home/zhihao/NeRSemble/data/MRI/luo_fit_result_rot.obj"
# FLAME_MESH_PATH = "/home/zhihao/NeRSemble/data/MRI/MRI_luotao_skin_rot.obj"


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + \
        (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


class MRIHead(nn.Module):
    """
    Simplified version that only loads static mesh without FLAME deformation
    """

    def __init__(
        self,
        shape_params=None,  # Keep for compatibility but ignore
        expr_params=None,   # Keep for compatibility but ignore
        flame_model_path=None,  # Keep for compatibility but ignore
        flame_lmk_embedding_path=None,  # Keep for compatibility but ignore
        flame_template_mesh_path=None,
        include_mask=False,
        add_teeth=False,  # Keep for compatibility but ignore
        mesh_path=FLAME_MESH_PATH,  # Default mesh path
        **kwargs,  # Ignore any other parameters

    ):
        super().__init__()

        self.dtype = torch.float32

        # Use default path if none provided
        if flame_template_mesh_path is None:
            flame_template_mesh_path = mesh_path

        # Load mesh data
        verts, faces, aux = load_obj(
            flame_template_mesh_path, load_textures=False)

        # Register mesh components
        self.register_buffer("v_template", verts.to(self.dtype))
        self.register_buffer("faces", faces.verts_idx, persistent=False)

        # UV coordinates if available
        if hasattr(aux, 'verts_uvs') and aux.verts_uvs is not None:
            vertex_uvs = aux.verts_uvs
            face_uvs_idx = faces.textures_idx

            # Create UV coordinates per face
            pad = torch.ones(vertex_uvs.shape[0], 1)
            vertex_uvs = torch.cat([vertex_uvs, pad], dim=-1)
            vertex_uvs = vertex_uvs * 2 - 1
            vertex_uvs[..., 1] = -vertex_uvs[..., 1]

            face_uv_coords = face_vertices(
                vertex_uvs[None], face_uvs_idx[None])[0]
            self.register_buffer(
                "face_uvcoords", face_uv_coords, persistent=False)
            self.register_buffer("verts_uvs", aux.verts_uvs, persistent=False)
            self.register_buffer(
                "textures_idx", faces.textures_idx, persistent=False)
        else:
            # Create dummy UV coordinates if not available
            self.register_buffer("verts_uvs", torch.zeros(
                verts.shape[0], 2), persistent=False)
            self.register_buffer(
                "textures_idx", self.faces.clone(), persistent=False)
            self.register_buffer("face_uvcoords", torch.zeros(
                self.faces.shape[0], 3, 3), persistent=False)

        # Optional mask functionality (simplified)
        if include_mask:
            self.mask = SimpleMask(
                faces=self.faces,
                faces_t=self.textures_idx,
                num_verts=self.v_template.shape[0],
                num_faces=self.faces.shape[0],
                vertices=self.v_template,  # Pass vertices for region computation
            )

    def forward(
        self,
        shape=None,  # Keep for compatibility but ignore
        expr=None,   # Keep for compatibility but ignore
        rotation=None,  # Keep for compatibility but ignore
        neck=None,   # Keep for compatibility but ignore
        jaw=None,    # Keep for compatibility but ignore
        eyes=None,   # Keep for compatibility but ignore
        translation=None,
        zero_centered_at_root_node=False,  # Keep for compatibility but ignore
        return_landmarks=False,  # Keep for compatibility but ignore
        return_verts_cano=False,  # Keep for compatibility but ignore
        static_offset=None,
        dynamic_offset=None,  # Keep for compatibility but ignore
        **kwargs  # Ignore any other parameters
    ):
        """
        Simplified forward pass that only applies translation and static offset
        """
        batch_size = 1
        if translation is not None:
            batch_size = translation.shape[0]
        elif static_offset is not None:
            batch_size = static_offset.shape[0]

        # Get template vertices
        vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Add static offset if provided
        if static_offset is not None:
            vertices = vertices + static_offset

        # SCALE_FACTOR = 0.8306812884531367
        # vertices = vertices * SCALE_FACTOR

        if rotation is not None:
            from pytorch3d.transforms import axis_angle_to_matrix
            rot_mat = axis_angle_to_matrix(rotation)  # [B, 3, 3]
            vertices = torch.bmm(vertices, rot_mat.transpose(1, 2))
        # Add translation if provided
        if translation is not None:
            vertices = vertices + translation[:, None, :]

        # Handle return format for compatibility
        ret_vals = [vertices]

        if return_verts_cano:
            ret_vals.append(self.v_template.unsqueeze(
                0).expand(batch_size, -1, -1))

        if return_landmarks:
            # Return dummy landmarks for compatibility
            dummy_landmarks = torch.zeros(
                batch_size, 68, 3, device=vertices.device, dtype=vertices.dtype)
            ret_vals.append(dummy_landmarks)

        if len(ret_vals) > 1:
            return ret_vals
        else:
            return ret_vals[0]


class BufferContainer(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        main_str = super().__repr__() + '\n'
        for name, buf in self.named_buffers():
            main_str += f'    {name:20}\t{buf.shape}\t{buf.dtype}\n'
        return main_str

    def __iter__(self):
        for name, buf in self.named_buffers():
            yield name, buf

    def keys(self):
        return [name for name, buf in self.named_buffers()]

    def items(self):
        return [(name, buf) for name, buf in self.named_buffers()]


class SimpleMask(nn.Module):
    """Simplified mask class for basic region definition"""

    def __init__(self, faces=None, faces_t=None, num_verts=None, num_faces=None, vertices=None):
        super().__init__()
        self.faces = faces
        self.faces_t = faces_t
        self.num_verts = num_verts
        self.num_faces = num_faces

        # Create basic vertex regions based on mesh structure
        self.v = BufferContainer()

        # Define some basic regions if vertices are provided
        if vertices is not None and num_verts is not None:
            # Simple left/right split based on x-coordinate
            left_mask = torch.where(vertices[:, 0] < 0)[0]
            right_mask = torch.where(vertices[:, 0] >= 0)[0]

            self.v.register_buffer("left_half", left_mask)
            self.v.register_buffer("right_half", right_mask)

    def get_vid_by_region(self, regions, keep_order=False):
        """Get vertex indices by regions"""
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            vid = torch.cat([self.v.get_buffer(k) for k in regions])
            if keep_order:
                return vid
            else:
                return vid.unique()
        else:
            return torch.tensor([], dtype=torch.long)
