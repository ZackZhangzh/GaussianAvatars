import os
import sys
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import chumpy as ch
import argparse
import matplotlib.pyplot as plt

# -- Robust Path Setup --
# This block ensures that all necessary subdirectories from `flame-fitting`
# are added to the Python path, resolving the import errors robustly.
try:
    script_dir = Path(__file__).parent.resolve()
    flame_fitting_dir = script_dir / "flame-fitting"

    # Add the flame-fitting directory itself
    if flame_fitting_dir.exists() and str(flame_fitting_dir) not in sys.path:
        sys.path.insert(0, str(flame_fitting_dir))

    # Add all necessary subdirectories from flame-fitting to the path
    subdirs_to_add = ["fitting", "sbody", "smpl_webuser", "psbody"]
    for subdir in subdirs_to_add:
        full_subdir_path = flame_fitting_dir / subdir
        if full_subdir_path.exists() and str(full_subdir_path) not in sys.path:
            pass  # The parent `flame-fitting` is already added.

    # Add the parent of the script dir, which is the project root `NeRSemble`
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Re-add the script dir to ensure relative imports within GaussianAvatars work
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


except NameError:
    # Fallback for interactive environments
    sys.path.insert(0, "./GaussianAvatars/flame-fitting")
    sys.path.insert(0, "./GaussianAvatars")
    sys.path.insert(0, ".")

# Now, the imports should be resolved.
from fitting.util import (
    load_binary_pickle,
    write_simple_obj,
    safe_mkdir,
    get_unit_factor,
)
from fitting.landmarks import (
    load_embedding,
    mesh_points_by_barycentric_coordinates,
    landmark_error_3d,
)
from smpl_webuser.serialization import load_model
from psbody.mesh import Mesh
from sbody.mesh_distance import ScanToMesh
from sbody.robustifiers import GMOf
from sbody.alignment.objectives import sample_from_mesh

# Import visualization utilities
from fitting_visualize import (
    visualize_fitting_stages,
    visualize_overlay_comparison,
    visualize_mesh,
    save_obj_mesh,
    create_fitting_summary_report,
)

# Import fitting related modules before torch
# Try to import torch and related modules with error handling
try:
    import torch

    # Import the same FLAME model used in training
    from flame_model.flame import FlameHead
    from scene import Scene, FlameGaussianModel
    from arguments import ModelParams

    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch-related imports failed: {e}")
    print("Running in fitting-only mode without visualization")
    TORCH_AVAILABLE = False


def call_load_flame_subprocess(data_path, timestep=0):
    """
    Call load_flame.py from the correct conda environment as subprocess
    Returns the vertices, faces, and parameter data in memory
    """
    print(f"Calling load_flame.py subprocess for dataset: {data_path}")

    # Create temporary directory for output
    temp_dir = Path(tempfile.mkdtemp())
    temp_obj_path = temp_dir / f"flame_mesh_t{timestep}.obj"
    temp_params_path = temp_dir / f"flame_params_t{timestep}.npz"

    try:
        # Construct the command to run load_flame.py in gaussian-avatars environment
        cmd = [
            "conda",
            "run",
            "-n",
            "gaussian-avatars",
            "python",
            "load_flame.py",
            "-s",
            str(data_path),
            "-t",
            str(timestep),
            "--save-to-temp",
            str(temp_dir),
        ]

        print(f"Running command: {' '.join(cmd)}")

        # Run the subprocess with compatible parameters
        result = subprocess.run(
            cmd,
            cwd=".",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            print(
                f"load_flame.py subprocess failed with return code {result.returncode}"
            )
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None, None, None

        print(f"load_flame.py subprocess completed successfully")
        print(f"STDOUT: {result.stdout}")

        # Load the generated OBJ file
        if not temp_obj_path.exists():
            print(f"Expected OBJ file not found: {temp_obj_path}")
            return None, None, None

        vertices, faces = load_obj_mesh(str(temp_obj_path))
        print(
            f"Loaded mesh from subprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
        )

        # Simple vertex count check
        if vertices.shape[0] not in [5023, 5143]:
            print(f"Warning: Unexpected vertex count: {vertices.shape[0]}")

        # Load parameters if available
        flame_params = None
        if temp_params_path.exists():
            flame_params = dict(np.load(str(temp_params_path)))
            print(f"Loaded FLAME parameters from subprocess")

        return vertices, faces, flame_params

    except subprocess.TimeoutExpired:
        print("load_flame.py subprocess timed out")
        return None, None, None
    except Exception as e:
        print(f"Error calling load_flame.py subprocess: {e}")
        return None, None, None
    finally:
        # Clean up temporary files with better error handling
        try:
            if temp_obj_path.exists():
                temp_obj_path.unlink()
            if temp_params_path.exists():
                temp_params_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
        except Exception as cleanup_error:
            print(f"Warning: Failed to clean up temporary files: {cleanup_error}")


def load_obj_mesh(filepath):
    """Load vertices and faces from OBJ file"""
    vertices = []
    faces = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                # Vertex line
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
            elif line.startswith("f "):
                # Face line (convert from 1-based to 0-based indexing)
                parts = line.split()
                face = []
                for part in parts[1:]:
                    # Handle face formats like "v1/vt1/vn1" or just "v1"
                    vertex_idx = int(part.split("/")[0]) - 1
                    face.append(vertex_idx)
                faces.append(face)

    return np.array(vertices), np.array(faces)


def load_flame_with_params_from_dataset(data_path, timestep=0):
    """
    Load FLAME model with parameters from dataset, try direct approach first then fallback to subprocess
    """
    print(f"Loading FLAME model with parameters from dataset: {data_path}")

    # Try direct PyTorch approach first (if available)
    if TORCH_AVAILABLE:
        try:
            # Create model params similar to train.py
            class Args:
                def __init__(self, source_path):
                    self.source_path = source_path
                    self.model_path = ""
                    self.sh_degree = 3
                    self.resolution = -1
                    self.white_background = False
                    self.data_device = "cuda"
                    self.eval = True
                    self.bind_to_mesh = True
                    self.use_mri_model = False
                    self.mesh_path = ""
                    self.points_per_face = 1
                    self.disable_flame_static_offset = False
                    self.not_finetune_flame_params = True

            args = Args(data_path)

            # Initialize FlameGaussianModel and Scene
            gaussians = FlameGaussianModel(
                args.sh_degree,
                args.disable_flame_static_offset,
                args.not_finetune_flame_params,
            )

            # Override flame_model to use no teeth version for fitting compatibility
            gaussians.flame_model = FlameHead(
                gaussians.n_shape,
                gaussians.n_expr,
                add_teeth=False,
            ).cuda()

            scene = Scene(args, gaussians)

            if gaussians.flame_param is None:
                raise Exception("No FLAME parameters found in dataset")

            # Check and handle parameter compatibility
            print("Checking FLAME parameter compatibility...")
            flame_vertices_count = gaussians.flame_model.v_template.shape[0]
            print(f"FLAME model vertex count: {flame_vertices_count}")

            # Handle static_offset dimension mismatch
            if "static_offset" in gaussians.flame_param:
                static_offset = gaussians.flame_param["static_offset"]
                static_offset_vertices = static_offset.shape[0]

                if static_offset_vertices != flame_vertices_count:
                    print(
                        f"Adjusting static_offset from {static_offset_vertices} to {flame_vertices_count} vertices"
                    )
                    if static_offset_vertices > flame_vertices_count:
                        gaussians.flame_param["static_offset"] = static_offset[
                            :flame_vertices_count
                        ]
                    else:
                        padding = torch.zeros(
                            (flame_vertices_count - static_offset_vertices, 3),
                            device=static_offset.device,
                            dtype=static_offset.dtype,
                        )
                        gaussians.flame_param["static_offset"] = torch.cat(
                            [static_offset, padding], dim=0
                        )

            # Handle dynamic_offset dimension mismatch
            if "dynamic_offset" in gaussians.flame_param:
                dynamic_offset = gaussians.flame_param["dynamic_offset"]
                if len(dynamic_offset.shape) == 3:  # (T, V, 3)
                    dynamic_offset_vertices = dynamic_offset.shape[1]

                    if dynamic_offset_vertices != flame_vertices_count:
                        print(
                            f"Adjusting dynamic_offset from {dynamic_offset_vertices} to {flame_vertices_count} vertices"
                        )
                        if dynamic_offset_vertices > flame_vertices_count:
                            gaussians.flame_param["dynamic_offset"] = dynamic_offset[
                                :, :flame_vertices_count
                            ]
                        else:
                            T = dynamic_offset.shape[0]
                            padding = torch.zeros(
                                (T, flame_vertices_count - dynamic_offset_vertices, 3),
                                device=dynamic_offset.device,
                                dtype=dynamic_offset.dtype,
                            )
                            gaussians.flame_param["dynamic_offset"] = torch.cat(
                                [dynamic_offset, padding], dim=1
                            )

            print("Parameter compatibility check completed")

            # Ensure we have the right timestep
            num_timesteps = gaussians.flame_param["expr"].shape[0]
            if timestep >= num_timesteps:
                timestep = 0

            # Use the same method as in training to generate mesh
            gaussians.select_mesh_by_timestep(timestep)

            # Get vertices and faces
            vertices = gaussians.verts.squeeze(0).cpu().numpy()
            faces = gaussians.faces.cpu().numpy()

            # Simple vertex count check
            if vertices.shape[0] not in [5023, 5143]:
                print(f"Warning: Unexpected vertex count: {vertices.shape[0]}")

            return vertices, faces, gaussians.flame_param

        except Exception as e:
            print(f"Direct PyTorch approach failed: {e}")
            print("Falling back to subprocess approach...")

    # Fallback to subprocess approach
    return call_load_flame_subprocess(data_path, timestep)


def apply_flame_params_to_chumpy_model(
    model, flame_param, timestep=0, stage1_vertices=None
):
    """Apply FLAME parameters from dataset to chumpy model and optionally replace vertices"""
    if flame_param is None:
        print("No FLAME parameters to apply, using neutral model")
        model.betas[:] = 0.0
        model.pose[:] = 0.0
        model.trans[:] = 0.0
        return

    # Handle both PyTorch tensor format and numpy array format
    def extract_param(param_dict, key, timestep=0):
        if key not in param_dict:
            return None

        param = param_dict[key]

        # Convert PyTorch tensor to numpy if needed
        if hasattr(param, "cpu"):
            param = param.cpu().numpy()

        # Handle time-dependent parameters
        if len(param.shape) > 1 and param.shape[0] > timestep:
            return param[timestep]
        elif len(param.shape) > 1:
            return param[0]
        else:
            return param

    # Apply shape parameters
    shape_params = extract_param(flame_param, "shape", timestep)
    if shape_params is not None:
        model.betas[: len(shape_params)] = shape_params

    # Apply expression parameters
    expr_params = extract_param(flame_param, "expr", timestep)
    if expr_params is not None:
        model.betas[300 : 300 + len(expr_params)] = expr_params

    # Apply pose parameters
    rotation = extract_param(flame_param, "rotation", timestep)
    if rotation is not None:
        model.pose[:3] = rotation

    neck_pose = extract_param(flame_param, "neck_pose", timestep)
    if neck_pose is not None:
        model.pose[3:6] = neck_pose

    jaw_pose = extract_param(flame_param, "jaw_pose", timestep)
    if jaw_pose is not None:
        model.pose[6:9] = jaw_pose

    # Apply translation
    translation = extract_param(flame_param, "translation", timestep)
    if translation is not None:
        model.trans[:] = translation

    print("FLAME parameters applied to chumpy model")

    # CRITICAL FIX: Apply Stage 1 vertices directly to ensure correct initial state
    if stage1_vertices is not None:
        print("Applying Stage 1 vertex corrections to chumpy model...")

        # Get current chumpy vertices after parameter application
        current_vertices = model.r.copy()

        # Calculate the difference
        vertex_diff = stage1_vertices - current_vertices
        mean_diff = np.linalg.norm(vertex_diff, axis=1).mean()
        print(f"Vertex difference before correction: {mean_diff:.6f}")

        # Apply vertex replacement: modify the template vertices to match Stage 1
        model.v_template[:] = stage1_vertices.copy()

        # Reset pose and translation to ensure vertices are exactly as expected
        model.pose[:] = 0.0
        model.trans[:] = 0.0

        # Verify the correction
        corrected_vertices = model.r.copy()
        final_diff = np.linalg.norm(corrected_vertices - stage1_vertices, axis=1).mean()
        print(f"Vertex difference after correction: {final_diff:.6f}")

        if final_diff < 1e-6:
            print("✓ Stage 1 vertices successfully applied to chumpy model")
        else:
            print(f"⚠ Warning: Residual difference remains: {final_diff:.6f}")


def load_and_visualize_flame(data_path, timestep=0, save_mesh=True):
    """Load FLAME parameters from dataset and visualize the result"""
    print(f"Loading dataset from {data_path}")

    # Create visualize directory
    visualize_dir = Path(data_path) / "visualize"
    visualize_dir.mkdir(exist_ok=True)

    # Load FLAME with parameters
    vertices, faces, flame_param = load_flame_with_params_from_dataset(
        data_path, timestep
    )

    if vertices is not None:
        # Visualize
        visualize_mesh(vertices, faces, visualize_dir, timestep)

        # Save mesh as OBJ file if requested
        if save_mesh:
            obj_path = visualize_dir / f"flame_mesh_t{timestep}.obj"
            save_obj_mesh(str(obj_path), vertices, faces)
            print(f"Mesh saved to {obj_path}")

        return vertices, faces, flame_param
    else:
        print("Failed to load FLAME model with parameters")
        return None, None, None


def procrustes_alignment(source_points, target_points, with_scaling=True):
    """
    Rigid alignment using Procrustes analysis (translation + rotation + scaling)

    Args:
        source_points: Nx3 array of source landmarks
        target_points: Nx3 array of target landmarks
        with_scaling: Whether to include scaling

    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        scale: scaling factor
        transformed_source: aligned source points
    """
    # Center the points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute rotation using SVD
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    if with_scaling:
        var_source = np.sum(np.square(source_centered))
        scale = np.sum(S) / var_source
    else:
        scale = 1.0

    # Compute translation
    t = target_centroid - scale * R @ source_centroid

    # Apply transformation
    transformed_source = (scale * (R @ source_points.T)).T + t

    return R, t, scale, transformed_source


def rigid_align_scan_to_flame(
    scan, lmk_3d, model, lmk_face_idx, lmk_b_coords, output_dir
):
    """Rigidly align scan landmarks to FLAME landmarks"""
    print("Step 1: Rigid alignment of scan to FLAME...")

    # Get FLAME landmarks from current model state
    flame_lmks = mesh_points_by_barycentric_coordinates(
        model, model.f, lmk_face_idx, lmk_b_coords
    )

    # Convert chumpy to numpy if needed
    if hasattr(flame_lmks, "r"):
        flame_lmks_np = flame_lmks.r
    else:
        flame_lmks_np = flame_lmks

    # Compute transformation for scan to align with FLAME
    R, t, scale, transformed_lmks = procrustes_alignment(
        lmk_3d, flame_lmks_np, with_scaling=True
    )

    # Save the transformation
    transform_path = output_dir / "stage2_alignment_transform.npz"
    np.savez(str(transform_path), R=R, t=t, scale=scale)
    print(f"Alignment transformation saved to: {transform_path}")

    # Apply transformation to scan
    scan_centered = scan.v - np.mean(lmk_3d, axis=0)
    aligned_scan_v = (scale * (R @ scan_centered.T)).T + np.mean(flame_lmks_np, axis=0)

    # Create aligned scan
    aligned_scan = Mesh(v=aligned_scan_v, f=scan.f.copy())
    aligned_lmk_3d = transformed_lmks

    # Save aligned scan
    aligned_scan_path = output_dir / "stage2_scan_aligned.obj"
    write_simple_obj(
        mesh_v=aligned_scan.v,
        mesh_f=aligned_scan.f,
        filepath=str(aligned_scan_path),
        verbose=False,
    )
    print(f"Aligned scan saved to: {aligned_scan_path}")

    # Verify alignment
    final_flame_lmks = mesh_points_by_barycentric_coordinates(
        model, model.f, lmk_face_idx, lmk_b_coords
    )
    if hasattr(final_flame_lmks, "r"):
        final_flame_lmks_np = final_flame_lmks.r
    else:
        final_flame_lmks_np = final_flame_lmks

    lmk_distances = np.linalg.norm(aligned_lmk_3d - final_flame_lmks_np, axis=1)
    print(f"Landmark alignment error - Mean: {np.mean(lmk_distances):.6f}")

    return aligned_scan, aligned_lmk_3d


def stage3_non_rigid_fitting(
    model_to_fit,
    target_scan,
    target_lmks,
    lmk_face_idx,
    lmk_b_coords,
    output_dir,
    shape_num=100,
    expr_num=50,
):
    """
    Stage 3: Non-rigidly fit the FLAME model to the aligned scan.
    """

    # Define weights for optimization objectives
    weights = {
        "s2m": 2.0,
        "lmk": 1e-2,
        "shape": 1e-4,
        "expr": 1e-4,
        "pose": 1e-3,
    }
    gmo_sigma = 1e-4

    # Define variables to be optimized
    shape_idx = np.arange(0, min(300, shape_num))
    expr_idx = np.arange(300, 300 + min(100, expr_num))
    used_idx = np.union1d(shape_idx, expr_idx)

    free_variables = [
        model_to_fit.trans,
        model_to_fit.pose,
        model_to_fit.betas[used_idx],
    ]

    # Define optimization objectives
    lmk_err = landmark_error_3d(
        mesh_verts=model_to_fit,
        mesh_faces=model_to_fit.f,
        lmk_3d=target_lmks,
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
    )

    sampler = sample_from_mesh(target_scan, sample_type="vertices")
    s2m = ScanToMesh(
        target_scan,
        model_to_fit,
        model_to_fit.f,
        scan_sampler=sampler,
        rho=lambda x: GMOf(x, sigma=gmo_sigma),
    )

    # Regularizers
    shape_err = weights["shape"] * model_to_fit.betas[shape_idx]
    expr_err = weights["expr"] * model_to_fit.betas[expr_idx]
    pose_err = weights["pose"] * model_to_fit.pose[3:]

    objectives = {
        "s2m": weights["s2m"] * s2m,
        "lmk": weights["lmk"] * lmk_err,
        "shape": shape_err,
        "expr": expr_err,
        "pose": pose_err,
    }

    # Setup optimization
    import scipy.sparse as sp

    opt_options = {
        "disp": 1,
        "delta_0": 0.1,
        "e_3": 1e-4,
        "maxiter": 2000,
        "sparse_solver": lambda A, x: sp.linalg.cg(A, x, maxiter=2000)[0],
    }

    # Create directory for iteration results
    iter_dir = output_dir / "stage3_iterations"
    iter_dir.mkdir(exist_ok=True)

    # Save initial state
    initial_mesh_path = iter_dir / "stage3_iter_0000.obj"
    write_simple_obj(
        mesh_v=model_to_fit.r,
        mesh_f=model_to_fit.f,
        filepath=str(initial_mesh_path),
        verbose=False,
    )

    # Define callback for saving intermediate results
    iter_count = 0

    def on_step(x):
        nonlocal iter_count
        iter_count += 1
        mesh_path = iter_dir / f"stage3_iter_{iter_count:04d}.obj"
        write_simple_obj(
            mesh_v=model_to_fit.r,
            mesh_f=model_to_fit.f,
            filepath=str(mesh_path),
            verbose=False,
        )

    # Run optimization
    print("Starting non-rigid optimization...")
    ch.minimize(
        fun=objectives,
        x0=free_variables,
        method="dogleg",
        callback=on_step,
        options=opt_options,
    )
    print("Non-rigid optimization finished.")

    # Save results
    fitted_vertices = model_to_fit.r
    fitted_faces = model_to_fit.f
    fitted_params = {
        "trans": model_to_fit.trans.r,
        "pose": model_to_fit.pose.r,
        "betas": model_to_fit.betas.r,
    }

    # Save fitted mesh
    fitted_mesh_path = output_dir / "stage3_fitted_flame.obj"
    write_simple_obj(
        mesh_v=fitted_vertices,
        mesh_f=fitted_faces,
        filepath=str(fitted_mesh_path),
        verbose=False,
    )
    print(f"Fitted FLAME model saved to: {fitted_mesh_path}")

    # Save fitted parameters
    fitted_params_path = output_dir / "stage3_fitted_flame_params.npz"
    np.savez(str(fitted_params_path), **fitted_params)
    print(f"Fitted parameters saved to: {fitted_params_path}")

    return fitted_vertices, fitted_faces, fitted_params


def find_model_file(filename, search_paths):
    """Helper function to find model files in various locations"""
    if os.path.exists(filename):
        return filename

    for path in search_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path

    raise FileNotFoundError(
        f"Could not find {filename} in any of the search paths: {search_paths}"
    )


def load_scan_with_unit_conversion(scan_path, unit_scale=1.0, apply_scale=True):
    """
    Load scan mesh with optional unit conversion
    
    Args:
        scan_path: Path to scan OBJ file
        unit_scale: Scale factor to convert units (e.g., 0.001 for mm to m)
        apply_scale: Whether to apply scaling
    
    Returns:
        scan: Loaded mesh with optional scaling applied
    """
    print(f"Loading scan from: {scan_path}")
    scan = Mesh(filename=scan_path)
    
    if apply_scale and unit_scale != 1.0:
        print(f"Applying unit scale factor: {unit_scale}")
        scan.v = scan.v * unit_scale
        
    print(f"Loaded scan: {scan.v.shape[0]} vertices, {scan.f.shape[0]} faces")
    print(f"Scan bounds: min={scan.v.min(axis=0)}, max={scan.v.max(axis=0)}")
    
    return scan

def load_landmarks_with_unit_conversion(lmk_path, unit_scale=1.0, apply_scale=True):
    """
    Load landmarks with optional unit conversion
    
    Args:
        lmk_path: Path to landmark file
        unit_scale: Scale factor to convert units
        apply_scale: Whether to apply scaling
        
    Returns:
        lmk_3d: Loaded landmarks with optional scaling applied
    """
    if lmk_path.endswith(".pp"):
        from fitting.landmarks import load_picked_points
        lmk_3d = load_picked_points(lmk_path)
        print(f"Loaded scan landmark from (pp): {lmk_path}")
    else:
        lmk_3d = np.load(lmk_path)
        print(f"Loaded scan landmark from (npy): {lmk_path}")
    
    if apply_scale and unit_scale != 1.0:
        print(f"Applying unit scale factor to landmarks: {unit_scale}")
        lmk_3d = lmk_3d * unit_scale
        
    print(f"Loaded landmarks: {lmk_3d.shape}")
    print(f"Landmark bounds: min={lmk_3d.min(axis=0)}, max={lmk_3d.max(axis=0)}")
    
    return lmk_3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load and visualize FLAME parameters or fit to scan"
    )
    parser.add_argument(
        "-s",
        "--source_path",
        type=str,
        required=True,
        help="Path to directory containing flame_param.npz",
    )
    parser.add_argument(
        "-t",
        "--timestep",
        type=int,
        default=0,
        help="Timestep to visualize (default: 0)",
    )
    parser.add_argument(
        "--no-save-mesh", action="store_true", help="Do not save mesh as OBJ file"
    )
    parser.add_argument(
        "--scan_path", type=str, default=None, help="Path to scan OBJ file for fitting"
    )
    parser.add_argument(
        "--lmk_path",
        type=str,
        default=None,
        help="Path to landmark NPY file for fitting",
    )
    parser.add_argument(
        "--unit_scale",
        type=float,
        default=1.0,
        help="Unit scale factor (e.g., 0.001 for mm to m conversion)",
    )
    parser.add_argument(
        "--no_unit_conversion",
        action="store_true",
        help="Disable automatic unit conversion",
    )

    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        print(f"Error: Source path {args.source_path} does not exist")
        sys.exit(1)

    # --- Fitting Mode ---
    if args.scan_path and args.lmk_path:
        if not os.path.exists(args.scan_path):
            print(f"Error: Scan path {args.scan_path} does not exist")
            sys.exit(1)
        if not os.path.exists(args.lmk_path):
            print(f"Error: Landmark path {args.lmk_path} does not exist")
            sys.exit(1)

        print("=" * 60)
        print("MRI FITTING PIPELINE")
        print("=" * 60)

        # Create output directory
        output_dir = Path(args.source_path) / "fitting_results"
        output_dir.mkdir(exist_ok=True)

        # Load the chumpy FLAME model
        print("Loading generic FLAME model...")
        model_path = find_model_file(
            "generic_model.pkl",
            ["flame-fitting/models", "./flame-fitting/models", "./models", "models"],
        )
        model = load_model(model_path)
        print(f"FLAME model loaded from: {model_path}")

        # Load scan and landmarks with unit conversion
        scan = load_scan_with_unit_conversion(
            args.scan_path, 
            unit_scale=args.unit_scale,
            apply_scale=not args.no_unit_conversion
        )
        
        lmk_3d = load_landmarks_with_unit_conversion(
            args.lmk_path,
            unit_scale=args.unit_scale, 
            apply_scale=not args.no_unit_conversion
        )

        print(f"Loaded scan: {scan.v.shape[0]} vertices, {scan.f.shape[0]} faces")
        print(f"Loaded landmarks: {lmk_3d.shape}")

        # Load landmark embedding
        print("Loading landmark embedding...")
        lmk_emb_path = find_model_file(
            "flame_static_embedding.pkl",
            ["flame-fitting/models", "./flame-fitting/models", "./models", "models"],
        )
        lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
        print("Loaded landmark embedding.")

        # Stage 1: Load FLAME model with parameters
        print("\n" + "=" * 50)
        print("STAGE 1: LOADING FLAME PARAMETERS")
        print("=" * 50)

        initial_vertices, initial_faces, stage1_flame_param = (
            load_flame_with_params_from_dataset(args.source_path, args.timestep)
        )

        if initial_vertices is None:
            print("Failed to load FLAME model with parameters. Exiting.")
            sys.exit(1)

        print("Stage1 FLAME parameters loaded")
        print(f"Stage1 FLAME mesh vertex count: {initial_vertices.shape[0]}")

        stage1_path = output_dir / "stage1_flame_with_params.obj"
        write_simple_obj(
            mesh_v=initial_vertices,
            mesh_f=initial_faces,
            filepath=str(stage1_path),
            verbose=False,
        )
        print(f"Stage 1 - FLAME model with parameters saved to: {stage1_path}")

        # Configure the chumpy model with Stage 1 parameters
        print("Configuring chumpy FLAME model with Stage 1 parameters...")
        apply_flame_params_to_chumpy_model(
            model, stage1_flame_param, args.timestep, stage1_vertices=initial_vertices
        )

        # Verify the chumpy model configuration
        print("Verifying chumpy model configuration...")
        chumpy_vertices_after_config = model.r.copy()

        # Compare with Stage 1 vertices to check if configuration worked
        vertex_diff = np.linalg.norm(
            chumpy_vertices_after_config - initial_vertices, axis=1
        )
        mean_diff = np.mean(vertex_diff)
        max_diff = np.max(vertex_diff)

        print(f"Verification results:")
        print(f"  Mean vertex difference: {mean_diff:.6f}")
        print(f"  Max vertex difference: {max_diff:.6f}")

        if mean_diff < 1e-5:
            print("✓ Chumpy model successfully configured with Stage 1 state")
        else:
            print(
                f"⚠ Warning: Configuration may not be perfect (mean diff: {mean_diff:.6f})"
            )

            # Save debug meshes for comparison
            debug_stage1_path = output_dir / "debug_stage1_reference.obj"
            debug_chumpy_path = output_dir / "debug_chumpy_configured.obj"

            write_simple_obj(
                mesh_v=initial_vertices,
                mesh_f=initial_faces,
                filepath=str(debug_stage1_path),
                verbose=False,
            )

            write_simple_obj(
                mesh_v=chumpy_vertices_after_config,
                mesh_f=model.f,
                filepath=str(debug_chumpy_path),
                verbose=False,
            )

            print(f"Debug meshes saved:")
            print(f"  Stage 1 reference: {debug_stage1_path}")
            print(f"  Chumpy configured: {debug_chumpy_path}")

        # Stage 2: Rigid alignment
        print("\n" + "=" * 50)
        print("STAGE 2: RIGID ALIGNMENT")
        print("=" * 50)

        aligned_scan, aligned_lmk_3d = rigid_align_scan_to_flame(
            scan, lmk_3d, model, lmk_face_idx, lmk_b_coords, output_dir
        )
        print("\n" + "=" * 50)
        print("STAGE 3: NON-RIGID FITTING")
        print("=" * 50)
        # Stage 3: Non-rigid fitting
        fitted_vertices, fitted_faces, fitted_params = stage3_non_rigid_fitting(
            model_to_fit=model,
            target_scan=aligned_scan,
            target_lmks=aligned_lmk_3d,
            lmk_face_idx=lmk_face_idx,
            lmk_b_coords=lmk_b_coords,
            output_dir=output_dir,
        )

        # Final Visualizations & Saving
        aligned_scan_vertices = aligned_scan.v.copy()
        aligned_scan_faces = aligned_scan.f.copy()

        visualize_fitting_stages(
            initial_vertices,
            initial_faces,
            aligned_scan_vertices,
            aligned_scan_faces,
            fitted_vertices,
            fitted_faces,
            output_dir,
        )

        visualize_overlay_comparison(
            initial_vertices, aligned_scan_vertices, fitted_vertices, output_dir
        )

        stage1_params_path = output_dir / "stage1_flame_params_reference.npz"
        if stage1_flame_param is not None:
            # Convert torch tensors to numpy for saving if they are tensors
            numpy_params = {}
            for key, value in stage1_flame_param.items():
                if hasattr(value, "cpu"):
                    numpy_params[key] = value.cpu().numpy()
                else:
                    numpy_params[key] = value
            np.savez(str(stage1_params_path), **numpy_params)
            print(f"Stage1 reference parameters saved to: {stage1_params_path}")

        create_fitting_summary_report(
            {
                "stage1_vertices": initial_vertices,
                "stage1_flame_param": stage1_flame_param,
                "stage2_aligned_scan_vertices": aligned_scan_vertices,
                "stage3_fitted_vertices": fitted_vertices,
                "stage3_fitted_params": fitted_params,
            },
            output_dir,
        )

        print("\n" + "=" * 60)
        print("MRI FITTING PIPELINE COMPLETED")
        print("=" * 60)

    # --- Visualization Mode ---
    else:
        if not TORCH_AVAILABLE:
            print("Error: PyTorch is required for visualization-only mode.")
            print("To run fitting, please provide --scan_path and --lmk_path.")
            sys.exit(1)

        print("Running in visualization-only mode...")
        load_and_visualize_flame(
            args.source_path, timestep=args.timestep, save_mesh=not args.no_save_mesh
        )
