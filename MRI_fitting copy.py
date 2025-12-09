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
    # This helps resolve direct imports like `from fitting.util import ...`
    subdirs_to_add = ["fitting", "sbody", "smpl_webuser", "psbody"]
    for subdir in subdirs_to_add:
        full_subdir_path = flame_fitting_dir / subdir
        if full_subdir_path.exists() and str(full_subdir_path) not in sys.path:
            # We add the parent of the subdir, which is flame-fitting
            # The imports should be relative to a path entry.
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
    import matplotlib.pyplot as plt


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

        # Verify vertex count matches expected 5023 (no teeth)
        if vertices.shape[0] == 5023:
            print(
                "✓ Subprocess returned 5023 vertices (no teeth) - compatible with chumpy model"
            )
        elif vertices.shape[0] == 5143:
            print("⚠ Warning: Subprocess returned 5143 vertices (with teeth)")
            print("This suggests load_flame.py still uses add_teeth=True")
        else:
            print(
                f"⚠ Warning: Unexpected vertex count from subprocess: {vertices.shape[0]}"
            )

        # Load parameters if available - keep in memory as numpy arrays
        flame_params = None
        if temp_params_path.exists():
            flame_params = dict(np.load(str(temp_params_path)))
            print(f"Loaded FLAME parameters from subprocess - keeping in memory")

            # Print parameter shapes for debugging
            print("FLAME parameter shapes loaded:")
            for key, value in flame_params.items():
                print(f"  {key}: {value.shape}")

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

    Modified to use add_teeth=False to ensure compatibility with chumpy model (5023 vertices)
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

            # Initialize FlameGaussianModel and Scene like in train.py
            # NOTE: Use add_teeth=False to match chumpy model (5023 vertices instead of 5143)
            gaussians = FlameGaussianModel(
                args.sh_degree,
                args.disable_flame_static_offset,
                args.not_finetune_flame_params,
            )

            # Override flame_model to use no teeth version for fitting compatibility
            gaussians.flame_model = FlameHead(
                gaussians.n_shape,
                gaussians.n_expr,
                add_teeth=False,  # Use 5023 vertices to match chumpy model
            ).cuda()

            scene = Scene(args, gaussians)

            if gaussians.flame_param is None:
                raise Exception("No FLAME parameters found in dataset")

            # Check and handle parameter compatibility for 5023 vs 5143 vertices
            print("Checking FLAME parameter compatibility...")
            flame_vertices_count = gaussians.flame_model.v_template.shape[0]
            print(f"FLAME model vertex count: {flame_vertices_count}")

            # Handle static_offset dimension mismatch
            if "static_offset" in gaussians.flame_param:
                static_offset = gaussians.flame_param["static_offset"]
                static_offset_vertices = static_offset.shape[0]
                print(f"Static offset vertex count: {static_offset_vertices}")

                if static_offset_vertices > flame_vertices_count:
                    print(
                        f"Truncating static_offset from {static_offset_vertices} to {flame_vertices_count} vertices"
                    )
                    gaussians.flame_param["static_offset"] = static_offset[
                        :flame_vertices_count
                    ]
                elif static_offset_vertices < flame_vertices_count:
                    print(
                        f"Padding static_offset from {static_offset_vertices} to {flame_vertices_count} vertices"
                    )
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
                    print(f"Dynamic offset vertex count: {dynamic_offset_vertices}")

                    if dynamic_offset_vertices > flame_vertices_count:
                        print(
                            f"Truncating dynamic_offset from {dynamic_offset_vertices} to {flame_vertices_count} vertices"
                        )
                        gaussians.flame_param["dynamic_offset"] = dynamic_offset[
                            :, :flame_vertices_count
                        ]
                    elif dynamic_offset_vertices < flame_vertices_count:
                        print(
                            f"Padding dynamic_offset from {dynamic_offset_vertices} to {flame_vertices_count} vertices"
                        )
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
                print(
                    f"Timestep {timestep} exceeds available timesteps {num_timesteps}, using timestep 0"
                )
                timestep = 0

            print(f"Generating mesh for timestep {timestep}/{num_timesteps}")

            # Use the same method as in training to generate mesh
            gaussians.select_mesh_by_timestep(timestep)

            # Get vertices and faces
            vertices = gaussians.verts.squeeze(0).cpu().numpy()
            faces = gaussians.faces.cpu().numpy()

            print(
                f"Direct approach: Generated FLAME mesh with parameters: {vertices.shape[0]} vertices, {faces.shape[0]} faces"
            )

            # Verify vertex count matches expected 5023 (no teeth)
            if vertices.shape[0] == 5023:
                print("✓ Vertex count matches chumpy model (5023 vertices, no teeth)")
            elif vertices.shape[0] == 5143:
                print(
                    "⚠ Warning: Got 5143 vertices (with teeth), may cause fitting issues"
                )
            else:
                print(f"⚠ Warning: Unexpected vertex count: {vertices.shape[0]}")

            return vertices, faces, gaussians.flame_param

        except Exception as e:
            print(f"Direct PyTorch approach failed: {e}")
            print("Falling back to subprocess approach...")

    # Fallback to subprocess approach
    return call_load_flame_subprocess(data_path, timestep)


def apply_flame_params_to_chumpy_model(model, flame_param, timestep=0):
    """Apply FLAME parameters from dataset to chumpy model - consistently using stage1 params"""
    if flame_param is None:
        print("No FLAME parameters to apply, using neutral model")
        model.betas[:] = 0.0
        model.pose[:] = 0.0
        model.trans[:] = 0.0
        return

    print(f"Applying Stage1 FLAME parameters to chumpy model for timestep {timestep}")

    # Check vertex count compatibility
    model_vertices = model.r.shape[0] if hasattr(model.r, "shape") else len(model.r)
    print(f"Chumpy model vertex count: {model_vertices}")

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
            return param[0]  # Use first timestep if requested timestep doesn't exist
        else:
            return param

    # Apply shape parameters
    shape_params = extract_param(flame_param, "shape", timestep)
    if shape_params is not None:
        model.betas[: len(shape_params)] = shape_params
        print(f"Applied shape parameters: {len(shape_params)} components")

    # Apply expression parameters
    expr_params = extract_param(flame_param, "expr", timestep)
    if expr_params is not None:
        model.betas[300 : 300 + len(expr_params)] = expr_params
        print(f"Applied expression parameters: {len(expr_params)} components")

    # Apply pose parameters
    rotation = extract_param(flame_param, "rotation", timestep)
    if rotation is not None:
        model.pose[:3] = rotation
        print(f"Applied rotation: {rotation}")

    neck_pose = extract_param(flame_param, "neck_pose", timestep)
    if neck_pose is not None:
        model.pose[3:6] = neck_pose
        print(f"Applied neck pose: {neck_pose}")

    jaw_pose = extract_param(flame_param, "jaw_pose", timestep)
    if jaw_pose is not None:
        model.pose[6:9] = jaw_pose
        print(f"Applied jaw pose: {jaw_pose}")

    # Apply translation
    translation = extract_param(flame_param, "translation", timestep)
    if translation is not None:
        model.trans[:] = translation
        print(f"Applied translation: {translation}")

    print("Successfully applied Stage1 FLAME parameters to chumpy model")


def load_and_visualize_flame(data_path, timestep=0, save_mesh=True):
    """
    Load FLAME parameters from dataset and visualize the result

    Args:
        data_path: Path to dataset directory (same as used in training)
        timestep: Which timestep to visualize (default: 0)
        save_mesh: Whether to save the mesh as OBJ file
    """
    print(f"Loading dataset from {data_path}")

    # Create visualize directory
    visualize_dir = Path(data_path) / "visualize"
    visualize_dir.mkdir(exist_ok=True)

    # Use subprocess approach to load FLAME with parameters
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
    """Rigidly align scan landmarks to FLAME landmarks - only move scan, keep FLAME fixed"""
    print("Step 1: Rigid alignment of scan to FLAME (moving scan only)...")

    # Get FLAME landmarks from current model state (already initialized with parameters)
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

    print(f"Procrustes alignment - Scale: {scale:.4f}")

    # Apply transformation to scan (transform scan to match FLAME)
    scan_centered = scan.v - np.mean(lmk_3d, axis=0)
    aligned_scan_v = (scale * (R @ scan_centered.T)).T + np.mean(flame_lmks_np, axis=0)

    # Create aligned scan
    aligned_scan = Mesh(v=aligned_scan_v, f=scan.f.copy())
    aligned_lmk_3d = transformed_lmks

    # Save aligned scan
    aligned_scan_path = output_dir / "scan_aligned.obj"
    write_simple_obj(
        mesh_v=aligned_scan.v,
        mesh_f=aligned_scan.f,
        filepath=str(aligned_scan_path),
        verbose=False,
    )
    print(f"Aligned scan saved to: {aligned_scan_path}")

    # Verify alignment by checking landmark distances
    final_flame_lmks = mesh_points_by_barycentric_coordinates(
        model, model.f, lmk_face_idx, lmk_b_coords
    )
    if hasattr(final_flame_lmks, "r"):
        final_flame_lmks_np = final_flame_lmks.r
    else:
        final_flame_lmks_np = final_flame_lmks

    lmk_distances = np.linalg.norm(aligned_lmk_3d - final_flame_lmks_np, axis=1)
    print(
        f"Landmark alignment error - Mean: {np.mean(lmk_distances):.6f}, Max: {np.max(lmk_distances):.6f}"
    )

    return aligned_scan, aligned_lmk_3d


def stage3_non_rigid_fitting(
    model_to_fit,  # chumpy model, initialized with stage1 params
    target_scan,  # aligned scan from stage2
    target_lmks,  # aligned landmarks from stage2
    lmk_face_idx,
    lmk_b_coords,
    output_dir,
    shape_num=100,
    expr_num=50,
):
    """
    Stage 3: Non-rigidly fit the FLAME model to the aligned scan.
    The model_to_fit is optimized, the target_scan remains fixed.
    """
    print("\n" + "=" * 50)
    print("STAGE 3: NON-RIGID FITTING (FLAME model fits to scan)")
    print("=" * 50)

    # --- 1. Define weights for optimization objectives ---
    weights = {
        "s2m": 2.0,  # scan-to-mesh distance
        "lmk": 1e-2,  # landmark distance
        "shape": 1e-4,  # shape regularization
        "expr": 1e-4,  # expression regularization
        "pose": 1e-3,  # pose regularization (neck, jaw)
    }
    gmo_sigma = 1e-4  # sigma for Geman-McClure robustifier

    print("Using optimization weights:")
    for kk, vv in weights.items():
        print(f"  - {kk}: {vv}")
    print(f"  - gmo_sigma: {gmo_sigma}")

    # --- 2. Define variables to be optimized ---
    shape_idx = np.arange(0, min(300, shape_num))
    expr_idx = np.arange(300, 300 + min(100, expr_num))
    used_idx = np.union1d(shape_idx, expr_idx)

    # The model is already initialized, we optimize its parameters
    free_variables = [
        model_to_fit.trans,
        model_to_fit.pose,
        model_to_fit.betas[used_idx],
    ]

    # --- 3. Define optimization objectives ---
    # Landmark error (model vs. target landmarks)
    lmk_err = landmark_error_3d(
        mesh_verts=model_to_fit,
        mesh_faces=model_to_fit.f,
        lmk_3d=target_lmks,
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
    )

    # Scan-to-mesh distance (target_scan vs. model)
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
    pose_err = weights["pose"] * model_to_fit.pose[3:]  # Exclude global rotation

    objectives = {
        "s2m": weights["s2m"] * s2m,
        "lmk": weights["lmk"] * lmk_err,
        "shape": shape_err,
        "expr": expr_err,
        "pose": pose_err,
    }

    # --- 4. Setup optimization ---
    import scipy.sparse as sp

    opt_options = {
        "disp": 1,
        "delta_0": 0.1,
        "e_3": 1e-4,
        "maxiter": 2000,
        "sparse_solver": lambda A, x: sp.linalg.cg(A, x, maxiter=2000)[0],
    }

    def on_step(_):
        pass

    # --- 5. Run optimization ---
    print("\nStarting non-rigid optimization...")
    ch.minimize(
        fun=objectives,
        x0=free_variables,
        method="dogleg",
        callback=on_step,
        options=opt_options,
    )
    print("Non-rigid optimization finished.")

    # --- 6. Save results ---
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
    print(f"Stage 3 - Fitted FLAME model saved to: {fitted_mesh_path}")

    # Save fitted parameters
    fitted_params_path = output_dir / "stage3_fitted_flame_params.npz"
    np.savez(str(fitted_params_path), **fitted_params)
    print(f"Stage 3 - Fitted parameters saved to: {fitted_params_path}")

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


def create_mri_fitting_pipeline(data_path, scan_path, lmk_path, timestep=0):
    """Complete MRI fitting pipeline - Stage1: FLAME with params, Stage2: scan alignment only"""
    print("=" * 60)
    print("MRI FITTING PIPELINE")
    print("=" * 60)

    # Create output directory
    output_dir = Path(data_path) / "fitting_results"
    output_dir.mkdir(exist_ok=True)

    # Stage 1: Load FLAME model with parameters - this is our reference
    print("\n" + "=" * 50)
    print("STAGE 1: LOADING FLAME MODEL WITH PARAMETERS (reference)")
    print("=" * 50)

    initial_vertices, initial_faces, stage1_flame_param = (
        load_flame_with_params_from_dataset(data_path, timestep)
    )

    if initial_vertices is None:
        print("Failed to load FLAME model with parameters")
        return None

    print("Stage1 FLAME parameters loaded - this will be our reference FLAME model")

    # Verify vertex count compatibility
    print(f"Stage1 FLAME mesh vertex count: {initial_vertices.shape[0]}")
    if initial_vertices.shape[0] != 5023:
        print(
            f"⚠ Warning: Expected 5023 vertices for chumpy compatibility, got {initial_vertices.shape[0]}"
        )
        if initial_vertices.shape[0] == 5143:
            print(
                "This suggests the model still has teeth. Consider checking add_teeth parameter."
            )
    else:
        print("✓ Vertex count is compatible with chumpy model")

    # Save Stage 1 result
    stage1_path = output_dir / "stage1_flame_with_params.obj"
    write_simple_obj(
        mesh_v=initial_vertices,
        mesh_f=initial_faces,
        filepath=str(stage1_path),
        verbose=False,
    )
    print(f"Stage 1 - FLAME model with parameters saved to: {stage1_path}")

    # Stage 2: Rigid alignment - FLAME stays fixed, move scan to align
    print("\n" + "=" * 50)
    print("STAGE 2: RIGID ALIGNMENT (FLAME fixed, scan moves)")
    print("=" * 50)

    # Load FLAME model for fitting and apply Stage1 parameters
    model_path = find_model_file(
        "generic_model.pkl",
        ["flame-fitting/models", "./flame-fitting/models", "./models", "models"],
    )

    model = load_model(model_path)

    # Apply Stage1 parameters to create the reference FLAME model
    apply_flame_params_to_chumpy_model(model, stage1_flame_param, timestep)

    print("FLAME model configured with Stage1 parameters - this will stay FIXED")

    # Verify chumpy model vertex count
    chumpy_vertices = model.r.shape[0] if hasattr(model.r, "shape") else len(model.r)
    print(f"Chumpy model vertex count: {chumpy_vertices}")

    if chumpy_vertices != initial_vertices.shape[0]:
        print(f"⚠ Warning: Vertex count mismatch!")
        print(f"  - Stage1 FLAME: {initial_vertices.shape[0]} vertices")
        print(f"  - Chumpy model: {chumpy_vertices} vertices")
        print("This may cause fitting issues.")
    else:
        print("✓ Vertex counts match between Stage1 FLAME and chumpy model")

    # Load scan and landmarks
    scan = Mesh(filename=scan_path)
    if lmk_path.endswith(".pp"):
        from fitting.landmarks import load_picked_points

        lmk_3d = load_picked_points(lmk_path)
        print("loaded scan landmark from (pp):", lmk_path)
    else:
        lmk_3d = np.load(lmk_path)
        print("loaded scan landmark from (npy):", lmk_path)

    print(f"Loaded scan: {scan.v.shape[0]} vertices, {scan.f.shape[0]} faces")
    print(f"Loaded landmarks: {lmk_3d.shape}")

    # Load landmark embedding
    lmk_emb_path = find_model_file(
        "flame_static_embedding.pkl",
        ["flame-fitting/models", "./flame-fitting/models", "./models", "models"],
    )

    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    print("Loaded landmark embedding")

    # Rigid alignment (FLAME stays fixed, only move scan)
    print("\n=== RIGID ALIGNMENT STAGE ===")
    print("FLAME model will remain fixed, only scan will be moved")
    aligned_scan, aligned_lmk_3d = rigid_align_scan_to_flame(
        scan, lmk_3d, model, lmk_face_idx, lmk_b_coords, output_dir
    )

    # --- Stage 3: Non-rigid fitting ---
    # Now, the aligned_scan is the target, and we fit the FLAME model to it.
    # We use the same `model` object, which is already initialized with Stage1 parameters.
    fitted_vertices, fitted_faces, fitted_params = stage3_non_rigid_fitting(
        model_to_fit=model,
        target_scan=aligned_scan,
        target_lmks=aligned_lmk_3d,
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        output_dir=output_dir,
    )

    # --- Final Visualizations & Saving ---
    aligned_scan_vertices = aligned_scan.v.copy()
    aligned_scan_faces = aligned_scan.f.copy()

    # Create visualizations including Stage 3 results
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

    # Save parameters
    stage1_params_path = output_dir / "stage1_flame_params_reference.npz"
    np.savez(str(stage1_params_path), **stage1_flame_param)
    print(f"Stage1 reference parameters saved to: {stage1_params_path}")

    print("\n" + "=" * 60)
    print("MRI FITTING PIPELINE COMPLETED SUCCESSFULLY!")
    print("Stage1: FLAME with loaded params (including shape) - REFERENCE")
    print("Stage2: Scan aligned to FLAME (FLAME fixed, scan moved)")
    print("Stage3: FLAME model non-rigidly fitted to aligned scan")
    print("=" * 60)

    return {
        "stage1_vertices": initial_vertices,
        "stage1_flame_param": stage1_flame_param,
        "stage2_aligned_scan_vertices": aligned_scan_vertices,
        "stage3_fitted_vertices": fitted_vertices,
        "stage3_fitted_params": fitted_params,
    }


def fit_scan_to_flame(data_path, scan_path, lmk_path, timestep=0):
    """Main function to run MRI fitting pipeline with scan alignment (no fitting stage)"""
    return create_mri_fitting_pipeline(data_path, scan_path, lmk_path, timestep)


def visualize_fitting_stages(
    initial_flame_v,
    initial_flame_f,
    aligned_scan_v,
    aligned_scan_f,
    fitted_flame_v,
    fitted_flame_f,
    output_dir,
):
    """Visualize the three key stages of fitting process"""
    print("Creating comprehensive visualization...")

    fig = plt.figure(figsize=(18, 12))
    plt.suptitle("MRI Fitting Pipeline Stages", fontsize=16, y=0.95)

    # --- Stage 1: Initial FLAME model (Reference) ---
    ax1 = fig.add_subplot(3, 3, 1, projection="3d")
    ax1.scatter(
        initial_flame_v[:, 0],
        initial_flame_v[:, 1],
        initial_flame_v[:, 2],
        s=0.1,
        alpha=0.6,
        c="blue",
    )
    ax1.set_title("Stage 1: Initial FLAME (Reference)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.scatter(
        initial_flame_v[:, 0], initial_flame_v[:, 1], s=0.1, alpha=0.6, c="blue"
    )
    ax2.set_title("Front View (X-Y)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(3, 3, 3)
    ax3.scatter(
        initial_flame_v[:, 1], initial_flame_v[:, 2], s=0.1, alpha=0.6, c="blue"
    )
    ax3.set_title("Side View (Y-Z)")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    ax3.set_aspect("equal")

    # --- Stage 2: Aligned Scan (Target) ---
    ax4 = fig.add_subplot(3, 3, 4, projection="3d")
    ax4.scatter(
        aligned_scan_v[:, 0],
        aligned_scan_v[:, 1],
        aligned_scan_v[:, 2],
        s=0.1,
        alpha=0.6,
        c="red",
    )
    ax4.set_title("Stage 2: Aligned Scan (Target)")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.scatter(aligned_scan_v[:, 0], aligned_scan_v[:, 1], s=0.1, alpha=0.6, c="red")
    ax5.set_title("Front View (X-Y)")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_aspect("equal")

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.scatter(aligned_scan_v[:, 1], aligned_scan_v[:, 2], s=0.1, alpha=0.6, c="red")
    ax6.set_title("Side View (Y-Z)")
    ax6.set_xlabel("Y")
    ax6.set_ylabel("Z")
    ax6.set_aspect("equal")

    # --- Stage 3: Fitted FLAME Model (Result) ---
    ax7 = fig.add_subplot(3, 3, 7, projection="3d")
    ax7.scatter(
        fitted_flame_v[:, 0],
        fitted_flame_v[:, 1],
        fitted_flame_v[:, 2],
        s=0.1,
        alpha=0.6,
        c="green",
    )
    ax7.set_title("Stage 3: Fitted FLAME (Result)")
    ax7.set_xlabel("X")
    ax7.set_ylabel("Y")
    ax7.set_zlabel("Z")

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.scatter(fitted_flame_v[:, 0], fitted_flame_v[:, 1], s=0.1, alpha=0.6, c="green")
    ax8.set_title("Front View (X-Y)")
    ax8.set_xlabel("X")
    ax8.set_ylabel("Y")
    ax8.set_aspect("equal")

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.scatter(fitted_flame_v[:, 1], fitted_flame_v[:, 2], s=0.1, alpha=0.6, c="green")
    ax9.set_title("Side View (Y-Z)")
    ax9.set_xlabel("Y")
    ax9.set_ylabel("Z")
    ax9.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save comprehensive visualization
    vis_path = output_dir / "fitting_stages_overview.png"
    plt.savefig(str(vis_path), dpi=150, bbox_inches="tight")
    print(f"Comprehensive visualization saved to: {vis_path}")
    plt.show()


def visualize_overlay_comparison(
    initial_flame_v, aligned_scan_v, fitted_flame_v, output_dir
):
    """Create overlay visualization to show alignment and fitting result"""
    print("Creating overlay comparison...")

    fig = plt.figure(figsize=(18, 6))
    plt.suptitle("Overlay Comparison", fontsize=16, y=0.98)

    # --- 3D Overlay ---
    ax1 = fig.add_subplot(131, projection="3d")
    # Target Scan
    ax1.scatter(
        aligned_scan_v[:, 0],
        aligned_scan_v[:, 1],
        aligned_scan_v[:, 2],
        s=0.1,
        alpha=0.3,
        c="red",
        label="Scan (Target)",
    )
    # Fitted FLAME
    ax1.scatter(
        fitted_flame_v[:, 0],
        fitted_flame_v[:, 1],
        fitted_flame_v[:, 2],
        s=0.1,
        alpha=0.5,
        c="green",
        label="FLAME (Fitted)",
    )
    ax1.set_title("3D Overlay: Fitted FLAME vs. Scan")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # --- Front View Overlay ---
    ax2 = fig.add_subplot(132)
    ax2.scatter(
        aligned_scan_v[:, 0],
        aligned_scan_v[:, 1],
        s=0.1,
        alpha=0.3,
        c="red",
        label="Scan",
    )
    ax2.scatter(
        fitted_flame_v[:, 0],
        fitted_flame_v[:, 1],
        s=0.1,
        alpha=0.5,
        c="green",
        label="FLAME",
    )
    ax2.set_title("Front View Overlay (X-Y)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")
    ax2.legend()

    # --- Side View Overlay ---
    ax3 = fig.add_subplot(133)
    ax3.scatter(
        aligned_scan_v[:, 1],
        aligned_scan_v[:, 2],
        s=0.1,
        alpha=0.3,
        c="red",
        label="Scan",
    )
    ax3.scatter(
        fitted_flame_v[:, 1],
        fitted_flame_v[:, 2],
        s=0.1,
        alpha=0.5,
        c="green",
        label="FLAME",
    )
    ax3.set_title("Side View Overlay (Y-Z)")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    ax3.set_aspect("equal")
    ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save overlay visualization
    overlay_path = output_dir / "fitting_overlay.png"
    plt.savefig(str(overlay_path), dpi=150, bbox_inches="tight")
    print(f"Overlay visualization saved to: {overlay_path}")
    plt.show()


def visualize_mesh(vertices, faces, output_dir, timestep):
    """Visualize mesh using matplotlib"""
    # Simple visualization using matplotlib
    fig = plt.figure(figsize=(12, 4))

    # Plot 1: 3D perspective view
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0.1, alpha=0.6)
    ax1.set_title("3D Mesh - Perspective View")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Plot 2: Front view (X-Y projection)
    ax2 = fig.add_subplot(132)
    ax2.scatter(vertices[:, 0], vertices[:, 1], s=0.1, alpha=0.6)
    ax2.set_title("Front View (X-Y)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")

    # Plot 3: Side view (Y-Z projection)
    ax3 = fig.add_subplot(133)
    ax3.scatter(vertices[:, 1], vertices[:, 2], s=0.1, alpha=0.6)
    ax3.set_title("Side View (Y-Z)")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    ax3.set_aspect("equal")

    plt.tight_layout()

    # Save visualization
    vis_path = output_dir / f"flame_visualization_t{timestep}.png"
    plt.savefig(str(vis_path), dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {vis_path}")
    plt.show()


def save_obj_mesh(filepath, vertices, faces):
    """Save mesh in OBJ format"""
    with open(filepath, "w") as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


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

    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        print(f"Error: Source path {args.source_path} does not exist")
        sys.exit(1)

    if args.scan_path and args.lmk_path:
        # Fitting mode
        if not os.path.exists(args.scan_path):
            print(f"Error: Scan path {args.scan_path} does not exist")
            sys.exit(1)
        if not os.path.exists(args.lmk_path):
            print(f"Error: Landmark path {args.lmk_path} does not exist")
            sys.exit(1)

        fit_scan_to_flame(
            args.source_path, args.scan_path, args.lmk_path, args.timestep
        )
    else:
        # Visualization mode
        if not TORCH_AVAILABLE:
            print("Error: PyTorch is required for visualization mode")
            print("Please provide --scan_path and --lmk_path for fitting mode")
            sys.exit(1)

        load_and_visualize_flame(
            args.source_path, timestep=args.timestep, save_mesh=not args.no_save_mesh
        )
