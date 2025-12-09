import chumpy as ch
import os
import sys
import numpy as np
from pathlib import Path
import argparse

# Add flame-fitting to path first
sys.path.append("flame-fitting")

from fitting.util import (
    load_binary_pickle,
    write_simple_obj,
    safe_mkdir,
    get_unit_factor,
)
from fitting.landmarks import (
    load_embedding,
    landmark_error_3d,
    mesh_points_by_barycentric_coordinates,
)
from sbody.alignment.objectives import sample_from_mesh
from sbody.robustifiers import GMOf
from sbody.mesh_distance import ScanToMesh
from smpl_webuser.serialization import load_model
from psbody.mesh import Mesh

# Import fitting related modules before torch
# Try to import torch and related modules with error handling
try:
    import torch
    import matplotlib.pyplot as plt

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


def load_and_visualize_flame(data_path, timestep=0, save_mesh=True):
    """
    Load FLAME parameters from dataset and visualize the result

    Args:
        data_path: Path to dataset directory (same as used in training)
        timestep: Which timestep to visualize (default: 0)
        save_mesh: Whether to save the mesh as OBJ file
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available, cannot load FLAME from training data")
        return None, None

    print(f"Loading dataset from {data_path}")

    # Create visualize directory
    visualize_dir = Path(data_path) / "visualize"
    visualize_dir.mkdir(exist_ok=True)

    # Check if flame_param.npz exists directly
    npz_path = Path(data_path) / "flame_param" / "00000.npz"
    if npz_path.exists():
        return load_from_npz(npz_path, timestep, save_mesh, data_path)

    # Otherwise, try to load from dataset structure like train.py does
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
        gaussians = FlameGaussianModel(
            args.sh_degree,
            args.disable_flame_static_offset,
            args.not_finetune_flame_params,
        )

        scene = Scene(args, gaussians)

        if gaussians.flame_param is None:
            print("No FLAME parameters found in dataset")
            return None, None

        # Print parameter shapes for debugging
        print("FLAME parameter shapes:")
        for key, value in gaussians.flame_param.items():
            print(f"  {key}: {value.shape}")

        # Ensure we have the right timestep
        num_timesteps = gaussians.flame_param["expr"].shape[0]
        if timestep >= num_timesteps:
            print(
                f"Timestep {timestep} exceeds available timesteps {num_timesteps}, using timestep 0"
            )
            timestep = 0

        print(f"Generating mesh for timestep {timestep}/{num_timesteps}")

        # Use the same method as in training
        gaussians.select_mesh_by_timestep(timestep)

        # Get vertices and faces
        vertices = gaussians.verts.squeeze(0).cpu().numpy()  # Remove batch dimension
        faces = gaussians.faces.cpu().numpy()

        print(f"Generated mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

        # Visualize
        visualize_mesh(vertices, faces, visualize_dir, timestep)

        # Save mesh as OBJ file if requested
        if save_mesh:
            obj_path = visualize_dir / f"flame_mesh_t{timestep}.obj"
            save_obj_mesh(str(obj_path), vertices, faces)
            print(f"Mesh saved to {obj_path}")

        return vertices, faces

    except Exception as e:
        print(f"Failed to load from dataset structure: {e}")
        print(
            "Please ensure the dataset path contains FLAME data or flame_param.npz file"
        )
        return None, None


def load_from_npz(npz_path, timestep, save_mesh, data_path):
    """Load directly from NPZ file"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, cannot process NPZ files")
        return None, None

    print(f"Loading FLAME parameters from {npz_path}")

    # Create visualize directory
    visualize_dir = Path(data_path) / "visualize"
    visualize_dir.mkdir(exist_ok=True)

    flame_data = np.load(str(npz_path))

    # Print parameter shapes for debugging
    print("FLAME parameter shapes:")
    for key, value in flame_data.items():
        print(f"  {key}: {value.shape}")

    # Initialize FLAME model (same as in FlameGaussianModel)
    flame_model = FlameHead(
        300,  # n_shape - positional argument
        100,  # n_expr - positional argument
        add_teeth=True,
    ).cuda()

    # Convert numpy arrays to torch tensors
    flame_param = {k: torch.from_numpy(v).float().cuda() for k, v in flame_data.items()}

    # Ensure we have the right timestep
    num_timesteps = flame_param["expr"].shape[0]
    if timestep >= num_timesteps:
        print(
            f"Timestep {timestep} exceeds available timesteps {num_timesteps}, using timestep 0"
        )
        timestep = 0

    print(f"Generating mesh for timestep {timestep}/{num_timesteps}")

    # Generate mesh using FLAME model (same as select_mesh_by_timestep)
    with torch.no_grad():
        verts, verts_cano = flame_model(
            flame_param["shape"][None, ...],  # Add batch dimension
            flame_param["expr"][[timestep]],
            flame_param["rotation"][[timestep]],
            flame_param["neck_pose"][[timestep]],
            flame_param["jaw_pose"][[timestep]],
            flame_param["eyes_pose"][[timestep]],
            flame_param["translation"][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param["static_offset"],
            dynamic_offset=flame_param.get("dynamic_offset", None),
        )

    # Get vertices and faces
    vertices = verts.squeeze(0).cpu().numpy()  # Remove batch dimension
    faces = flame_model.faces.cpu().numpy()

    print(f"Generated mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Visualize
    visualize_mesh(vertices, faces, visualize_dir, timestep)

    # Save mesh as OBJ file if requested
    if save_mesh:
        obj_path = visualize_dir / f"flame_mesh_t{timestep}.obj"
        save_obj_mesh(str(obj_path), vertices, faces)
        print(f"Mesh saved to {obj_path}")

    return vertices, faces


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


def rotation_matrix_to_axis_angle(R):
    """
    Convert rotation matrix to axis-angle representation

    Args:
        R: 3x3 rotation matrix

    Returns:
        axis_angle: 3x1 axis-angle vector (direction is rotation axis, magnitude is angle)
    """
    # Compute rotation angle
    trace = np.trace(R)
    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

    if angle < 1e-6:  # Very small rotation
        return np.array([0.0, 0.0, 0.0])
    elif angle > np.pi - 1e-6:  # Close to 180 degrees
        # Find the eigenvector corresponding to eigenvalue 1
        w, v = np.linalg.eig(R)
        axis = v[:, np.argmax(np.real(w))]
        axis = axis / np.linalg.norm(axis)
        return axis * angle
    else:
        # Standard case
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2 * np.sin(angle))
        return axis * angle


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
        flame_lmks_np = np.array(flame_lmks)

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
        final_flame_lmks_np = np.array(final_flame_lmks)

    lmk_distances = np.linalg.norm(aligned_lmk_3d - final_flame_lmks_np, axis=1)
    print(
        f"Landmark alignment error - Mean: {np.mean(lmk_distances):.6f}, Max: {np.max(lmk_distances):.6f}"
    )

    return aligned_scan, aligned_lmk_3d


def fit_flame_to_scan(scan, lmk_3d, model, lmk_face_idx, lmk_b_coords, output_dir):
    """Fit FLAME model to aligned scan - start from initialized FLAME parameters"""
    print("Step 2: Non-rigid fitting of FLAME to scan...")
    print("Using initialized FLAME model as starting point (not resetting parameters)")

    # Create visualization directory
    vis_dir = output_dir / "fitting_iterations"
    vis_dir.mkdir(exist_ok=True)

    # Variables for optimization
    shape_idx = np.arange(0, 300)  # shape components
    expr_idx = np.arange(300, 400)  # expression components
    used_idx = np.union1d(shape_idx, expr_idx)

    # DO NOT reset parameters - keep initialized values from previous stage
    # The model already has the loaded parameters from NPZ file

    # Save current state before fitting
    print("Current model state before fitting:")
    print(f"  - Shape params norm: {np.linalg.norm(model.betas[:300].r):.6f}")
    print(f"  - Expr params norm: {np.linalg.norm(model.betas[300:400].r):.6f}")
    print(f"  - Pose params norm: {np.linalg.norm(model.pose.r):.6f}")
    print(f"  - Translation: {model.trans.r}")

    free_variables = [model.trans, model.pose, model.betas[used_idx]]

    # Weights
    weights = {
        "s2m": 2.0,  # scan to mesh distance
        "lmk": 1e-2,  # landmark term
        "shape": 1e-4,  # shape regularizer
        "expr": 1e-4,  # expression regularizer
        "pose": 1e-3,  # pose regularizer
    }
    gmo_sigma = 1e-4

    print("Using weights:", weights)

    # Objectives
    lmk_err = landmark_error_3d(
        mesh_verts=model,
        mesh_faces=model.f,
        lmk_3d=lmk_3d,
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
    )

    sampler = sample_from_mesh(scan, sample_type="vertices")
    s2m = ScanToMesh(
        scan,
        model,
        model.f,
        scan_sampler=sampler,
        rho=lambda x: GMOf(x, sigma=gmo_sigma),
    )

    shape_err = weights["shape"] * model.betas[shape_idx]
    expr_err = weights["expr"] * model.betas[expr_idx]
    pose_err = weights["pose"] * model.pose[3:]

    objectives = {
        "s2m": weights["s2m"] * s2m,
        "lmk": weights["lmk"] * lmk_err,
        "shape": shape_err,
        "expr": expr_err,
        "pose": pose_err,
    }

    # Optimization options
    import scipy.sparse as sp

    opt_options = {"disp": 1, "delta_0": 0.1, "e_3": 1e-4, "maxiter": 2000}
    opt_options["sparse_solver"] = lambda A, x: sp.linalg.cg(
        A, x, maxiter=opt_options["maxiter"]
    )[0]

    # Iteration counters for saving intermediate results
    iteration_counter = {"step1": 0, "step2": 0}

    # Callback functions for saving iterations
    def on_step_rigid(_):
        iteration_counter["step1"] += 1
        if iteration_counter["step1"] % 50 == 0 or iteration_counter["step1"] <= 5:
            output_path = (
                vis_dir / f'step1_rigid_iter_{iteration_counter["step1"]:04d}.obj'
            )
            write_simple_obj(
                mesh_v=model.r, mesh_f=model.f, filepath=str(output_path), verbose=False
            )
            print(
                f"Saved rigid fitting iteration {iteration_counter['step1']} to: {output_path}"
            )

    def on_step_nonrigid(_):
        iteration_counter["step2"] += 1
        if iteration_counter["step2"] % 100 == 0 or iteration_counter["step2"] <= 10:
            output_path = (
                vis_dir / f'step2_nonrigid_iter_{iteration_counter["step2"]:04d}.obj'
            )
            write_simple_obj(
                mesh_v=model.r, mesh_f=model.f, filepath=str(output_path), verbose=False
            )
            print(
                f"Saved non-rigid fitting iteration {iteration_counter['step2']} to: {output_path}"
            )

    # Step 2a: Rigid fitting with iteration saving (fine-tune pose and translation only)
    from time import time

    timer_start = time()
    print("Step 2a: Fine-tuning pose and translation...")
    ch.minimize(
        fun=lmk_err,
        x0=[model.trans, model.pose[:3]],
        method="dogleg",
        callback=on_step_rigid,
        options=opt_options,
    )
    timer_end = time()
    print(f"Step 2a: fitting done in {timer_end - timer_start:.2f} sec")

    # Save rigid fitting final result
    rigid_result_path = output_dir / "flame_rigid_fit.obj"
    write_simple_obj(
        mesh_v=model.r, mesh_f=model.f, filepath=str(rigid_result_path), verbose=False
    )
    print(f"Rigid fitting result saved to: {rigid_result_path}")

    # Save step 1 final result in iterations folder
    step1_final_path = vis_dir / "step1_rigid_final.obj"
    write_simple_obj(
        mesh_v=model.r, mesh_f=model.f, filepath=str(step1_final_path), verbose=False
    )
    print(f"Saved step 1 final result to: {step1_final_path}")

    # Step 2b: Non-rigid fitting with iteration saving
    timer_start = time()
    print("Step 2b: Non-rigid fitting (shape + expression + pose)...")
    ch.minimize(
        fun=objectives,
        x0=free_variables,
        method="dogleg",
        callback=on_step_nonrigid,
        options=opt_options,
    )
    timer_end = time()
    print(f"Step 2b: fitting done in {timer_end - timer_start:.2f} sec")

    # Save final fitting result
    final_result_path = output_dir / "flame_final_fit.obj"
    write_simple_obj(
        mesh_v=model.r, mesh_f=model.f, filepath=str(final_result_path), verbose=False
    )
    print(f"Final fitting result saved to: {final_result_path}")

    # Save step 2 final result in iterations folder
    step2_final_path = vis_dir / "step2_nonrigid_final.obj"
    write_simple_obj(
        mesh_v=model.r, mesh_f=model.f, filepath=str(step2_final_path), verbose=False
    )
    print(f"Saved step 2 final result to: {step2_final_path}")

    # Print final state
    print("Final model state after fitting:")
    print(f"  - Shape params norm: {np.linalg.norm(model.betas[:300].r):.6f}")
    print(f"  - Expr params norm: {np.linalg.norm(model.betas[300:400].r):.6f}")
    print(f"  - Pose params norm: {np.linalg.norm(model.pose.r):.6f}")
    print(f"  - Translation: {model.trans.r}")

    # Return fitted parameters
    parms = {"trans": model.trans.r, "pose": model.pose.r, "betas": model.betas.r}

    return model.r, model.f, parms


def fit_scan_to_flame(data_path, scan_path, lmk_path, timestep=0):
    """Main function to fit FLAME to scan"""
    print(f"Fitting FLAME to scan: {scan_path}")
    print(f"Using landmarks: {lmk_path}")

    # Create output directory
    output_dir = Path(data_path) / "fitting_results"
    output_dir.mkdir(exist_ok=True)

    # Load FLAME model once
    model_path = "flame-fitting/models/generic_model.pkl"
    if not os.path.exists(model_path):
        alt_paths = [
            "./flame-fitting/models/generic_model.pkl",
            "./models/generic_model.pkl",
            "models/generic_model.pkl",
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find FLAME model file")

    model = load_model(model_path)
    print(f"Loaded FLAME model from: {model_path}")

    # Stage 1: Save initial neutral FLAME model
    model.betas[:] = 0.0
    model.pose[:] = 0.0
    model.trans[:] = 0.0

    initial_flame_vertices = model.r.copy()
    initial_flame_faces = model.f.copy()

    initial_flame_path = output_dir / "stage1_flame_neutral.obj"
    write_simple_obj(
        mesh_v=initial_flame_vertices,
        mesh_f=initial_flame_faces,
        filepath=str(initial_flame_path),
        verbose=False,
    )
    print(f"Stage 1 - Neutral FLAME mesh saved to: {initial_flame_path}")

    # Stage 2: Initialize model with existing FLAME parameters if available
    npz_path = Path(data_path) / "flame_param" / "00000.npz"
    if npz_path.exists():
        flame_data = {k: v for k, v in np.load(str(npz_path)).items()}
        print("Found existing FLAME parameters, initializing model...")

        # Set shape parameters (first 300 components)
        if "shape" in flame_data:
            shape_params = flame_data["shape"]
            model.betas[: len(shape_params)] = shape_params
            print(f"Initialized shape parameters: {len(shape_params)} components")

        # Set expression parameters (components 300-399)
        if "expr" in flame_data:
            expr_params = (
                flame_data["expr"][timestep]
                if flame_data["expr"].ndim > 1
                else flame_data["expr"]
            )
            model.betas[300 : 300 + len(expr_params)] = expr_params
            print(f"Initialized expression parameters: {len(expr_params)} components")

        # Set pose parameters
        if "rotation" in flame_data:
            rotation = (
                flame_data["rotation"][timestep]
                if flame_data["rotation"].ndim > 1
                else flame_data["rotation"]
            )
            model.pose[:3] = rotation

        if "neck_pose" in flame_data:
            neck_pose = (
                flame_data["neck_pose"][timestep]
                if flame_data["neck_pose"].ndim > 1
                else flame_data["neck_pose"]
            )
            model.pose[3:6] = neck_pose

        if "jaw_pose" in flame_data:
            jaw_pose = (
                flame_data["jaw_pose"][timestep]
                if flame_data["jaw_pose"].ndim > 1
                else flame_data["jaw_pose"]
            )
            model.pose[6:9] = jaw_pose

        # Set translation
        if "translation" in flame_data:
            translation = (
                flame_data["translation"][timestep]
                if flame_data["translation"].ndim > 1
                else flame_data["translation"]
            )
            model.trans[:] = translation

        print("Model initialized with existing FLAME parameters")
    else:
        print(f"No existing FLAME parameters found at {npz_path}")
        print("Keeping neutral model...")

    # Save initialized FLAME mesh (this will be kept fixed during rigid alignment)
    initialized_flame_vertices = model.r.copy()
    initialized_flame_faces = model.f.copy()

    initialized_flame_path = output_dir / "stage2_flame_initialized.obj"
    write_simple_obj(
        mesh_v=initialized_flame_vertices,
        mesh_f=initialized_flame_faces,
        filepath=str(initialized_flame_path),
        verbose=False,
    )
    print(f"Stage 2 - Initialized FLAME mesh saved to: {initialized_flame_path}")

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
    lmk_emb_path = "flame-fitting/models/flame_static_embedding.pkl"
    if not os.path.exists(lmk_emb_path):
        alt_paths = [
            "./flame-fitting/models/flame_static_embedding.pkl",
            "./models/flame_static_embedding.pkl",
            "models/flame_static_embedding.pkl",
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                lmk_emb_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find landmark embedding file")

    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    print("Loaded landmark embedding")

    # Step 1: Rigid alignment (FLAME stays fixed, only move scan)
    print("=== RIGID ALIGNMENT STAGE ===")
    print("FLAME model will remain fixed, only scan will be moved")
    aligned_scan, aligned_lmk_3d = rigid_align_scan_to_flame(
        scan, lmk_3d, model, lmk_face_idx, lmk_b_coords, output_dir
    )

    # Stage 3: Save aligned scan
    aligned_scan_vertices = aligned_scan.v.copy()
    aligned_scan_faces = aligned_scan.f.copy()

    # Create comprehensive visualizations
    visualize_fitting_stages(
        initial_flame_vertices,
        initial_flame_faces,
        initialized_flame_vertices,
        initialized_flame_faces,
        aligned_scan_vertices,
        aligned_scan_faces,
        output_dir,
    )

    # Create overlay comparison
    visualize_overlay_comparison(
        initialized_flame_vertices, aligned_scan_vertices, output_dir
    )

    # Step 2: Non-rigid fitting (now FLAME can change to fit the aligned scan)
    print("\n=== NON-RIGID FITTING STAGE ===")
    print("FLAME model will be optimized to fit the aligned scan")
    fitted_vertices, fitted_faces, fitted_params = fit_flame_to_scan(
        aligned_scan, aligned_lmk_3d, model, lmk_face_idx, lmk_b_coords, output_dir
    )

    # Save fitted parameters
    params_path = output_dir / "fitted_flame_params.npz"
    np.savez(str(params_path), **fitted_params)
    print(f"Fitted parameters saved to: {params_path}")

    print("Fitting completed successfully!")
    return fitted_vertices, fitted_faces, fitted_params


def visualize_fitting_stages(
    initial_flame_v,
    initial_flame_f,
    initialized_flame_v,
    initialized_flame_f,
    aligned_scan_v,
    aligned_scan_f,
    output_dir,
):
    """Visualize the three key stages of fitting process"""
    print("Creating comprehensive visualization...")

    fig = plt.figure(figsize=(18, 12))

    # Stage 1: Initial neutral FLAME model
    ax1 = fig.add_subplot(3, 3, 1, projection="3d")
    ax1.scatter(
        initial_flame_v[:, 0],
        initial_flame_v[:, 1],
        initial_flame_v[:, 2],
        s=0.1,
        alpha=0.6,
        c="blue",
        label="Initial FLAME",
    )
    ax1.set_title("Stage 1: Initial FLAME Model\n(Neutral/Loaded)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

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

    # Stage 2: FLAME with applied parameters
    ax4 = fig.add_subplot(3, 3, 4, projection="3d")
    ax4.scatter(
        initialized_flame_v[:, 0],
        initialized_flame_v[:, 1],
        initialized_flame_v[:, 2],
        s=0.1,
        alpha=0.6,
        c="green",
        label="Initialized FLAME",
    )
    ax4.set_title("Stage 2: FLAME with Parameters\n(Shape + Expression)")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")
    ax4.legend()

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.scatter(
        initialized_flame_v[:, 0],
        initialized_flame_v[:, 1],
        s=0.1,
        alpha=0.6,
        c="green",
    )
    ax5.set_title("Front View (X-Y)")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_aspect("equal")

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.scatter(
        initialized_flame_v[:, 1],
        initialized_flame_v[:, 2],
        s=0.1,
        alpha=0.6,
        c="green",
    )
    ax6.set_title("Side View (Y-Z)")
    ax6.set_xlabel("Y")
    ax6.set_ylabel("Z")
    ax6.set_aspect("equal")

    # Stage 3: Aligned scan
    ax7 = fig.add_subplot(3, 3, 7, projection="3d")
    ax7.scatter(
        aligned_scan_v[:, 0],
        aligned_scan_v[:, 1],
        aligned_scan_v[:, 2],
        s=0.1,
        alpha=0.6,
        c="red",
        label="Aligned Scan",
    )
    ax7.set_title("Stage 3: Aligned Scan\n(After Rigid Registration)")
    ax7.set_xlabel("X")
    ax7.set_ylabel("Y")
    ax7.set_zlabel("Z")
    ax7.legend()

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.scatter(aligned_scan_v[:, 0], aligned_scan_v[:, 1], s=0.1, alpha=0.6, c="red")
    ax8.set_title("Front View (X-Y)")
    ax8.set_xlabel("X")
    ax8.set_ylabel("Y")
    ax8.set_aspect("equal")

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.scatter(aligned_scan_v[:, 1], aligned_scan_v[:, 2], s=0.1, alpha=0.6, c="red")
    ax9.set_title("Side View (Y-Z)")
    ax9.set_xlabel("Y")
    ax9.set_ylabel("Z")
    ax9.set_aspect("equal")

    plt.tight_layout()

    # Save comprehensive visualization
    vis_path = output_dir / "fitting_stages_overview.png"
    plt.savefig(str(vis_path), dpi=150, bbox_inches="tight")
    print(f"Comprehensive visualization saved to: {vis_path}")
    plt.show()


def visualize_overlay_comparison(initialized_flame_v, aligned_scan_v, output_dir):
    """Create overlay visualization to show alignment"""
    print("Creating overlay comparison...")

    fig = plt.figure(figsize=(15, 5))

    # 3D overlay
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(
        initialized_flame_v[:, 0],
        initialized_flame_v[:, 1],
        initialized_flame_v[:, 2],
        s=0.1,
        alpha=0.4,
        c="green",
        label="FLAME (initialized)",
    )
    ax1.scatter(
        aligned_scan_v[:, 0],
        aligned_scan_v[:, 1],
        aligned_scan_v[:, 2],
        s=0.1,
        alpha=0.4,
        c="red",
        label="Scan (aligned)",
    )
    ax1.set_title("3D Overlay - FLAME vs Aligned Scan")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # Front view overlay
    ax2 = fig.add_subplot(132)
    ax2.scatter(
        initialized_flame_v[:, 0],
        initialized_flame_v[:, 1],
        s=0.1,
        alpha=0.4,
        c="green",
        label="FLAME",
    )
    ax2.scatter(
        aligned_scan_v[:, 0],
        aligned_scan_v[:, 1],
        s=0.1,
        alpha=0.4,
        c="red",
        label="Scan",
    )
    ax2.set_title("Front View Overlay (X-Y)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")
    ax2.legend()

    # Side view overlay
    ax3 = fig.add_subplot(133)
    ax3.scatter(
        initialized_flame_v[:, 1],
        initialized_flame_v[:, 2],
        s=0.1,
        alpha=0.4,
        c="green",
        label="FLAME",
    )
    ax3.scatter(
        aligned_scan_v[:, 1],
        aligned_scan_v[:, 2],
        s=0.1,
        alpha=0.4,
        c="red",
        label="Scan",
    )
    ax3.set_title("Side View Overlay (Y-Z)")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    ax3.set_aspect("equal")
    ax3.legend()

    plt.tight_layout()

    # Save overlay visualization
    overlay_path = output_dir / "alignment_overlay.png"
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
        # Fitting mode - doesn't require PyTorch
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
        # Visualization mode - requires PyTorch
        if not TORCH_AVAILABLE:
            print("Error: PyTorch is required for visualization mode")
            print("Please provide --scan_path and --lmk_path for fitting mode")
            sys.exit(1)

        load_and_visualize_flame(
            args.source_path, timestep=args.timestep, save_mesh=not args.no_save_mesh
        )
        print(f"Error: Scan path {args.scan_path} does not exist")
        sys.exit(1)
        if not os.path.exists(args.lmk_path):
            print(f"Error: Landmark path {args.lmk_path} does not exist")
            sys.exit(1)

        fit_scan_to_flame(
            args.source_path, args.scan_path, args.lmk_path, args.timestep
        )
        # else:
        #     # Visualization mode - requires PyTorch
        #     if not TORCH_AVAILABLE:
        #         print("Error: PyTorch is required for visualization mode")
        #         print("Please provide --scan_path and --lmk_path for fitting mode")
        #         sys.exit(1)

        #     load_and_visualize_flame(
        #         args.source_path,
        #         timestep=args.timestep,
        #         save_mesh=not args.no_save_mesh
        #     )
        print(f"Error: Scan path {args.scan_path} does not exist")
        sys.exit(1)
        if not os.path.exists(args.lmk_path):
            print(f"Error: Landmark path {args.lmk_path} does not exist")
            sys.exit(1)

        fit_scan_to_flame(
            args.source_path, args.scan_path, args.lmk_path, args.timestep
        )
    # else:
    #     # Visualization mode - requires PyTorch
    #     if not TORCH_AVAILABLE:
    #         print("Error: PyTorch is required for visualization mode")
    #         print("Please provide --scan_path and --lmk_path for fitting mode")
    #         sys.exit(1)

    #     load_and_visualize_flame(
    #         args.source_path,
    #         timestep=args.timestep,
    #         save_mesh=not args.no_save_mesh
    #     )
