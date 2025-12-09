import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Import the same FLAME model used in training
from flame_model.flame import FlameHead
from scene import Scene, FlameGaussianModel
from arguments import ModelParams


def load_and_visualize_flame(
    data_path, timestep=0, save_mesh=True, temp_output_dir=None
):
    """
    Load FLAME parameters from dataset and visualize the result

    Args:
        data_path: Path to dataset directory (same as used in training)
        timestep: Which timestep to visualize (default: 0)
        save_mesh: Whether to save the mesh as OBJ file
        temp_output_dir: Optional temporary directory for subprocess output
    """
    print(f"Loading dataset from {data_path}")

    # Create visualize directory
    if temp_output_dir is not None:
        visualize_dir = Path(temp_output_dir)
        visualize_dir.mkdir(exist_ok=True)
        print(f"Using temporary output directory: {visualize_dir}")
    else:
        visualize_dir = Path(data_path) / "visualize"
        visualize_dir.mkdir(exist_ok=True)

    # Check if flame_param.npz exists directly
    npz_path = Path(data_path) / "flame_param" / "00000.npz"
    if npz_path.exists():
        return load_from_npz(
            npz_path, timestep, save_mesh, visualize_dir, temp_output_dir
        )

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
        # NOTE: Override to use add_teeth=False for fitting compatibility
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
            print("No FLAME parameters found in dataset")
            return None, None

        # Check and handle parameter compatibility for 5023 vs 5143 vertices
        print("Checking FLAME parameter compatibility...")
        flame_vertices_count = gaussians.flame_model.v_template.shape[0]
        print(f"FLAME model vertex count: {flame_vertices_count}")

        # Handle static_offset dimension mismatch
        if "static_offset" in gaussians.flame_param:
            static_offset = gaussians.flame_param["static_offset"]
            static_offset_vertices = (
                static_offset.shape[0]
                if len(static_offset.shape) == 2
                else static_offset.shape[1]
            )
            print(f"Static offset vertex count: {static_offset_vertices}")

            if static_offset_vertices > flame_vertices_count:
                print(
                    f"Truncating static_offset from {static_offset_vertices} to {flame_vertices_count} vertices"
                )
                if len(static_offset.shape) == 2:  # (V, 3)
                    gaussians.flame_param["static_offset"] = static_offset[
                        :flame_vertices_count
                    ]
                else:  # (1, V, 3)
                    gaussians.flame_param["static_offset"] = static_offset[
                        :, :flame_vertices_count
                    ]

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

        print("Parameter compatibility check completed")

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

        # Verify vertex count matches expected 5023 (no teeth)
        if vertices.shape[0] == 5023:
            print("✓ Vertex count matches chumpy model (5023 vertices, no teeth)")
        elif vertices.shape[0] == 5143:
            print("⚠ Warning: Got 5143 vertices (with teeth), may cause fitting issues")
        else:
            print(f"⚠ Warning: Unexpected vertex count: {vertices.shape[0]}")

        # Visualize (only if not in temp mode)
        if temp_output_dir is None:
            visualize_mesh(vertices, faces, visualize_dir, timestep)

        # Save mesh as OBJ file if requested
        if save_mesh:
            obj_path = visualize_dir / f"flame_mesh_t{timestep}.obj"
            save_obj_mesh(str(obj_path), vertices, faces)
            print(f"Mesh saved to {obj_path}")

        # Save parameters for subprocess integration
        if temp_output_dir is not None:
            params_path = visualize_dir / f"flame_params_t{timestep}.npz"
            # Convert torch tensors to numpy for saving
            numpy_params = {}
            for key, value in gaussians.flame_param.items():
                if hasattr(value, "cpu"):
                    numpy_params[key] = value.cpu().numpy()
                else:
                    numpy_params[key] = value
            np.savez(str(params_path), **numpy_params)
            print(f"Parameters saved to {params_path}")

        return vertices, faces

    except Exception as e:
        print(f"Failed to load from dataset structure: {e}")
        print(
            "Please ensure the dataset path contains FLAME data or flame_param.npz file"
        )
        return None, None


def load_from_npz(npz_path, timestep, save_mesh, visualize_dir, temp_output_dir=None):
    """Load directly from NPZ file and ensure proper output to temp directory"""
    print(f"Loading FLAME parameters from {npz_path}")

    flame_data = np.load(str(npz_path))

    # Print parameter shapes for debugging
    print("FLAME parameter shapes:")
    for key, value in flame_data.items():
        print(f"  {key}: {value.shape}")

    # Initialize FLAME model (same as in FlameGaussianModel)
    # NOTE: Use add_teeth=False to ensure compatibility with chumpy model (5023 vertices)
    flame_model = FlameHead(
        300,  # n_shape - positional argument
        100,  # n_expr - positional argument
        add_teeth=False,  # Use 5023 vertices to match chumpy model
    ).cuda()

    # Convert numpy arrays to torch tensors
    flame_param = {k: torch.from_numpy(v).float().cuda() for k, v in flame_data.items()}

    # Check and handle parameter compatibility for 5023 vs 5143 vertices
    print("Checking FLAME parameter compatibility...")
    flame_vertices_count = flame_model.v_template.shape[0]
    print(f"FLAME model vertex count: {flame_vertices_count}")

    # Handle static_offset dimension mismatch
    if "static_offset" in flame_param:
        static_offset = flame_param["static_offset"]
        static_offset_vertices = (
            static_offset.shape[0]
            if len(static_offset.shape) == 2
            else static_offset.shape[1]
        )
        print(f"Static offset vertex count: {static_offset_vertices}")

        if static_offset_vertices > flame_vertices_count:
            print(
                f"Truncating static_offset from {static_offset_vertices} to {flame_vertices_count} vertices"
            )
            if len(static_offset.shape) == 2:  # (V, 3)
                flame_param["static_offset"] = static_offset[:flame_vertices_count]
            else:  # (1, V, 3)
                flame_param["static_offset"] = static_offset[:, :flame_vertices_count]

    # Handle dynamic_offset dimension mismatch
    if "dynamic_offset" in flame_param:
        dynamic_offset = flame_param["dynamic_offset"]
        if len(dynamic_offset.shape) == 3:  # (T, V, 3)
            dynamic_offset_vertices = dynamic_offset.shape[1]
            print(f"Dynamic offset vertex count: {dynamic_offset_vertices}")

            if dynamic_offset_vertices > flame_vertices_count:
                print(
                    f"Truncating dynamic_offset from {dynamic_offset_vertices} to {flame_vertices_count} vertices"
                )
                flame_param["dynamic_offset"] = dynamic_offset[:, :flame_vertices_count]

    print("Parameter compatibility check completed")

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

    # Visualize (only if not in temp mode)
    if temp_output_dir is None:
        visualize_mesh(vertices, faces, visualize_dir, timestep)

    # Save mesh as OBJ file if requested
    if save_mesh:
        obj_path = visualize_dir / f"flame_mesh_t{timestep}.obj"
        save_obj_mesh(str(obj_path), vertices, faces)
        print(f"Mesh saved to {obj_path}")

    # Save parameters for subprocess integration - always save when temp_output_dir is provided
    if temp_output_dir is not None:
        params_path = visualize_dir / f"flame_params_t{timestep}.npz"
        # Convert torch tensors to numpy for saving
        numpy_params = {}
        for key, value in flame_param.items():
            if hasattr(value, "cpu"):
                numpy_params[key] = value.cpu().numpy()
            else:
                numpy_params[key] = value
        np.savez(str(params_path), **numpy_params)
        print(f"Parameters saved to {params_path} for subprocess integration")

    return vertices, faces


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
    parser = argparse.ArgumentParser(description="Load and visualize FLAME parameters")
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
        "--save-to-temp",
        type=str,
        help="Save output to temporary directory for subprocess integration",
    )

    args = parser.parse_args()

    if not os.path.exists(args.source_path):
        print(f"Error: Source path {args.source_path} does not exist")
        sys.exit(1)

    load_and_visualize_flame(
        args.source_path,
        timestep=args.timestep,
        save_mesh=not args.no_save_mesh,
        temp_output_dir=args.save_to_temp,
    )
