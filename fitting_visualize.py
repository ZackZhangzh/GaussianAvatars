"""
Visualization utilities for MRI fitting pipeline
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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
    ax5.scatter(aligned_scan_v[:, 0],
                aligned_scan_v[:, 1], s=0.1, alpha=0.6, c="red")
    ax5.set_title("Front View (X-Y)")
    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_aspect("equal")

    ax6 = fig.add_subplot(3, 3, 6)
    ax6.scatter(aligned_scan_v[:, 1],
                aligned_scan_v[:, 2], s=0.1, alpha=0.6, c="red")
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
    ax8.scatter(fitted_flame_v[:, 0],
                fitted_flame_v[:, 1], s=0.1, alpha=0.6, c="green")
    ax8.set_title("Front View (X-Y)")
    ax8.set_xlabel("X")
    ax8.set_ylabel("Y")
    ax8.set_aspect("equal")

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.scatter(fitted_flame_v[:, 1],
                fitted_flame_v[:, 2], s=0.1, alpha=0.6, c="green")
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
    ax1.scatter(vertices[:, 0], vertices[:, 1],
                vertices[:, 2], s=0.1, alpha=0.6)
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


def visualize_landmarks(vertices, landmarks, output_dir, title="Landmarks"):
    """Visualize mesh with landmarks highlighted"""
    fig = plt.figure(figsize=(15, 5))

    # 3D view
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                s=0.1, alpha=0.3, c="lightblue", label="Mesh")
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2],
                s=10, alpha=0.8, c="red", label="Landmarks")
    ax1.set_title(f"{title} - 3D View")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # Front view
    ax2 = fig.add_subplot(132)
    ax2.scatter(vertices[:, 0], vertices[:, 1],
                s=0.1, alpha=0.3, c="lightblue")
    ax2.scatter(landmarks[:, 0], landmarks[:, 1], s=10, alpha=0.8, c="red")
    ax2.set_title(f"{title} - Front View")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal")

    # Side view
    ax3 = fig.add_subplot(133)
    ax3.scatter(vertices[:, 1], vertices[:, 2],
                s=0.1, alpha=0.3, c="lightblue")
    ax3.scatter(landmarks[:, 1], landmarks[:, 2], s=10, alpha=0.8, c="red")
    ax3.set_title(f"{title} - Side View")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Z")
    ax3.set_aspect("equal")

    plt.tight_layout()

    # Save visualization
    vis_path = output_dir / f"{title.lower().replace(' ', '_')}_landmarks.png"
    plt.savefig(str(vis_path), dpi=150, bbox_inches="tight")
    print(f"Landmarks visualization saved to: {vis_path}")
    plt.show()


def create_fitting_summary_report(results, output_dir):
    """Create a summary report of the fitting process"""
    report_path = output_dir / "fitting_summary.txt"

    with open(report_path, "w") as f:
        f.write("MRI FITTING PIPELINE SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")

        # Stage 1 info
        if "stage1_vertices" in results and results["stage1_vertices"] is not None:
            stage1_vertices = results["stage1_vertices"]
            f.write(f"STAGE 1 - FLAME WITH PARAMETERS:\n")
            f.write(f"  Vertices: {stage1_vertices.shape[0]}\n")
            f.write(
                f"  Faces: {results.get('stage1_faces', ['Unknown'])[0] if 'stage1_faces' in results else 'Unknown'}\n\n")

        # Stage 2 info
        if "stage2_aligned_scan_vertices" in results and results["stage2_aligned_scan_vertices"] is not None:
            stage2_vertices = results["stage2_aligned_scan_vertices"]
            f.write(f"STAGE 2 - ALIGNED SCAN:\n")
            f.write(f"  Vertices: {stage2_vertices.shape[0]}\n\n")

        # Stage 3 info
        if "stage3_fitted_vertices" in results and results["stage3_fitted_vertices"] is not None:
            stage3_vertices = results["stage3_fitted_vertices"]
            f.write(f"STAGE 3 - FITTED FLAME:\n")
            f.write(f"  Vertices: {stage3_vertices.shape[0]}\n")

            if "stage3_fitted_params" in results and results["stage3_fitted_params"] is not None:
                fitted_params = results["stage3_fitted_params"]
                f.write(f"  Fitted parameters:\n")
                for key, value in fitted_params.items():
                    if hasattr(value, 'shape'):
                        f.write(f"    {key}: {value.shape}\n")
                    else:
                        f.write(f"    {key}: {type(value)}\n")

        f.write("\nPipeline completed successfully!\n")

    print(f"Summary report saved to: {report_path}")
