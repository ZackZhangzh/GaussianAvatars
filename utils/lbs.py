"""
Linear Blend Skinning (LBS) Controller for GaussianAvatars

This module provides LBS functionality to control FLAME skin mesh using 
skull and jaw bone meshes. Ported from the standalone LBS project with 
minimal modifications to integrate into remote_viewer.py.

Key Features:
- Geodesic distance-based weight computation
- Smooth weight transitions with Laplacian smoothing
- Real-time mesh deformation
- Configurable proximity and damping parameters
"""

import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import heapq
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path


@dataclass
class LBSConfig:
    """Configuration parameters for LBS weight computation"""
    
    weight_damping: float = 20.0
    """Damping factor for weight decay. Larger = smoother falloff"""
    
    small_step_radius: float = 5.0
    """Radius for graph connectivity in geodesic computation"""
    
    proximity_threshold: float = 12.0
    """Distance threshold for seed selection"""
    
    use_anatomical_regions: bool = False
    """Whether to use anatomical region-based seed selection"""
    
    jaw_region_y_ratio: float = 0.4
    """Bottom percentage of face considered jaw-influenced"""
    
    jaw_region_z_forward: float = 0.3
    """Front percentage of face considered jaw-influenced"""
    
    max_seeds: int = 300
    """Maximum number of seeds per control group"""
    
    laplacian_iterations: int = 5
    """Number of Laplacian smoothing iterations"""
    
    laplacian_alpha: float = 0.5
    """Laplacian smoothing factor (0=no smoothing, 1=full smoothing)"""
    
    debug: bool = False
    """Enable debug output"""


class LBSController:
    """
    Linear Blend Skinning controller for FLAME mesh deformation.
    
    Controls skin mesh deformation using skull (static) and jaw (moving) bones.
    Implements geodesic distance-based weight computation with smooth falloff.
    """
    
    def __init__(
        self,
        skin_vertices: np.ndarray,
        skull_vertices: np.ndarray,
        jaw_vertices: np.ndarray,
        config: Optional[LBSConfig] = None
    ):
        """
        Initialize LBS controller.
        
        Args:
            skin_vertices: Skin mesh vertices, shape (N, 3)
            skull_vertices: Skull mesh vertices, shape (M, 3)
            jaw_vertices: Jaw mesh vertices, shape (K, 3)
            config: LBS configuration parameters
        """
        self.config = config or LBSConfig()
        
        # Store original vertices
        self.skin_verts_orig = skin_vertices.copy()
        self.skull_verts_orig = skull_vertices.copy()
        self.jaw_verts_orig = jaw_vertices.copy()
        
        # Initialize weights
        print("[LBS] Initializing weights...")
        self.weights = None
        self.jaw_seeds = None
        self.skull_seeds = None
        self._init_weights()
        print("[LBS] Weights initialized successfully")
        
        # Current transformation state
        self.current_transform = {
            'rotation': np.zeros(2),  # (pitch, yaw) in radians
            'translation': np.zeros(3)
        }
    
    def _init_weights(self):
        """Calculate skin weights based on geometry"""
        # 1. Find control seeds
        if self.config.use_anatomical_regions:
            self.jaw_seeds = self._find_jaw_seeds_anatomical()
            self.skull_seeds = self._find_skull_seeds_anatomical()
        else:
            # Pure proximity-based (more reliable)
            self.jaw_seeds = self._find_proximal_points(
                self.skin_verts_orig, 
                self.jaw_verts_orig, 
                self.config.proximity_threshold
            )
            self.skull_seeds = self._find_proximal_points(
                self.skin_verts_orig,
                self.skull_verts_orig,
                self.config.proximity_threshold
            )
        
        print(f"[LBS] Found {len(self.jaw_seeds)} jaw seeds and {len(self.skull_seeds)} skull seeds")
        
        # Spatial diagnostics (only if debug enabled)
        if self.config.debug:
            print(f"\n[LBS] === Spatial Diagnostics (Debug) ===")
            # Bounding boxes
            skin_min, skin_max = self.skin_verts_orig.min(axis=0), self.skin_verts_orig.max(axis=0)
            skin_center = (skin_min + skin_max) / 2
            jaw_min, jaw_max = self.jaw_verts_orig.min(axis=0), self.jaw_verts_orig.max(axis=0)
            jaw_center = (jaw_min + jaw_max) / 2
            skull_min, skull_max = self.skull_verts_orig.min(axis=0), self.skull_verts_orig.max(axis=0)
            skull_center = (skull_min + skull_max) / 2
            
            print(f"[LBS] Skin   bbox: [{skin_min[0]:.3f}, {skin_min[1]:.3f}, {skin_min[2]:.3f}] to [{skin_max[0]:.3f}, {skin_max[1]:.3f}, {skin_max[2]:.3f}], center: [{skin_center[0]:.3f}, {skin_center[1]:.3f}, {skin_center[2]:.3f}]")
            print(f"[LBS] Jaw    bbox: [{jaw_min[0]:.3f}, {jaw_min[1]:.3f}, {jaw_min[2]:.3f}] to [{jaw_max[0]:.3f}, {jaw_max[1]:.3f}, {jaw_max[2]:.3f}], center: [{jaw_center[0]:.3f}, {jaw_center[1]:.3f}, {jaw_center[2]:.3f}]")
            print(f"[LBS] Skull  bbox: [{skull_min[0]:.3f}, {skull_min[1]:.3f}, {skull_min[2]:.3f}] to [{skull_max[0]:.3f}, {skull_max[1]:.3f}, {skull_max[2]:.3f}], center: [{skull_center[0]:.3f}, {skull_center[1]:.3f}, {skull_center[2]:.3f}]")
            
            # Distance analysis
            tree_jaw = KDTree(self.jaw_verts_orig)
            tree_skull = KDTree(self.skull_verts_orig)
            dists_to_jaw, _ = tree_jaw.query(self.skin_verts_orig)
            dists_to_skull, _ = tree_skull.query(self.skin_verts_orig)
            
            print(f"[LBS] Distance to Jaw  - min: {dists_to_jaw.min():.3f}, median: {np.median(dists_to_jaw):.3f}, max: {dists_to_jaw.max():.3f}")
            print(f"[LBS] Distance to Skull - min: {dists_to_skull.min():.3f}, median: {np.median(dists_to_skull):.3f}, max: {dists_to_skull.max():.3f}")
            print(f"[LBS] Proximity threshold: {self.config.proximity_threshold:.3f}")
            print(f"[LBS] Skin points within threshold - jaw: {(dists_to_jaw < self.config.proximity_threshold).sum()}/{len(self.skin_verts_orig)}, skull: {(dists_to_skull < self.config.proximity_threshold).sum()}/{len(self.skin_verts_orig)}")
            
            # Seed spatial distribution
            if len(self.jaw_seeds) > 0:
                jaw_seed_verts = self.skin_verts_orig[self.jaw_seeds]
                jaw_seed_center = jaw_seed_verts.mean(axis=0)
                print(f"[LBS] Jaw seeds   - center: [{jaw_seed_center[0]:.3f}, {jaw_seed_center[1]:.3f}, {jaw_seed_center[2]:.3f}], Y-range: [{jaw_seed_verts[:, 1].min():.3f}, {jaw_seed_verts[:, 1].max():.3f}]")
            
            if len(self.skull_seeds) > 0:
                skull_seed_verts = self.skin_verts_orig[self.skull_seeds]
                skull_seed_center = skull_seed_verts.mean(axis=0)
                print(f"[LBS] Skull seeds - center: [{skull_seed_center[0]:.3f}, {skull_seed_center[1]:.3f}, {skull_seed_center[2]:.3f}], Y-range: [{skull_seed_verts[:, 1].min():.3f}, {skull_seed_verts[:, 1].max():.3f}]")
            
            print(f"[LBS] =====================================\n")

        
        # 2. Compute geodesic weights
        self.weights = self._compute_geodesic_weights()
        
        # 3. Diagnostic output
        w_jaw = self.weights[:, 0]
        w_skull = self.weights[:, 1]
        w_rest = 1.0 - w_jaw - w_skull
        
        print(f"[LBS] Jaw weights   - min: {w_jaw.min():.4f}, max: {w_jaw.max():.4f}, mean: {w_jaw.mean():.4f}")
        print(f"[LBS] Skull weights - min: {w_skull.min():.4f}, max: {w_skull.max():.4f}, mean: {w_skull.mean():.4f}")
        print(f"[LBS] Rest weights  - min: {w_rest.min():.4f}, max: {w_rest.max():.4f}, mean: {w_rest.mean():.4f}")
        print(f"[LBS] Points with >5% jaw influence: {(w_jaw > 0.05).sum()}/{len(w_jaw)}")
    
    def _find_proximal_points(
        self, 
        source_points: np.ndarray, 
        target_surface_points: np.ndarray, 
        threshold: float
    ) -> np.ndarray:
        """Find indices of source points close to target surface"""
        tree = KDTree(target_surface_points)
        dists, _ = tree.query(source_points)
        indices = np.where(dists < threshold)[0]
        
        # Downsample if too many seeds
        if len(indices) > self.config.max_seeds:
            step = len(indices) // self.config.max_seeds
            indices = indices[::step][:self.config.max_seeds]
        
        return indices
    
    def _find_jaw_seeds_anatomical(self) -> np.ndarray:
        """Find jaw control seeds using anatomical geometry"""
        # Proximity-based candidates
        tree = KDTree(self.jaw_verts_orig)
        dists, _ = tree.query(self.skin_verts_orig)
        proximity_candidates = np.where(dists < self.config.proximity_threshold)[0]
        
        # Anatomical region candidates
        y_min, y_max = self.skin_verts_orig[:, 1].min(), self.skin_verts_orig[:, 1].max()
        z_min, z_max = self.skin_verts_orig[:, 2].min(), self.skin_verts_orig[:, 2].max()
        
        y_threshold = y_min + self.config.jaw_region_y_ratio * (y_max - y_min)
        z_threshold = z_min + (1 - self.config.jaw_region_z_forward) * (z_max - z_min)
        
        anatomical_candidates = np.where(
            (self.skin_verts_orig[:, 1] < y_threshold) |
            (self.skin_verts_orig[:, 2] > z_threshold)
        )[0]
        
        # Combine
        combined = np.union1d(proximity_candidates, anatomical_candidates)
        
        # Downsample
        if len(combined) > 200:
            step = len(combined) // 200
            combined = combined[::step][:200]
        
        return combined
    
    def _find_skull_seeds_anatomical(self) -> np.ndarray:
        """Find skull control seeds using anatomical geometry"""
        # Proximity to skull
        tree = KDTree(self.skull_verts_orig)
        dists, _ = tree.query(self.skin_verts_orig)
        proximity_candidates = np.where(dists < self.config.proximity_threshold)[0]
        
        # Upper face region
        y_min, y_max = self.skin_verts_orig[:, 1].min(), self.skin_verts_orig[:, 1].max()
        y_threshold = y_min + self.config.jaw_region_y_ratio * (y_max - y_min)
        
        anatomical_candidates = np.where(self.skin_verts_orig[:, 1] >= y_threshold)[0]
        
        # Combine
        combined = np.union1d(proximity_candidates, anatomical_candidates)
        
        # Downsample
        if len(combined) > 200:
            step = len(combined) // 200
            combined = combined[::step][:200]
        
        return combined
    
    def _dijkstra(self, n: int, adj_list: list, start_nodes: np.ndarray) -> np.ndarray:
        """Multi-source Dijkstra algorithm for geodesic distance computation"""
        dist = np.full(n, np.inf)
        heap = []
        
        for start in start_nodes:
            dist[start] = 0
            heapq.heappush(heap, (0.0, start))
        
        while heap:
            d_u, u = heapq.heappop(heap)
            if d_u > dist[u]:
                continue
            
            for v, weight in adj_list[u]:
                if dist[v] > d_u + weight:
                    dist[v] = d_u + weight
                    heapq.heappush(heap, (dist[v], v))
        
        return dist
    
    def _compute_geodesic_weights(self) -> np.ndarray:
        """Compute LBS weights using geodesic distances"""
        import time
        
        n = len(self.skin_verts_orig)
        
        # Build graph
        print("[LBS] Building connectivity graph...")
        
        # 🐛 CHECK MESH SCALE
        bbox_min = self.skin_verts_orig.min(axis=0)
        bbox_max = self.skin_verts_orig.max(axis=0)
        bbox_size = bbox_max - bbox_min
        mesh_diagonal = np.linalg.norm(bbox_size)
        print(f"[LBS] 🔍 Mesh bounding box: {bbox_size}")
        print(f"[LBS] 🔍 Mesh diagonal: {mesh_diagonal:.4f}")
        print(f"[LBS] 🔍 Configured radius: {self.config.small_step_radius:.4f}")
        
        # Auto-adjust radius if too large
        recommended_radius = mesh_diagonal * 0.02  # 2% of diagonal
        if self.config.small_step_radius > mesh_diagonal * 0.5:
            print(f"[LBS] ⚠️  WARNING: Radius {self.config.small_step_radius:.4f} is TOO LARGE!")
            print(f"[LBS] ⚠️  This will create a complete graph with {len(self.skin_verts_orig)**2} edges!")
            print(f"[LBS] 💡 Recommended radius: {recommended_radius:.4f} (2% of diagonal)")
            print(f"[LBS] 🔧 Auto-adjusting to recommended value...")
            self.config.small_step_radius = recommended_radius
        
        t0 = time.time()
        tree = KDTree(self.skin_verts_orig)
        print(f"[LBS]   KDTree built in {time.time()-t0:.2f}s")
        
        t1 = time.time()
        neighbors = tree.query_ball_point(self.skin_verts_orig, r=self.config.small_step_radius)
        print(f"[LBS]   query_ball_point completed in {time.time()-t1:.2f}s")
        
        # Check neighbor statistics
        neighbor_counts = [len(n_list) for n_list in neighbors]
        avg_neighbors = np.mean(neighbor_counts)
        max_neighbors = np.max(neighbor_counts)
        print(f"[LBS]   Avg neighbors: {avg_neighbors:.1f}, Max: {max_neighbors}")
        
        t2 = time.time()
        adj_list = [[] for _ in range(n)]
        for i, idxs in enumerate(neighbors):
            if i % 1000 == 0:
                print(f"[LBS]   Building adjacency list: {i}/{n} ({100*i/n:.1f}%)")
            for neighbor in idxs:
                if i == neighbor:
                    continue
                w = np.linalg.norm(self.skin_verts_orig[i] - self.skin_verts_orig[neighbor])
                adj_list[i].append((neighbor, w))
        print(f"[LBS]   Adjacency list built in {time.time()-t2:.2f}s")

        
        # Compute geodesic distances
        print("[LBS] Computing geodesic distances...")
        t3 = time.time()
        dist_jaw = self._dijkstra(n, adj_list, self.jaw_seeds)
        print(f"[LBS]   Dijkstra (jaw) completed in {time.time()-t3:.2f}s")
        
        t4 = time.time()
        dist_skull = self._dijkstra(n, adj_list, self.skull_seeds)
        print(f"[LBS]   Dijkstra (skull) completed in {time.time()-t4:.2f}s")
        
        # Compute weights with exponential decay
        print("[LBS] Computing weight decay...")
        t5 = time.time()
        w_jaw = np.exp(-dist_jaw / self.config.weight_damping)
        w_skull = np.exp(-dist_skull / self.config.weight_damping)
        print(f"[LBS]   Weight decay computed in {time.time()-t5:.2f}s")
        
        # Soft threshold with smooth transition
        soft_threshold = 0.001
        smoothness = 0.01
        
        def smooth_cutoff(weights, threshold, smooth_width):
            transition = 0.5 * (np.tanh((weights - threshold) / smooth_width) + 1)
            return weights * transition
        
        w_jaw = smooth_cutoff(w_jaw, soft_threshold, smoothness)
        w_skull = smooth_cutoff(w_skull, soft_threshold, smoothness)
        
        # Apply Laplacian smoothing
        print("[LBS] Applying Laplacian smoothing...")
        t6 = time.time()
        w_jaw = self._laplacian_smooth(
            w_jaw, adj_list, 
            iterations=self.config.laplacian_iterations,
            alpha=self.config.laplacian_alpha
        )
        w_skull = self._laplacian_smooth(
            w_skull, adj_list,
            iterations=self.config.laplacian_iterations,
            alpha=self.config.laplacian_alpha
        )
        print(f"[LBS]   Laplacian smoothing completed in {time.time()-t6:.2f}s")
        
        # Stack weights: [jaw (moving), skull (static)]
        # Note: w_jaw + w_skull can be < 1.0, rest stays in place
        return np.stack([w_jaw, w_skull], axis=1)

    
    def _laplacian_smooth(
        self, 
        weights: np.ndarray, 
        adj_list: list, 
        iterations: int = 2,
        alpha: float = 0.3
    ) -> np.ndarray:
        """Apply Laplacian smoothing to weight field"""
        smoothed = weights.copy()
        
        for _ in range(iterations):
            new_weights = smoothed.copy()
            for i, neighbors in enumerate(adj_list):
                if len(neighbors) == 0:
                    continue
                neighbor_indices = [n[0] for n in neighbors]
                neighbor_avg = smoothed[neighbor_indices].mean()
                new_weights[i] = (1 - alpha) * smoothed[i] + alpha * neighbor_avg
            smoothed = new_weights
        
        return smoothed
    
    def apply_transformation(
        self, 
        rotation: Tuple[float, float] = (0, 0),
        translation: np.ndarray = np.zeros(3)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply jaw transformation and compute deformed meshes.
        
        Args:
            rotation: (pitch, yaw) in radians
            translation: (x, y, z) displacement
        
        Returns:
            Tuple of (deformed_jaw, skull, deformed_skin) vertices
        """
        # Construct rotation matrix
        # Original mapping: angle[1]*X + angle[0]*Z
        rot_vec = rotation[1] * np.array([1, 0, 0]) + rotation[0] * np.array([0, 0, 1])
        rot_matrix = Rotation.from_rotvec(rot_vec).as_matrix()
        
        # Transform jaw (rigid body)
        updated_jaw = (rot_matrix @ self.jaw_verts_orig.T).T + translation
        
        # Skull stays static
        updated_skull = self.skull_verts_orig.copy()
        
        # Transform skin using LBS
        w_jaw = self.weights[:, 0][:, None]
        w_skull = self.weights[:, 1][:, None]
        w_rest = 1.0 - w_jaw - w_skull
        
        # Deformed components
        skin_jaw_transformed = (rot_matrix @ self.skin_verts_orig.T).T + translation
        skin_skull_transformed = self.skin_verts_orig  # Static
        skin_rest = self.skin_verts_orig  # Keep original
        
        # Blend all components
        updated_skin = (
            w_jaw * skin_jaw_transformed +
            w_skull * skin_skull_transformed +
            w_rest * skin_rest
        )
        
        # Update current state
        self.current_transform['rotation'] = np.array(rotation)
        self.current_transform['translation'] = translation.copy()
        
        return updated_jaw, updated_skull, updated_skin
    
    def get_current_transform(self) -> dict:
        """Get current transformation parameters"""
        return self.current_transform.copy()
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reset to original mesh positions"""
        return self.apply_transformation(
            rotation=(0, 0),
            translation=np.zeros(3)
        )
    
    def to_torch(
        self,
        jaw_verts: np.ndarray,
        skull_verts: np.ndarray, 
        skin_verts: np.ndarray,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert numpy arrays to torch tensors.
        
        Args:
            jaw_verts: Jaw vertices (N, 3)
            skull_verts: Skull vertices (M, 3)
            skin_verts: Skin vertices (K, 3)
            device: Target device
        
        Returns:
            Tuple of torch tensors
        """
        return (
            torch.from_numpy(jaw_verts).float().to(device),
            torch.from_numpy(skull_verts).float().to(device),
            torch.from_numpy(skin_verts).float().to(device)
        )


def load_mesh_from_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh from OBJ/PLY file.
    
    Args:
        filepath: Path to mesh file
    
    Returns:
        Tuple of (vertices, faces)
    """
    import trimesh
    
    mesh = trimesh.load(filepath)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    
    return vertices, faces


# Example usage
if __name__ == "__main__":
    # Test with dummy data
    print("Testing LBS Controller...")
    
    # Create dummy meshes
    skin_verts = np.random.randn(1000, 3) * 10
    skull_verts = np.random.randn(200, 3) * 10 + np.array([0, 5, 0])
    jaw_verts = np.random.randn(200, 3) * 10 + np.array([0, -5, 0])
    
    # Initialize controller
    config = LBSConfig(weight_damping=20.0, proximity_threshold=12.0)
    controller = LBSController(skin_verts, skull_verts, jaw_verts, config)
    
    # Apply transformation
    rotation = (np.radians(-10), np.radians(5))  # Pitch, Yaw
    translation = np.array([0, 0, -5])
    
    jaw_def, skull_def, skin_def = controller.apply_transformation(rotation, translation)
    
    print(f"Original skin vertices: {skin_verts.shape}")
    print(f"Deformed skin vertices: {skin_def.shape}")
    print("Test completed successfully!")
