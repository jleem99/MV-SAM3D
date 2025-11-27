"""
SAM 3D Objects Weighted Inference Script

This script extends the standard inference with attention-based weighted fusion.
Instead of simple averaging across views, it uses attention entropy to determine
per-latent fusion weights.

Key features:
    - Per-latent weighting based on attention entropy
    - Configurable weighting parameters (alpha, layer, step)
    - Optional visualization of weights and entropy
    - Extensible architecture for adding new confidence factors
    - GLB to video rendering (360-degree rotation)

Usage:
    # Basic weighted inference (default: use entropy weighting)
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3
    
    # Disable weighting (simple average, like original)
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 --no_weighting
    
    # With visualization
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 --visualize_weights
    
    # Custom weighting parameters
    python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
        --entropy_alpha 3.0 --attention_layer 6 --attention_step 0
"""
import sys
import argparse
from pathlib import Path
import subprocess
import tempfile
import shutil
from typing import List, Optional
from datetime import datetime
import torch
from loguru import logger

# 导入推理代码
sys.path.append("notebook")
from inference import Inference
from load_images_and_masks import load_images_and_masks_from_path

from sam3d_objects.utils.cross_attention_logger import CrossAttentionLogger
from sam3d_objects.utils.latent_weighting import WeightingConfig, LatentWeightManager


def parse_image_names(image_names_str: Optional[str]) -> Optional[List[str]]:
    """Parse image names string."""
    if image_names_str is None or image_names_str == "":
        return None
    names = [x.strip() for x in image_names_str.split(",") if x.strip()]
    return names if names else None


def parse_attention_layers(layers_str: Optional[str]) -> Optional[List[int]]:
    """Parse attention layer indices from CLI string."""
    if layers_str is None:
        return None
    tokens = [token.strip() for token in layers_str.split(",") if token.strip()]
    if not tokens:
        return None
    indices: List[int] = []
    for token in tokens:
        try:
            indices.append(int(token))
        except ValueError as exc:
            raise ValueError(f"Invalid attention layer index: {token}") from exc
    return indices


def get_output_dir(
    input_path: Path, 
    mask_prompt: Optional[str] = None, 
    image_names: Optional[List[str]] = None,
    is_single_view: bool = False,
    use_weighting: bool = True,
    entropy_alpha: float = 30.0,
) -> Path:
    """Create output directory based on input path and parameters."""
    visualization_dir = Path("visualization")
    visualization_dir.mkdir(exist_ok=True)
    
    if mask_prompt:
        dir_name = mask_prompt
    else:
        dir_name = input_path.name if input_path.is_dir() else input_path.parent.name
    
    # Add weighting suffix with alpha value
    if use_weighting:
        # Format alpha: remove trailing zeros, e.g., 5.0 -> "5", 5.5 -> "5.5"
        alpha_str = f"{entropy_alpha:g}"
        suffix = f"_weighted_a{alpha_str}"
    else:
        suffix = "_avg"
    
    if is_single_view:
        if image_names and len(image_names) == 1:
            safe_name = image_names[0].replace("/", "_").replace("\\", "_")
            dir_name = f"{dir_name}_{safe_name}{suffix}"
        else:
            dir_name = f"{dir_name}_single{suffix}"
    elif image_names:
        if len(image_names) == 1:
            safe_name = image_names[0].replace("/", "_").replace("\\", "_")
            dir_name = f"{dir_name}_{safe_name}{suffix}"
        else:
            safe_names = [name.replace("/", "_").replace("\\", "_") for name in image_names]
            dir_name = f"{dir_name}_{'_'.join(safe_names[:3])}{suffix}"
            if len(safe_names) > 3:
                dir_name += f"_and_{len(safe_names)-3}_more"
    else:
        dir_name = f"{dir_name}_multiview{suffix}"
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{dir_name}_{timestamp}"
    
    output_dir = visualization_dir / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    return output_dir


def run_weighted_inference(
    input_path: Path,
    mask_prompt: Optional[str] = None,
    image_names: Optional[List[str]] = None,
    seed: int = 42,
    stage1_steps: int = 50,
    stage2_steps: int = 25,
    decode_formats: List[str] = None,
    model_tag: str = "hf",
    # Weighting parameters
    use_weighting: bool = True,
    entropy_alpha: float = 30.0,
    attention_layer: int = 6,
    attention_step: int = 0,
    min_weight: float = 0.01,
    # Visualization
    visualize_weights: bool = False,
    save_attention: bool = False,
    attention_layers_to_save: Optional[List[int]] = None,
    save_coords: bool = True,  # Default True for weighted inference
):
    """
    Run weighted inference.
    
    Args:
        input_path: Input path
        mask_prompt: Mask folder name
        image_names: List of image names
        seed: Random seed
        stage1_steps: Stage 1 inference steps
        stage2_steps: Stage 2 inference steps
        decode_formats: List of decode formats
        model_tag: Model tag
        use_weighting: Whether to use entropy-based weighting (default True)
        entropy_alpha: Gibbs temperature for entropy weighting
        attention_layer: Which layer to use for weight computation
        attention_step: Which step to use for weight computation
        min_weight: Minimum weight to prevent complete zeroing
        visualize_weights: Whether to save weight visualizations
        save_attention: Whether to save attention weights
        attention_layers_to_save: Which layers to save attention for
        save_coords: Whether to save 3D coordinates
    """
    config_path = f"checkpoints/{model_tag}/pipeline.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    logger.info(f"Loading model: {config_path}")
    inference = Inference(config_path, compile=False)
    
    if hasattr(inference._pipeline, 'rendering_engine'):
        if inference._pipeline.rendering_engine != "pytorch3d":
            logger.warning(f"Rendering engine is set to {inference._pipeline.rendering_engine}, changing to pytorch3d")
            inference._pipeline.rendering_engine = "pytorch3d"
    
    logger.info(f"Loading data: {input_path}")
    if mask_prompt:
        logger.info(f"Mask prompt: {mask_prompt}")
    
    view_images, view_masks = load_images_and_masks_from_path(
        input_path=input_path,
        mask_prompt=mask_prompt,
        image_names=image_names,
    )
    
    num_views = len(view_images)
    logger.info(f"Successfully loaded {num_views} views")
    
    is_single_view = num_views == 1
    
    if is_single_view:
        logger.warning("Single view detected - weighting is not applicable, using standard inference")
        use_weighting = False
    
    output_dir = get_output_dir(input_path, mask_prompt, image_names, is_single_view, use_weighting, entropy_alpha)
    
    # Setup logging
    log_file = output_dir / "inference.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
    )
    
    decode_formats = decode_formats or ["gaussian", "mesh"]
    
    # Setup weighting config
    weighting_config = WeightingConfig(
        use_entropy=use_weighting,
        entropy_alpha=entropy_alpha,
        attention_layer=attention_layer,
        attention_step=attention_step,
        min_weight=min_weight,
    )
    
    logger.info(f"Weighting config: use_weighting={use_weighting}, alpha={entropy_alpha}, "
                f"layer={attention_layer}, step={attention_step}, min_weight={min_weight}")
    
    # Setup attention logger (only if explicitly requested for analysis)
    attention_logger: Optional[CrossAttentionLogger] = None
    if save_attention:
        # Only save attention when explicitly requested (for analysis purposes)
        layers_to_hook = attention_layers_to_save or [attention_layer]
        if attention_layer not in layers_to_hook:
            layers_to_hook.append(attention_layer)
        
        attention_dir = output_dir / "attention"
        attention_logger = CrossAttentionLogger(
            attention_dir,
            enabled_stages=["slat"],
            layer_indices=layers_to_hook,
            save_coords=save_coords,
        )
        attention_logger.attach_to_pipeline(inference._pipeline)
        logger.info(f"Cross-attention logging enabled → layers={layers_to_hook}, save_coords={save_coords}")
    
    # Note: Weighting uses in-memory AttentionCollector, not CrossAttentionLogger
    # The attention for weight computation is collected directly during warmup pass
    
    # Run inference
    if is_single_view:
        logger.info("Single-view inference mode")
        image = view_images[0]
        mask = view_masks[0] if view_masks else None
        result = inference._pipeline.run(
            image,
            mask,
            seed=seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            attention_logger=attention_logger,
        )
        weight_manager = None
    else:
        logger.info(f"Multi-view inference mode ({'weighted' if use_weighting else 'average'})")
        result = inference._pipeline.run_multi_view(
            view_images=view_images,
            view_masks=view_masks,
            seed=seed,
            mode="multidiffusion",
            stage1_inference_steps=stage1_steps,
            stage2_inference_steps=stage2_steps,
            decode_formats=decode_formats,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            use_vertex_color=True,
            attention_logger=attention_logger,
            # Pass weighting config for weighted fusion
            weighting_config=weighting_config if use_weighting else None,
        )
        weight_manager = result.get("weight_manager")
    
    # Save results
    saved_files = []
    
    print(f"\n{'='*60}")
    print(f"Inference completed!")
    print(f"Mode: {'Weighted' if use_weighting else 'Average'} fusion")
    print(f"Generated coordinates: {result['coords'].shape[0] if 'coords' in result else 'N/A'}")
    print(f"{'='*60}")
    
    glb_path = None
    if 'glb' in result and result['glb'] is not None:
        glb_path = output_dir / "result.glb"
        result['glb'].export(str(glb_path))
        saved_files.append("result.glb")
        print(f"✓ GLB file saved to: {glb_path}")
    
    if 'gs' in result:
        output_path = output_dir / "result.ply"
        result['gs'].save_ply(str(output_path))
        saved_files.append("result.ply")
        print(f"✓ Gaussian Splatting (PLY) saved to: {output_path}")
    elif 'gaussian' in result:
        if isinstance(result['gaussian'], list) and len(result['gaussian']) > 0:
            output_path = output_dir / "result.ply"
            result['gaussian'][0].save_ply(str(output_path))
            saved_files.append("result.ply")
            print(f"✓ Gaussian Splatting (PLY) saved to: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"All output files saved to: {output_dir}")
    print(f"Saved files: {', '.join(saved_files)}")
    print(f"{'='*60}")
    
    if attention_logger is not None:
        attention_logger.close()
    
    # Save weighting analysis if enabled
    if weight_manager is not None and visualize_weights:
        logger.info("Saving weight visualizations...")
        
        # Save weights and visualizations
        weights_dir = output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_data = weight_manager.get_analysis_data()
        weights_downsampled = analysis_data.get("weights", {})  # 降采样维度的权重
        weights_expanded = analysis_data.get("expanded_weights", {})  # 扩展后的权重
        entropy_per_view = analysis_data.get("entropy_per_view", {})
        original_coords = analysis_data.get("original_coords")  # 原始 coords
        downsampled_coords = analysis_data.get("downsampled_coords")  # 降采样 coords
        downsample_idx = analysis_data.get("downsample_idx")  # idx 映射
        
        # Log dimension info
        if weights_downsampled:
            sample_w = list(weights_downsampled.values())[0]
            logger.info(f"Downsampled weights dimension: {sample_w.shape[0]}")
        if weights_expanded:
            sample_w = list(weights_expanded.values())[0]
            logger.info(f"Expanded weights dimension: {sample_w.shape[0]}")
        if original_coords is not None:
            logger.info(f"Original coords shape: {original_coords.shape}")
        if downsampled_coords is not None:
            logger.info(f"Downsampled coords shape: {downsampled_coords.shape}")
        
        # Save weights as .pt file
        torch.save({
            "weights_downsampled": {k: v.cpu() for k, v in weights_downsampled.items()} if weights_downsampled else {},
            "weights_expanded": {k: v.cpu() for k, v in weights_expanded.items()} if weights_expanded else {},
            "entropy": {k: v.cpu() for k, v in entropy_per_view.items()} if entropy_per_view else {},
            "config": {
                "entropy_alpha": weighting_config.entropy_alpha,
                "attention_layer": weighting_config.attention_layer,
                "attention_step": weighting_config.attention_step,
            },
            "original_coords": original_coords.cpu() if original_coords is not None else None,
            "downsampled_coords": downsampled_coords.cpu() if downsampled_coords is not None else None,
            "downsample_idx": downsample_idx.cpu() if downsample_idx is not None else None,
        }, weights_dir / "fusion_weights.pt")
        
        logger.info(f"Saved fusion weights to {weights_dir / 'fusion_weights.pt'}")
        
        # ============ Weight Analysis ============
        analysis_log = weights_dir / "weight_analysis.log"
        with open(analysis_log, "w") as f:
            def log_analysis(msg):
                f.write(msg + "\n")
                logger.info(msg)
            
            log_analysis("=" * 60)
            log_analysis("Weight Analysis Report")
            log_analysis("=" * 60)
            log_analysis(f"Number of views: {len(weights_downsampled)}")
            log_analysis(f"Entropy alpha: {weighting_config.entropy_alpha}")
            log_analysis(f"Attention layer: {weighting_config.attention_layer}")
            log_analysis(f"Attention step: {weighting_config.attention_step}")
            
            # Entropy analysis
            if entropy_per_view:
                log_analysis("\n--- Entropy Analysis ---")
                entropy_values = []
                for view_idx, e in sorted(entropy_per_view.items()):
                    log_analysis(
                        f"  View {view_idx}: min={e.min():.4f}, max={e.max():.4f}, "
                        f"mean={e.mean():.4f}, std={e.std():.4f}"
                    )
                    entropy_values.append(e)
                
                # Cross-view entropy difference
                if len(entropy_values) > 1:
                    entropy_stack = torch.stack(entropy_values, dim=0)
                    view_std = entropy_stack.std(dim=0)
                    log_analysis(f"\n  Cross-view entropy std (per latent):")
                    log_analysis(f"    min={view_std.min():.4f}, max={view_std.max():.4f}, mean={view_std.mean():.4f}")
            
            # Weight analysis
            log_analysis("\n--- Weight Analysis (Downsampled) ---")
            for view_idx, w in sorted(weights_downsampled.items()):
                log_analysis(
                    f"  View {view_idx}: min={w.min():.6f}, max={w.max():.6f}, "
                    f"mean={w.mean():.6f}, std={w.std():.6f}"
                )
            
            # Check weight sum
            views = sorted(weights_downsampled.keys())
            weight_sum = sum(weights_downsampled[v] for v in views)
            log_analysis(f"\n  Weight sum: min={weight_sum.min():.4f}, max={weight_sum.max():.4f}")
            
            # Cross-view weight difference
            weight_stack = torch.stack([weights_downsampled[v] for v in views], dim=0)
            view_std = weight_stack.std(dim=0)
            log_analysis(f"\n  Cross-view weight std (per latent):")
            log_analysis(f"    min={view_std.min():.6f}, max={view_std.max():.6f}, mean={view_std.mean():.6f}")
            
            # Find latents with most weight variation
            top_k = 5
            top_indices = torch.argsort(view_std, descending=True)[:top_k]
            log_analysis(f"\n  Top {top_k} latents with most weight variation:")
            for idx in top_indices:
                log_analysis(f"    Latent {idx.item()}: std={view_std[idx]:.4f}")
                for v in views:
                    log_analysis(f"      View {v}: {weights_downsampled[v][idx]:.4f}")
            
            log_analysis("\n" + "=" * 60)
        
        logger.info(f"Saved weight analysis to {analysis_log}")
        
        # Generate visualizations
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Weight distribution histogram (downsampled)
            if weights_downsampled:
                fig, axes = plt.subplots(1, len(weights_downsampled), figsize=(4 * len(weights_downsampled), 4))
                if len(weights_downsampled) == 1:
                    axes = [axes]
                
                for ax, (view_idx, w) in zip(axes, sorted(weights_downsampled.items())):
                    w_np = w.cpu().numpy()
                    ax.hist(w_np, bins=50, alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Weight')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx} (downsampled)\nmean={w_np.mean():.4f}, std={w_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'weight_distribution_downsampled.png', dpi=150)
                plt.close()
                logger.info("Saved downsampled weight distribution plot")
            
            # Weight distribution histogram (expanded)
            if weights_expanded:
                fig, axes = plt.subplots(1, len(weights_expanded), figsize=(4 * len(weights_expanded), 4))
                if len(weights_expanded) == 1:
                    axes = [axes]
                
                for ax, (view_idx, w) in zip(axes, sorted(weights_expanded.items())):
                    w_np = w.cpu().numpy()
                    ax.hist(w_np, bins=50, alpha=0.7, edgecolor='black', color='green')
                    ax.set_xlabel('Weight')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx} (expanded)\nmean={w_np.mean():.4f}, std={w_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'weight_distribution_expanded.png', dpi=150)
                plt.close()
                logger.info("Saved expanded weight distribution plot")
            
            # Entropy distribution histogram
            if entropy_per_view:
                fig, axes = plt.subplots(1, len(entropy_per_view), figsize=(4 * len(entropy_per_view), 4))
                if len(entropy_per_view) == 1:
                    axes = [axes]
                
                for ax, (view_idx, e) in zip(axes, sorted(entropy_per_view.items())):
                    e_np = e.cpu().numpy()
                    ax.hist(e_np, bins=50, alpha=0.7, edgecolor='black', color='orange')
                    ax.set_xlabel('Entropy')
                    ax.set_ylabel('Count')
                    ax.set_title(f'View {view_idx}\nmean={e_np.mean():.4f}, std={e_np.std():.4f}')
                
                plt.tight_layout()
                plt.savefig(weights_dir / 'entropy_distribution.png', dpi=150)
                plt.close()
                logger.info("Saved entropy distribution plot")
            
            # 3D visualization with DOWNSAMPLED coords (where attention is computed)
            if downsampled_coords is not None and weights_downsampled:
                coords_np = downsampled_coords.cpu().numpy()
                x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
                
                # Normalize coordinates
                x = (x - x.min()) / (x.max() - x.min() + 1e-6)
                y = (y - y.min()) / (y.max() - y.min() + 1e-6)
                z = (z - z.min()) / (z.max() - z.min() + 1e-6)
                
                for view_idx, w in sorted(weights_downsampled.items()):
                    w_np = w.cpu().numpy()
                    
                    # Robust normalization
                    vmin, vmax = np.percentile(w_np, [2, 98])
                    w_norm = np.clip((w_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    
                    order = np.argsort(z)
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        x[order], y[order], z[order],
                        c=w_norm[order],
                        cmap='viridis',
                        s=2,
                        alpha=0.6,
                    )
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'View {view_idx} Weight (Downsampled, {len(w_np)} points)')
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.6, label='Weight')
                    plt.savefig(weights_dir / f'weight_3d_downsampled_view{view_idx:02d}.png', dpi=150)
                    plt.close()
                
                logger.info("Saved 3D weight visualizations (downsampled)")
            
            # 3D visualization with ORIGINAL coords (expanded weights)
            if original_coords is not None and weights_expanded:
                coords_np = original_coords.cpu().numpy()
                x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
                
                # Normalize coordinates
                x = (x - x.min()) / (x.max() - x.min() + 1e-6)
                y = (y - y.min()) / (y.max() - y.min() + 1e-6)
                z = (z - z.min()) / (z.max() - z.min() + 1e-6)
                
                for view_idx, w in sorted(weights_expanded.items()):
                    w_np = w.cpu().numpy()
                    
                    # Robust normalization
                    vmin, vmax = np.percentile(w_np, [2, 98])
                    w_norm = np.clip((w_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    
                    order = np.argsort(z)
                    
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        x[order], y[order], z[order],
                        c=w_norm[order],
                        cmap='viridis',
                        s=0.5,  # 更小的点，因为点更多
                        alpha=0.4,
                    )
                    
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'View {view_idx} Weight (Expanded, {len(w_np)} points)')
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.6, label='Weight')
                    plt.savefig(weights_dir / f'weight_3d_expanded_view{view_idx:02d}.png', dpi=150)
                    plt.close()
                
                logger.info("Saved 3D weight visualizations (expanded)")
                
        except ImportError as e:
            logger.warning(f"Could not generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects Weighted Inference - Per-latent weighted multi-view fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic weighted inference
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3
  
  # Disable weighting (simple average)
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 --no_weighting
  
  # With visualization
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 --visualize_weights
  
  # Custom parameters
  python run_inference_weighted.py --input_path ./data --mask_prompt stuffed_toy --image_names 0,1,2,3 \
      --entropy_alpha 3.0 --attention_layer 6
        """
    )
    
    # Input/Output
    parser.add_argument("--input_path", type=str, required=True, help="Input path")
    parser.add_argument("--mask_prompt", type=str, default=None, help="Mask folder name")
    parser.add_argument("--image_names", type=str, default=None, help="Image names (comma-separated)")
    parser.add_argument("--model_tag", type=str, default="hf", help="Model tag")
    
    # Inference parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stage1_steps", type=int, default=50, help="Stage 1 steps")
    parser.add_argument("--stage2_steps", type=int, default=25, help="Stage 2 steps")
    parser.add_argument("--decode_formats", type=str, default="gaussian,mesh", help="Decode formats")
    
    # Weighting parameters
    parser.add_argument("--no_weighting", action="store_true", 
                        help="Disable entropy-based weighting (use simple average)")
    parser.add_argument("--entropy_alpha", type=float, default=30.0,
                        help="Gibbs temperature for entropy weighting (higher = more contrast)")
    parser.add_argument("--attention_layer", type=int, default=6,
                        help="Which attention layer to use for weight computation")
    parser.add_argument("--attention_step", type=int, default=0,
                        help="Which diffusion step to use for weight computation")
    parser.add_argument("--min_weight", type=float, default=0.001,
                        help="Minimum weight to prevent complete zeroing")
    
    # Visualization
    parser.add_argument("--visualize_weights", action="store_true",
                        help="Save weight and entropy visualizations")
    parser.add_argument("--save_attention", action="store_true",
                        help="Save all attention weights (for analysis)")
    parser.add_argument("--attention_layers", type=str, default=None,
                        help="Which layers to save attention for (comma-separated)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    image_names = parse_image_names(args.image_names)
    decode_formats = [fmt.strip() for fmt in args.decode_formats.split(",") if fmt.strip()]
    
    try:
        run_weighted_inference(
            input_path=input_path,
            mask_prompt=args.mask_prompt,
            image_names=image_names,
            seed=args.seed,
            stage1_steps=args.stage1_steps,
            stage2_steps=args.stage2_steps,
            decode_formats=decode_formats,
            model_tag=args.model_tag,
            use_weighting=not args.no_weighting,
            entropy_alpha=args.entropy_alpha,
            attention_layer=args.attention_layer,
            attention_step=args.attention_step,
            min_weight=args.min_weight,
            visualize_weights=args.visualize_weights,
            save_attention=args.save_attention,
            attention_layers_to_save=parse_attention_layers(args.attention_layers),
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

