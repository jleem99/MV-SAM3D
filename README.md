# MV-SAM3D: Adaptive Multi-View 3D Reconstruction

An enhanced multi-view extension for [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), featuring **adaptive fusion** strategies for improved 3D reconstruction quality from multiple viewpoints.

> 🔗 **Basic Version**: For a simpler averaging-based approach, check out our [basic multi-view fork](https://github.com/YOUR_USERNAME/sam3d-objects-fork).

## 🔥 Highlights

- **Adaptive Multi-View Fusion**: Unlike simple averaging, we employ a confidence-aware fusion mechanism that automatically weighs contributions from different views based on their reliability.
  
- **Per-Latent Weighting**: Our method operates at the latent level, enabling fine-grained control over how information from different views is combined for each spatial location.

- **Improved Reconstruction Quality**: Better handling of occluded regions and view-dependent artifacts through intelligent fusion.

- **Easy to Use**: Drop-in replacement with minimal changes to the original pipeline.

## 📢 Research in Progress

This is an **active research project**. We are continuously exploring new fusion strategies and will release updates as our research progresses.

## Results Comparison

<table>
<tr>
  <td align="center" width="33%"><b>Single-View (View 3)</b></td>
  <td align="center" width="33%"><b>Single-View (View 6)</b></td>
  <td align="center" width="33%"><b>Multi-View (Adaptive Fusion)</b></td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/3.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Image</b><br>
    <img src="data/example/images/6.png" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>Input Images</b><br>
    <table width="100%" cellpadding="2" cellspacing="2">
    <tr>
      <td align="center"><img src="data/example/images/1.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/2.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/3.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/4.png" width="80px"/></td>
    </tr>
    <tr>
      <td align="center"><img src="data/example/images/5.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/6.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/7.png" width="80px"/></td>
      <td align="center"><img src="data/example/images/8.png" width="80px"/></td>
    </tr>
    </table>
  </td>
</tr>
<tr>
  <td align="center" colspan="3">
    <b>↓ 3D Reconstruction ↓</b>
  </td>
</tr>
<tr>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <img src="data/example/visualization_results/view3_cropped.gif" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <img src="data/example/visualization_results/view6_cropped.gif" width="100%" style="max-width: 300px;"/>
  </td>
  <td align="center" width="33%" style="padding: 5px;">
    <b>3D Result</b><br>
    <img src="data/example/visualization_results/all_views_cropped.gif" width="100%" style="max-width: 300px;"/>
  </td>
</tr>
</table>

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/MV-SAM3D.git
cd MV-SAM3D
pip install -r requirements.txt
```

For model checkpoints, please follow the setup instructions in [doc/setup.md](doc/setup.md).

## Quick Start

### Basic Multi-View Reconstruction (Simple Average)

```bash
python run_inference.py --input_path ./data --mask_prompt object_name
```

### Adaptive Multi-View Reconstruction (Recommended)

```bash
python run_inference_weighted.py \
    --input_path ./data \
    --mask_prompt object_name \
    --image_names 0,1,2,3,4,5
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_path` | Path to input directory | Required |
| `--mask_prompt` | Mask folder name | None |
| `--image_names` | Image names (comma-separated) | All images |
| `--entropy_alpha` | Fusion sharpness (higher = more selective) | 5.0 |
| `--no_weighting` | Disable adaptive fusion | False |
| `--visualize_weights` | Visualize fusion weights | False |

### Data Structure

```
input_path/
├── images/
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── object_name/  # mask folder
    ├── 0.png
    ├── 1.png
    └── ...
```

**Mask Format**: RGBA format where alpha channel stores mask (alpha=255 for object, alpha=0 for background).

## Roadmap

- [x] Basic multi-view fusion (averaging)
- [x] Adaptive confidence-based fusion
- [ ] More sophisticated confidence estimation
- [ ] Support for additional input modalities
- [ ] Quantitative evaluation on benchmarks

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{mv-sam3d,
  author = {Your Name},
  title = {MV-SAM3D: Adaptive Multi-View 3D Reconstruction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/MV-SAM3D}
}
```

## Acknowledgments

This project builds upon [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) by Meta. We thank the original authors for their excellent work.

## License

This project is licensed under the [SAM License](./LICENSE).
