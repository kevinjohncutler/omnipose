# 2D / 3D verification of omnipose flows, GT and loss functions

Test object: two identical cubes (3D) / squares (2D) sharing exactly one face / edge, as one label image (labels 1 and 2). All numbers below come from the *shipped* `omnipose.core.loss.loss` on GT produced by the real `masks_to_flows_batch` + `batch_labels` pipeline. Per-voxel maps are validated to reduce to these scalars to a relative error < 2e-7.

## 1. Ground-truth field correctness

| Check | Expected | dim=2 | dim=3 |
|---|---|---|---|
| distance background fill | -5 | -5.0 | -5.0 |
| max distance, cube 1 | equal to cube 2 | 7.789 | 6.481 |
| max distance, cube 2 | equal to cube 1 | 7.789 | 6.481 |
| dist at contact-adjacent voxel | equals outer-face value | 1.000 | 0.841 |
| dist at outer-face-adjacent voxel | equals contact value | 1.000 | 0.841 |
| flow axis0 at cube1 contact layer | negative (toward cube1) | -5.000 | -6.098 |
| flow axis0 at cube2 contact layer | positive (toward cube2) | +5.000 | +6.098 |
| mean |flow| inside cells | about 5 | 4.460 | 5.164 |
| max |flow| in background | 0 | 0.0e+00 | 0.0e+00 |

The contact face behaves as a true boundary (distance there equals the outer-face distance), so labels do not bleed across it, and the two flow fields point away from each other at the contact plane in both dimensions.

## 2. Per-voxel analytic check (scenario flow set to 0 in cube 2)

Inside cube 2 the prediction is flow = 0, so the per-voxel loss values are predictable in closed form. Max abs difference between the analytic prediction and the shipped per-voxel map:

| Term | Analytic per-voxel value (in cube 2) | max abs diff dim=2 | max abs diff dim=3 |
|---|---|---|---|
| flow_mse | w * sum_d veci_d^2 | 0.00e+00 | 0.00e+00 |
| norm_loss | (0 - \|veci\|)^2 = \|veci\|^2 | 0.00e+00 | 0.00e+00 |
| SSL | w * 1 where dist>0 (cos^2 -> 0) | 0.00e+00 | 0.00e+00 |

## 3. Expected behavior and computed scalar losses

"fire" = the analytic prediction that the term should be nonzero. Every computed value below is nonzero exactly where predicted and zero elsewhere, with matching structure in 2D and 3D.

### dim = 2

| scenario | flow_mse | dist_loss | SSL | norm_loss | bd_loss | lossDC | lossE | lossA |
|---|---|---|---|---|---|---|---|---|
| perfect | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| flow_zero_cube2 | 1.383 | 0 | 0.1368 | 2.767 | 0 | 0.6256 | 0.4095 | 0.1104 |
| flow_flip_cube2 | 5.533 | 0 | 0 | 0 | 0 | 2.502 | 1.574 | 0.0298 |
| dist_offset | 0 | 1.094 | 0 | 0 | 0.0075 | 0 | 0 | 0 |
| dist_bump | 0 | 0.2443 | 0 | 0 | 0.002712 | 0 | 0 | 0 |

### dim = 3

| scenario | flow_mse | dist_loss | SSL | norm_loss | bd_loss | lossDC | lossE | lossA |
|---|---|---|---|---|---|---|---|---|
| perfect | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| flow_zero_cube2 | 0.5519 | 0 | 0.06078 | 1.656 | 0 | 0.5022 | 0.216 | 0.05118 |
| flow_flip_cube2 | 2.208 | 0 | 0 | 0 | 0 | 2.009 | 0.7394 | 0.03224 |
| dist_offset | 0 | 0.4862 | 0 | 0 | 0.0125 | 0 | 0 | 0 |
| dist_bump | 0 | 0.05148 | 0 | 0 | 0.00178 | 0 | 0 | 0 |

## 4. Label equivariance — four cubes (2x2), flow set to 0 in all cells

The four tiles are reflections of one another, so a symmetric perturbation must give an identical per-label contribution to every loss term. This is the direct test that no label (e.g. label 1) is special-cased. "max spread" = (max-min)/mean across labels 1..4.

### dim = 2

| term | label 1 | label 2 | label 3 | label 4 | max spread |
|---|---|---|---|---|---|
| GT max distance | 6.7894 | 6.7894 | 6.7894 | 6.7894 | 0e+00 |
| flow_mse | 3837 | 3837 | 3837 | 3837 | 0e+00 |
| dist_loss | 0 | 0 | 0 | 0 | 0e+00 |
| SSL | 196 | 196 | 196 | 196 | 0e+00 |
| norm_loss | 3837 | 3837 | 3837 | 3837 | 0e+00 |
| bd_loss | 0 | 0 | 0 | 0 | 0e+00 |
| lossDC | 1089 | 1089 | 1089 | 1089 | 1e-07 |
| lossE | 1123 | 1123 | 1123 | 1123 | 1e-07 |
| lossA | 1404 | 1404 | 1404 | 1404 | 0e+00 |

### dim = 3

| term | label 1 | label 2 | label 3 | label 4 | max spread |
|---|---|---|---|---|---|
| GT max distance | 5.6336 | 5.6336 | 5.6336 | 5.6336 | 0e+00 |
| flow_mse | 7.121e+04 | 7.121e+04 | 7.121e+04 | 7.121e+04 | 0e+00 |
| dist_loss | 0 | 0 | 0 | 0 | 0e+00 |
| SSL | 2744 | 2744 | 2744 | 2744 | 0e+00 |
| norm_loss | 7.121e+04 | 7.121e+04 | 7.121e+04 | 7.121e+04 | 0e+00 |
| bd_loss | 0 | 0 | 0 | 0 | 0e+00 |
| lossDC | 2.303e+04 | 2.303e+04 | 2.303e+04 | 2.303e+04 | 0e+00 |
| lossE | 2.739e+04 | 2.739e+04 | 2.739e+04 | 2.739e+04 | 0e+00 |
| lossA | 6.114e+04 | 6.114e+04 | 6.114e+04 | 6.114e+04 | 0e+00 |

Every per-label contribution is equal across labels 1..4 (spread 0, or ~1e-7 for terms whose maps involve floating-point reduction order, i.e. divergence and Euler integration). A label-1-only perturbation lights only the label-1 tile (see four_cubes_{2,3}d.png), confirming label 1 is not ignored.
