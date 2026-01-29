# Affinity Graph Ordering (Omnirefactor vs. William)

This note summarizes the ordering differences between Omnirefactor’s affinity-graph layout and William’s bitfield layout, and suggests how to interoperate.

## Omnirefactor ordering (internal)

- **Purpose:** Simple, stable ordering for direct array use (not bit-packed).
- **Construction:** Uses `get_steps(dim)` → cartesian product of `[-1, 0, 1]` per axis.
- **Ordering:** **Lexicographic** order (row-major) by axis.
- **Center index:** Always `idx = (3**dim)//2`.
- **Opposites:** Step at index `i` has its opposite at index `-(i+1)`.

### 2D example (y, x)

Index → step:

```
0: [-1, -1]
1: [-1,  0]
2: [-1,  1]
3: [ 0, -1]
4: [ 0,  0]  (center)
5: [ 0,  1]
6: [ 1, -1]
7: [ 1,  0]
8: [ 1,  1]
```

The affinity graph uses these rows in the exact order above (including the center row, which is later zeroed).

---

## William’s ordering (bitfield)

William’s order is designed for **packed bitfields** where **face connections live in the least-significant bits**, so 4-/6-connectivity can be expressed by truncating low bits.

### 2D 8-connected example (bit positions 1..8)

```
8      7      6      5      4      3      2      1
-x-y   x-y   -x+y   x+y    -y     +y     -x     +x
```

Key property: **corners → edges → faces**, with **faces in lowest bits**.

---

## Compatibility: mapping between orders (2D)

William order (excluding center) vs. Omnirefactor index:

| William | Direction | Omnirefactor index |
|--------:|-----------|--------------------:|
| 0 | -x -y | 0 |
| 1 | +x -y | 2 |
| 2 | -x +y | 6 |
| 3 | +x +y | 8 |
| 4 | -y    | 1 |
| 5 | +y    | 7 |
| 6 | -x    | 3 |
| 7 | +x    | 5 |

So William order is simply:

```
steps[[0, 2, 6, 8, 1, 7, 3, 5]]
```

---

## Recommendation

- **Keep Omnirefactor’s ordering for internal computation** (simple, stable, symmetric).
- **Add a lightweight reorder function** when interchanging with bitfield formats.

This avoids breaking existing code while preserving William’s compact bitfield layout.

---

## References in code

- Ordering is defined by `utils/neighbor.py:get_steps()` and `kernel_setup(dim)`.
- `steps` is a cartesian product of `[-1,0,1]` with lexicographic ordering.
- The center index is always `3**dim // 2`.
