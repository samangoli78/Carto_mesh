import numpy as np
import math
# ---------- VTK writers ----------

def write_mesh_with_scalar_vtk(verts, faces, scalars, fname, scalar_name="voltage"):
    """
    Legacy ASCII VTK PolyData with triangles and a per-vertex scalar.
    verts: (N,3) float
    faces: (M,3) int
    scalars: (N,) float
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    scalars = np.asarray(scalars, dtype=np.float64)
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert scalars.shape[0] == verts.shape[0]

    with open(fname, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{scalar_name} on mesh\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(verts)} float\n")
        for p in verts:
            f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        # POLYGONS: each triangle written as "3 i j k"
        m = faces.shape[0]
        f.write(f"POLYGONS {m} {m*4}\n")
        for tri in faces:
            f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")
        # Scalars
        f.write(f"POINT_DATA {len(verts)}\n")
        f.write(f"SCALARS {scalar_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in scalars:
            # NaN-safe: VTK accepts "nan"
            f.write(f"{v:.9g}\n")

def write_points_with_scalars_vtk(points, vals, ids=None, radii=None,
                                  fname="electrodes.vtk",
                                  val_name="V", id_name="id", radius_name="radius"):
    """
    Legacy ASCII VTK PolyData with points only (VERTICES), plus per-point scalars.
    Use ParaView's Glyph filter (Sphere) to visualize as spheres; scale by 'radius' or 'V'.
    points: (K,3) float
    vals:   (K,)  float
    ids:    (K,)  (optional)
    radii:  (K,)  (optional) physical radius for glyph scaling
    """
    points = np.asarray(points, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    K = points.shape[0]
    assert points.ndim == 2 and points.shape[1] == 3
    assert vals.shape[0] == K
    if ids is None:
        ids = np.arange(K, dtype=np.int64)
    else:
        ids = np.asarray(ids)
        assert ids.shape[0] == K
    if radii is None:
        # reasonable default radius based on point spacing; tweak later in ParaView
        # here just a constant (e.g., 1.5 units); you can pass your own array.
        radii = np.full(K, 1.5, dtype=np.float64)
    else:
        radii = np.asarray(radii, dtype=np.float64)
        assert radii.shape[0] == K

    with open(fname, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Projected electrodes with values\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {K} float\n")
        for p in points:
            f.write(f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        # One vertex cell per point: "1 i"
        f.write(f"VERTICES {K} {K*2}\n")
        for i in range(K):
            f.write(f"1 {i}\n")

        f.write(f"POINT_DATA {K}\n")
        # primary scalar
        f.write(f"SCALARS {val_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in vals:
            f.write(f"{v:.9g}\n")
        # id array (as scalar integers)
        if ids is not None:
            f.write(f"SCALARS {id_name} int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in ids:
                f.write(f"{int(i)}\n")
        # radius for glyph scaling
        if radii is not None:
            f.write(f"SCALARS {radius_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for r in radii:
                f.write(f"{float(r):.9g}\n")
