import os, numpy as np
from scipy.sparse.linalg import splu

import matplotlib.pyplot as plt
from claculate_area import band_area_graph,mesh_area_valid,band_integral
import heapq
from area_hist import compute_area_histograms_all_scalars, plot_area_histograms

from smooth import *

# ---- needs build_cotan_laplacian_and_mass(V,F) from your smooth.py ----
def _edge_graph_lengths(V, F):
    n = V.shape[0]
    nbr = [[] for _ in range(n)]
    F = F.astype(np.int32, copy=False)
    for a,b,c in F:
        for u,v in ((a,b),(b,c),(c,a)):
            w = float(np.linalg.norm(V[u] - V[v]))
            nbr[u].append((v, w))
            nbr[v].append((u, w))
    return nbr

def _dijkstra_multi(nbr, sources, max_dist=np.inf):
    n = len(nbr)
    dist = np.full(n, np.inf, np.float64)
    h = []
    for s in np.asarray(sources, np.int32):
        if dist[s] > 0.0:
            dist[s] = 0.0
            heapq.heappush(h, (0.0, int(s)))
    while h:
        d,u = heapq.heappop(h)
        if d != dist[u] or d > max_dist:
            continue
        for v,w in nbr[u]:
            nd = d + w
            if nd < dist[v] and nd <= max_dist:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist

def _interp_many_nanaware(
    verts, faces, known_idx, values_dict,
    lam=1e-6,
    radius_factor=10.0,
    add_diag=1e-12,
):
    """
    Harmonic interpolate each field on the FULL mesh, then keep ONLY vertices
    inside a geodesic ball around that field's usable known constraints.
    Outside the ball -> NaN.

    Uses your existing:
      - build_cotan_laplacian_and_mass(V,F)
      - _edge_graph_lengths(V,F)
      - _dijkstra_multi(nbr, sources, max_dist)

    Parameters
    ----------
    verts, faces : mesh
    known_idx    : (k,) vertex ids corresponding to per-point values in values_dict
    values_dict  : dict[name] -> (k,) array (NaN allowed); aligned with known_idx
    lam          : screened-Laplacian weight (A = L + lam*M)
    radius_factor: geodesic radius = radius_factor * mean_edge_length(mesh)
    add_diag     : tiny M-weighted diagonal added to Auu for numerical robustness

    Returns
    -------
    out : dict[name] -> (n_vertices,) float64
        Interpolated field, NaN outside the per-field geodesic ball.
    """
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import splu

    V = np.asarray(verts, np.float64)
    F = np.asarray(faces,  np.int32)
    n = V.shape[0]

    # --- operators ---
    L, M = build_cotan_laplacian_and_mass(V, F)
    A = (L + lam * M).tocsr()

    # --- mean edge length (scale for geodesic radius) ---
    def _mean_edge_length(V, F):
        E = set()
        for a,b,c in F:
            E.add((min(a,b), max(a,b)))
            E.add((min(b,c), max(c,a)))
            E.add((min(a,c), max(a,c)))
        if not E:
            return 1.0
        E = np.array(list(E), dtype=np.int32)
        le = np.linalg.norm(V[E[:,0]] - V[E[:,1]], axis=1)
        m = float(le.mean()) if le.size else 1.0
        return m if np.isfinite(m) and m > 0 else 1.0

    Lmean  = _mean_edge_length(V, F)
    radius = float(radius_factor * Lmean)

    # --- adjacency for geodesic ---
    nbr = _edge_graph_lengths(V, F)

    # --- bookkeeping on known indices (shared) ---
    ki_all = np.asarray(known_idx, np.int64)
    order  = np.argsort(ki_all)
    ki     = ki_all[order]
    uk_all, start = np.unique(ki, return_index=True)

    out = {}

    for name, vals in values_dict.items():
        kv = np.asarray(vals, np.float64)[order]

        # nan-mean across duplicates at each unique known vertex (THIS field only)
        sums   = np.add.reduceat(np.nan_to_num(kv, nan=0.0), start)
        counts = np.add.reduceat(~np.isnan(kv),               start).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            fk = sums / np.maximum(counts, 1.0)
        fk[counts == 0] = np.nan

        # usable constraints for THIS field
        keep = ~np.isnan(fk)
        if not np.any(keep):
            out[name] = np.full(n, np.nan, dtype=np.float64)
            continue

        K  = uk_all[keep]   # known vertex ids for this field
        fK = fk[keep]

        # ---- GLOBAL solve FOR THIS FIELD (unknowns = all verts except K) ----
        mask_known = np.zeros(n, dtype=bool); mask_known[K] = True
        uu = np.where(~mask_known)[0]

        Auu = A[uu][:, uu].tocsc()
        Auk = A[uu][:, K]
        rhs = -Auk @ fK

        if add_diag > 0.0:
            Auu = (Auu + add_diag * diags(np.maximum(M.diagonal()[uu], 1.0))).tocsc()

        fu = splu(Auu).solve(rhs)

        f = np.empty(n, dtype=np.float64)
        f[K]  = fK
        f[uu] = fu

        # ---- per-field geodesic keep-mask using your Dijkstra ----
        dist = _dijkstra_multi(nbr, K, max_dist=radius)
        keep_vertices = np.isfinite(dist) & (dist <= radius)

        # ---- return ONLY inside the geodesic; outside -> NaN ----
        f[~keep_vertices] = np.nan
        out[name] = f

    return out

import numpy as np, vtk

def write_mesh_multi_scalar_vtp_vtk(verts, faces, scalars_dict, fname):
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)

    pts = vtk.vtkPoints()
    pts.SetDataTypeToFloat()
    pts.SetNumberOfPoints(V.shape[0])
    for i, p in enumerate(V):
        pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

    polys = vtk.vtkCellArray()
    for tri in F:
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, int(tri[0]))
        cell.GetPointIds().SetId(1, int(tri[1]))
        cell.GetPointIds().SetId(2, int(tri[2]))
        polys.InsertNextCell(cell)

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(polys)

    pd = poly.GetPointData()
    n = V.shape[0]
    first_set = False
    for name, arr in scalars_dict.items():
        a = np.asarray(arr)
        if a.ndim == 1:
            if a.size != n:
                raise ValueError(f"Scalar '{name}' has length {a.size}, expected {n}")
            va = vtk.vtkFloatArray()
            va.SetName(str(name))
            va.SetNumberOfComponents(1)
            va.SetNumberOfTuples(n)
            for i, v in enumerate(a):
                va.SetValue(i, float(v))
            pd.AddArray(va)
            if not first_set:
                pd.SetScalars(va); first_set = True
        elif a.ndim == 2 and a.shape[0] == n and a.shape[1] in (2,3):
            comps = a.shape[1]
            va = vtk.vtkFloatArray()
            va.SetName(str(name))
            va.SetNumberOfComponents(comps)
            va.SetNumberOfTuples(n)
            for i in range(n):
                tup = [float(x) for x in a[i]]
                if comps == 2: tup = tup + [0.0]
                va.SetTuple(i, tup[:3])
            pd.AddArray(va)
        else:
            raise ValueError(f"Array '{name}' has shape {a.shape}, expected (N,) or (N,3)")

    if not fname.endswith(".vtp"):
        fname = fname.rsplit(".", 1)[0] + ".vtp"
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(fname)
    w.SetInputData(poly)
    w.SetDataModeToAppended()
    w.EncodeAppendedDataOff()
    w.Write()
from differential_ops import (
    per_face_gradient_scalar, gradient_vertices_from_faces, cv_from_grad,
    laplacian_scalar_cotan, divergence_cotan, curl_normal, hessian_quadratic_fit,
    vertex_normals_area_weighted
)
def _pick_lat_key(S):
    # choose which field acts as LAT for derivatives
    for k in ("First", "SR", "Second", "Third"):
        if k in S: return k
    # fallback: any scalar present
    return next(iter(S.keys()))

def _append_derivatives_to_S(S, V, F, L, M, lat_key=None):
    if lat_key is None:
        lat_key = _pick_lat_key(S)
    t = np.asarray(S[lat_key], float)

    # grad (NaN-aware)
    gF = per_face_gradient_scalar(V, F, t)
    gV = gradient_vertices_from_faces(V, F, gF)  # tangent-projected

    # CV
    slowness, CV_mag, CV_vec = cv_from_grad(gV)

    # Laplacian
    lap_t = laplacian_scalar_cotan(L, M, t)

    # Divergence & curl of CV vector
    # project CV_vec to tangent again for safety
    nrm = vertex_normals_area_weighted(V, F)
    vdotn = np.sum(CV_vec*nrm, axis=1)
    CV_vec_t = CV_vec - vdotn[:,None]*nrm

    div_v = divergence_cotan(V, F, CV_vec_t)
    curln = curl_normal(V, F, CV_vec_t)

    # Hessian (2x2 in tangent frame → store components)
    H11, H12, H22 = hessian_quadratic_fit(V, F, t, rho_scale=1.5)

    # Add to dict: vectors as (N,3), scalars as (N,)
    S[f"{lat_key}_grad"]   = gV
    S[f"{lat_key}_slowness"] = slowness
    S[f"{lat_key}_CV_mag"] = CV_mag
    S[f"{lat_key}_CV_vec"] = CV_vec_t
    S[f"{lat_key}_laplace"] = lap_t
    S[f"{lat_key}_divCV"]   = div_v
    S[f"{lat_key}_curlnCV"] = curln
    S[f"{lat_key}_H11"]     = H11
    S[f"{lat_key}_H12"]     = H12
    S[f"{lat_key}_H22"]     = H22
    return S


def _xml_header(): return '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
def _xml_footer(): return '</VTKFile>\n'
def _dtype_str(a): return "Int32" if np.issubdtype(np.asarray(a).dtype, np.integer) else "Float32"

def _write_points_multi_vtp(points, arrays_numeric, arrays_string, fname):
    P = np.asarray(points, float); K = P.shape[0]
    with open(fname, "w", encoding="utf-8") as f:
        f.write(_xml_header())
        f.write(f'  <PolyData>\n    <Piece NumberOfPoints="{K}" NumberOfVerts="{K}" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">\n')
        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" Name="Points" format="ascii">\n')
        for p in P: f.write(f"          {p[0]:.9g} {p[1]:.9g} {p[2]:.9g}\n")
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        # Verts (one vertex per point)
        f.write('      <Verts>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i in range(K): f.write(f"          {i}\n")
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(1, K+1): f.write(f"          {i}\n")
        f.write('        </DataArray>\n')
        f.write('      </Verts>\n')
        # PointData
        f.write('      <PointData>\n')
        for name, arr in arrays_numeric.items():
            a = np.asarray(arr)
            f.write(f'        <DataArray type="{_dtype_str(a)}" Name="{name}" format="ascii">\n')
            for v in a: f.write(f"          {v}\n")
            f.write('        </DataArray>\n')
        for name, arr in arrays_string.items():
            f.write(f'        <DataArray type="String" Name="{name}" format="ascii">\n')
            for s in arr: f.write(f"          {str(s)}\n")
            f.write('        </DataArray>\n')
        f.write('      </PointData>\n')
        f.write('    </Piece>\n  </PolyData>\n')
        f.write(_xml_footer())

def export_lat_to_vtk(
    lat,
    verts, faces,
    coords, mesh_index,         # from cp.get_projection(...)
    cp_pnums,                   # cp.p_number (list/array of CARTO point numbers on the mesh side)
    out_dir,
    lam=1e-6
):
    """
    Produces:
      out_dir/mesh_multi.vtk  -> surface with ALL interpolated arrays
      out_dir/electrodes.vtp  -> projected points with raw per-point arrays + PointNumber + Label
    """
    os.makedirs(out_dir, exist_ok=True)

    # ----- align by point_number -----
    a = np.asarray(lat.p_numbers, dtype=int)      # LAT side
    b = np.asarray(cp_pnums,    dtype=int)        # CP side (same order as coords/mesh_index)
    a_index = {v:i for i,v in enumerate(a)}
    b_index = {v:i for i,v in enumerate(b)}
    common = sorted(set(a_index)&set(b_index), key=lambda x:(a_index[x], b_index[x]))
    if not common:
        raise RuntimeError("No common point numbers between LAT and CP sets.")
    ai = np.array([a_index[v] for v in common], dtype=int)  # indices into LAT arrays
    bi = np.array([b_index[v] for v in common], dtype=int)  # indices into CP arrays

    # ----- gather per-point arrays (LAT order -> aligned to ai) -----
    def A(x, dtype=float):
        return np.asarray(x, dtype=dtype)[ai]

    labels   = np.asarray(lat.labels, dtype=str)[ai]
    pnums    = np.asarray(lat.p_numbers, dtype=int)[ai]
    pt_coords = np.asarray(coords, float)[bi]
    known_idx = np.asarray(mesh_index, int)[bi]

    # numeric fields (add/remove as needed)
    First        = A(lat.First,        float)
    Second       = A(lat.Second,       float)
    Third        = A(lat.Third,        float)
    SR           = A(lat.SR,           float) if getattr(lat, "SR", None) is not None else np.full_like(First, np.nan)

    Sinus_dur    = A(lat.Sinus_dur,    float) if getattr(lat, "Sinus_dur", None) is not None else np.full_like(First, np.nan)

    First_V      = A(lat.First_Voltage,  float)
    Second_V     = A(lat.Second_Voltage, float)
    Third_V      = A(lat.Third_Voltage,  float)
    SR_V        = A(lat.Voltage_sinus,  float)
    min_V       = A(lat.min_Voltage, float)

    First_dur    = A(lat.First_dur,    float)
    Second_dur   = A(lat.Second_dur,   float)
    Third_dur    = A(lat.Third_dur,    float)

    First_delta  = A(lat.First_Delta,  float)
    Second_delta = A(lat.Second_Delta, float)
    Third_delta  = A(lat.Third_Delta,  float)

    First_defl   = A(lat.First_deflection,  float) if getattr(lat, "First_deflection", None)  is not None else np.full_like(First, np.nan)
    Second_defl  = A(lat.Second_deflection, float) if getattr(lat, "Second_deflection", None) is not None else np.full_like(First, np.nan)
    Third_defl   = A(lat.Third_deflection,  float) if getattr(lat, "Third_deflection", None)  is not None else np.full_like(First, np.nan)

    # per-point numeric dict (aligned to known_idx)
    values_point = {
        # LATs / timings
        "First": First, "Second": Second, "Third": Third, "SR": SR,
        "Sinus_dur": Sinus_dur,
        # Voltages
        "First_Voltage": First_V, "Second_Voltage": Second_V, "Third_Voltage": Third_V,
        "SR_Voltage": SR_V,
        "min_Voltage": min_V,
        # Durations
        "First_dur": First_dur, "Second_dur": Second_dur, "Third_dur": Third_dur,
        # Deltas
        "First_Delta": First_delta, "Second_Delta": Second_delta, "Third_Delta": Third_delta,
        # Deflections
        "First_deflection": First_defl, "Second_deflection": Second_defl, "Third_deflection": Third_defl,
    }

    # ----- interpolate ALL to mesh -----
    S = _interp_many_nanaware(verts, faces, known_idx, values_point, lam=lam)
    # --- Compare areas for voltages in the band [0.0, 0.5] and save a PNG ---
    # ---- build L, M once ----
    V = np.asarray(verts, float); F = np.asarray(faces, int)
    L, M = build_cotan_laplacian_and_mass(V, F)

    # ---- append derivative fields for a chosen LAT (First/SR/Second/Third) ----
    try:
        if "First"  in S: S = _append_derivatives_to_S(S, V, F, L, M, lat_key="First")
        if "Second" in S: S = _append_derivatives_to_S(S, V, F, L, M, lat_key="Second")
        if "Third"  in S: S = _append_derivatives_to_S(S, V, F, L, M, lat_key="Third")
        if "SR"     in S: S = _append_derivatives_to_S(S, V, F, L, M, lat_key="SR")
    except Exception as e:
        print("[DERIV]", e)

    # choose which voltage fields to compare (only ones present in S will be used)
    voltage_keys = ["SR_Voltage","First_Voltage", "Second_Voltage", "Third_Voltage"]  # edit if needed

    # compute absolute areas for each selected voltage in [0, 0.5]
    areas_5 = {}
    areas_15={}
    integrals={}
    area_total = mesh_area_valid(verts, faces, S[voltage_keys[1]])
    for key in voltage_keys:
        if key in S:
            try:
                A = band_area_graph(verts, faces, S[key], 0.0, 0.5)
                int_band = band_integral(verts, faces, S[key], np.nanmin(S[key]), np.nanmax(S[key]))
                areas_5[key] = float(A)/area_total
                integrals[key] = float(int_band)
            except Exception as e:
                print(f"[AREA] Skipped {key} due to error: {e}")

    for key in voltage_keys:
        if key in S:
            try:
                A = band_area_graph(verts, faces, S[key], 0.0, 1.5)
                areas_15[key] = float(A)/area_total
            except Exception as e:
                print(f"[AREA] Skipped {key} due to error: {e}")

    if areas_5:
        labels_ = list(areas_5.keys())
        vals   = np.array([areas_5[k] for k in labels_], dtype=float)
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        # plot
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        bars = ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("Area/ mesh Area")
        ax.set_title("Band area comparison: 0.0–0.5")

        # annotate bars with absolute + percentage
        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        # save in your repo/export directory
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, "voltage_band_area_0_0p5.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No matching voltage scalars found in S; nothing plotted.")

    if areas_15:
        labels_ = list(areas_15.keys())
        vals   = np.array([areas_15[k] for k in labels_], dtype=float)
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        # plot
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        bars = ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("Area/mesh Area")
        ax.set_title("Band area comparison: 0.0–1.5")

        # annotate bars with absolute + percentage
        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        # save in your repo/export directory
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, "voltage_band_area_0_1p5.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No matching voltage scalars found in S; nothing plotted.")
    # ---- optional: area histograms with 0.5 bins ----

    if integrals:
        labels_ = list(integrals.keys())
        vals   = np.array([integrals[k] for k in labels_], dtype=float)/area_total
        total  = float(np.sum(vals))
        perc   = (vals / total * 100.0) if total > 0 else np.zeros_like(vals)

        # plot
        fig, ax = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(labels_))
        bars = ax.bar(x, vals, align="center")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_, rotation=0)
        ax.set_ylabel("V(mv)")
        ax.set_title("mean_voltage")

        # annotate bars with absolute + percentage
        for xi, v, p in zip(x, vals, perc):
            ax.text(xi, v, f"{v:.3g}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9)

        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()

        # save in your repo/export directory
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, "mean_integral.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print("[AREA] Saved comparison plot:", out_png)
    else:
        print("[AREA] No matching voltage scalars found in S; nothing plotted.")
    # ---- optional: area histograms with 0.5 bins ----

    # ----- write mesh (multi scalar) -----
    mesh_vtk = os.path.join(out_dir, "mesh_multi.vtp")
    write_mesh_multi_scalar_vtp_vtk(verts, faces, S, mesh_vtk)


    # ----- write points (numeric + string) -----
    labels = np.asarray(labels)
    labels_num=(labels == "POS").astype(np.int8)
    pt_arrays_num = dict(values_point)
    pt_arrays_num["PointNumber"] = pnums
    pt_arrays_num.update({"Label": labels_num})
    pt_arrays_str={}
    
    pts_vtp = os.path.join(out_dir, "electrodes.vtp")
    _write_points_multi_vtp(pt_coords, pt_arrays_num, pt_arrays_str, pts_vtp)

    print("[VTK] wrote:")
    print("  ", mesh_vtk)
    print("  ", pts_vtp)
    return mesh_vtk, pts_vtp

