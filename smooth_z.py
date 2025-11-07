import numpy as np
import heapq
from scipy.sparse import diags
from smooth import heat_kernel_smooth_scalar

# ---- graph utils (edge lengths, Dijkstra ball) ----
def build_edge_graph_lengths(V, F):
    """Return adjacency list where each entry i contains [(j, edge_length), ...]."""
    n = V.shape[0]
    nbr = [[] for _ in range(n)]
    for a, b, c in F.astype(np.int32):
        for u, v in ((a, b), (b, c), (c, a)):
            # check whether v already in neighbor list for u
            if not any(v == p for (p, _) in nbr[u]):
                w = np.linalg.norm(V[u] - V[v])
                nbr[u].append((v, w))
            if not any(u == p for (p, _) in nbr[v]):
                nbr[v].append((u, w))
    return nbr
def dijkstra_ball(nbr, src, max_dist):
    """Return (idxs, dists) within geodesic radius max_dist from src."""
    dist = {src: 0.0}
    h = [(0.0, src)]
    out_idx = []
    out_dst = []
    while h:
        d,u = heapq.heappop(h)
        if d>max_dist: break
        if d != dist[u]: continue
        out_idx.append(u); out_dst.append(d)
        for v,w in nbr[u]:
            nd = d + w
            if nd <= max_dist and (v not in dist or nd < dist[v]):
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return np.array(out_idx, dtype=np.int32), np.array(out_dst, dtype=np.float64)

# ---- normals ----
def vertex_normals(V, F):
    tri = V[F]
    ntri = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    N = np.zeros_like(V, np.float64)
    for (i,j,k), nt in zip(F.astype(np.int32), ntri):
        N[i] += nt; N[j] += nt; N[k] += nt
    nrm = np.linalg.norm(N, axis=1) + 1e-15
    return N / nrm[:,None]

# ---- main geodesic-based smoother ----
def curvature_smooth_mesh_normals_geodesic(
    V, F,
    sample_radius,          # geodesic ball radius for building z (units of V)
    sample_sigma=None,      # Gaussian sigma on geodesic distance; default sample_radius/2
    diffuse_radius=0.0,     # heat-kernel radius for z denoising (geodesic)
    alpha=0.5,              # projection step along normals
    gamma=0.0,              # normal-agreement exponent (0 => ignore normals)
    nsteps=1,               # heat steps for z diffusion
):
    V = np.asarray(V, np.float64).copy()
    F = np.asarray(F, np.int32)
    n = V.shape[0]

    if sample_sigma is None:
        sample_sigma = max(1e-12, 0.5 * sample_radius)

    # geometry
    N = vertex_normals(V, F)
    nbr = build_edge_graph_lengths(V, F)

    # 1) build z using geodesic Gaussian weights in a ball of radius 'sample_radius'
    z = np.zeros(n, dtype=np.float64)
    inv2s2 = 1.0 / (2.0 * (sample_sigma**2) + 1e-30)

    for i in range(n):
        idxs, dists = dijkstra_ball(nbr, i, sample_radius)
        if idxs.size == 0:
            z[i] = 0.0
            continue
        pj = V[idxs]
        dij = (pj - V[i]) @ N[i]                      # signed offsets to tangent plane at i
        w_geo = np.exp(- (dists**2) * inv2s2)         # geodesic Gaussian
        if gamma > 0.0:
            w_n = np.maximum(0.0, (N[idxs] @ N[i]))**gamma
            w = w_geo * w_n
        else:
            w = w_geo
        s = w.sum()
        z[i] = (w @ dij) / s if s > 1e-15 else 0.0

    # 2) diffuse z by heat kernel on the mesh (your FEM solver)
    if diffuse_radius > 0.0:
        z_smooth = heat_kernel_smooth_scalar(V, F, z, radius=diffuse_radius, nsteps=nsteps)
    else:
        z_smooth = z

    # 3) project vertices along normals
    V_new = V + (alpha * z_smooth)[:,None] * N
    return V_new, z, z_smooth
def mean_edge_length(V, F):
    """Return mean length of all unique edges in a triangular mesh."""
    E = set()
    for a, b, c in F:
        E.add(tuple(sorted((a, b))))
        E.add(tuple(sorted((b, c))))
        E.add(tuple(sorted((c, a))))
    if not E:
        return 1.0  # fallback for degenerate case
    E = np.array(list(E), dtype=np.int32)
    lengths = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return float(lengths.mean()) if len(lengths) else 1.0
