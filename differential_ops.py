import numpy as np
from scipy.sparse import diags

# ---- normals & areas ----
def face_normals_and_areas(V, F):
    V = np.asarray(V, float); F = np.asarray(F, int)
    e1 = V[F[:,1]] - V[F[:,0]]
    e2 = V[F[:,2]] - V[F[:,0]]
    n  = np.cross(e1, e2)
    A  = 0.5*np.linalg.norm(n, axis=1)
    nz = np.maximum(A, 1e-30)
    n_unit = n / (2*nz[:,None]) * 2.0  # normalize: n / |n|
    n_unit = n / np.maximum(np.linalg.norm(n, axis=1)[:,None], 1e-30)
    return n_unit, A

def vertex_normals_area_weighted(V, F):
    n_face, A = face_normals_and_areas(V, F)
    n = V.shape[0]
    acc = np.zeros((n,3), float)
    for f,(i,j,k) in enumerate(F):
        w = A[f]
        nf = n_face[f]*w
        acc[i]+=nf; acc[j]+=nf; acc[k]+=nf
    norms = np.linalg.norm(acc, axis=1)
    norms[norms==0] = 1.0
    return acc / norms[:,None]

def vertex_voronoi_areas_lumped(V, F):
    # simple lumped 1/3 triangle area per incident face
    _, Af = face_normals_and_areas(V, F)
    n = V.shape[0]
    A = np.zeros(n, float)
    for f,(i,j,k) in enumerate(F):
        a = Af[f]/3.0
        A[i]+=a; A[j]+=a; A[k]+=a
    A[A<=0] = 1.0
    return A

# ---- per-face gradient of scalar (skip NaN faces) ----
def per_face_gradient_scalar(V, F, t):
    V = np.asarray(V, float); F = np.asarray(F, int)
    t = np.asarray(t, float).ravel()
    nF = F.shape[0]
    grad = np.full((nF,3), np.nan, float)

    e1 = V[F[:,1]] - V[F[:,0]]
    e2 = V[F[:,2]] - V[F[:,0]]
    n  = np.cross(e1, e2)
    dblA = np.linalg.norm(n, axis=1) * 1.0  # 2*Area
    mask_geom = dblA > 1e-30
    n_hat = np.zeros_like(n)
    n_hat[mask_geom] = n[mask_geom] / dblA[mask_geom][:,None]

    ti = t[F[:,0]]; tj = t[F[:,1]]; tk = t[F[:,2]]
    valid_face = mask_geom & np.isfinite(ti) & np.isfinite(tj) & np.isfinite(tk)
    if not np.any(valid_face):
        return grad  # all NaN

    # formula: grad_f t = (Δt_j*(n̂ × e2) + Δt_k*(e1 × n̂)) / (2A) = (...) / dblA
    dtj = (tj - ti)[valid_face]
    dtk = (tk - ti)[valid_face]
    e1v = e1[valid_face]; e2v = e2[valid_face]; nv = n_hat[valid_face]; dA = dblA[valid_face][:,None]

    term1 = np.cross(nv, e2v) * dtj[:,None]
    term2 = np.cross(e1v, nv) * dtk[:,None]
    grad[valid_face] = (term1 + term2) / np.maximum(dA, 1e-30)
    return grad

# ---- average face gradients to vertices (area-weighted), project to tangent ----
def gradient_vertices_from_faces(V, F, grad_face):
    V = np.asarray(V, float); F = np.asarray(F, int)
    n = V.shape[0]
    _, Af = face_normals_and_areas(V, F)
    acc = np.zeros((n,3), float)
    wts = np.zeros(n, float)

    valid = np.all(np.isfinite(grad_face), axis=1)
    for f,(i,j,k) in enumerate(F):
        if not valid[f]: continue
        w = Af[f]
        g = grad_face[f]
        acc[i]+=w*g; acc[j]+=w*g; acc[k]+=w*g
        wts[i]+=w;   wts[j]+=w;   wts[k]+=w

    grad_v = np.full((n,3), np.nan, float)
    nz = wts>0
    grad_v[nz] = acc[nz] / wts[nz][:,None]

    # project to tangent plane
    n_v = vertex_normals_area_weighted(V, F)
    # v_tan = v - (v·n)n
    vdotn = np.sum(grad_v*n_v, axis=1)
    grad_v[nz] = grad_v[nz] - (vdotn[nz][:,None]*n_v[nz])
    return grad_v

# ---- CV fields from gradient ----
def cv_from_grad(grad_v, eps=None):
    s = np.linalg.norm(grad_v, axis=1)  # slowness
    finite = np.isfinite(s)
    if eps is None:
        med = np.nanmedian(s[finite]) if np.any(finite) else 1.0
        eps = 1e-3*max(med, 1e-6)
    CV_mag = np.full_like(s, np.nan)
    CV_vec = np.full_like(grad_v, np.nan)
    nz = finite & (s>0)
    CV_mag[nz] = 1.0/np.maximum(s[nz], eps)
    # v = -grad / ||grad||^2
    CV_vec[nz] = -grad_v[nz] / (np.maximum(s[nz], eps)**2)[:,None]
    return s, CV_mag, CV_vec

# ---- cotan Laplacian application (needs L and M from your smooth.py) ----
def laplacian_scalar_cotan(L, M, t):
    t = np.asarray(t, float).ravel()
    # Δ t ≈ M^{-1} L t
    Mdiag = M.diagonal() if hasattr(M, "diagonal") else np.asarray(M, float).ravel()
    out = np.full_like(t, np.nan)
    ok = np.isfinite(t)
    if not np.any(ok):
        return out
    Lt = (L @ np.nan_to_num(t, nan=0.0))
    # zero out rows where inputs were NaN to avoid pollution near holes
    out[ok] = Lt[ok] / np.maximum(Mdiag[ok], 1e-30)
    return out

# ---- divergence (cotan) of a vertex vector field ----
def divergence_cotan(V, F, vec_v):
    V = np.asarray(V, float); F = np.asarray(F, int)
    vec_v = np.asarray(vec_v, float)
    n = V.shape[0]
    # precompute cot weights
    # for edge (i,j) in tri (i,j,k): weight = cot(alpha_k) + cot(beta_k) shared by two triangles
    I = []; J = []; W = []
    for (i,j,k) in F:
        vi, vj, vk = V[i], V[j], V[k]
        # angles at vertices opposite to edges:
        # cot at k for edge (i,j)
        cot_k = _cot_angle(vj-vi, vk-vi, vj-vk)  # helper below not to confuse: we just need cot(alpha_ij^k)
        cot_i = _cot_angle(vk-vj, vi-vj, vk-vi)
        cot_j = _cot_angle(vi-vk, vj-vk, vi-vj)
        # accumulate symmetric weights
        W.extend([(i,j,cot_k),(j,i,cot_k),
                  (j,k,cot_i),(k,j,cot_i),
                  (k,i,cot_j),(i,k,cot_j)])
    # build neighbor list with summed weights
    from collections import defaultdict
    accum = defaultdict(float)
    for i,j,w in W:
        accum[(i,j)] += w
    # Voronoi areas
    A = vertex_voronoi_areas_lumped(V, F)
    div = np.full(n, np.nan, float)
    # compute divergence with symmetric stencil:
    # div_i ≈ (1/(2A_i)) * sum_j w_ij (p_i - p_j) · (v_i + v_j)/2
    for i in range(n):
        num = 0.0; den = 2.0*A[i]
        if den <= 0: continue
        # neighbors j are those for which (i,j) in accum
        # to avoid O(n^2), iterate only j with a recorded weight
        # collect js:
        js = [j for (ii,j) in accum.keys() if ii==i]
        if not js:
            continue
        vi = V[i]; vi_vec = vec_v[i]
        s = 0.0; count = 0
        for j in js:
            w = accum[(i,j)]
            if not np.isfinite(w): continue
            if not (np.all(np.isfinite(vec_v[i])) and np.all(np.isfinite(vec_v[j]))):
                continue
            pij = vi - V[j]
            vij = 0.5*(vi_vec + vec_v[j])
            s += w * np.dot(pij, vij)
            count += 1
        if count>0:
            div[i] = s/den
    return div

def _cot_angle(a,b,c):
    # (simple, stable-ish) cot at the angle between vectors (b-a) and (c-a) ?
    # Here we just need magnitudes; use standard cot(theta)=u·v/||u×v||
    # For our callers we pass edge-related vectors directly.
    u = b; v = c
    num = float(np.dot(u, v))
    den = np.linalg.norm(np.cross(u, v))
    if den < 1e-30: return 0.0
    return num/den

# ---- curl normal (per-face then to vertices) ----
def curl_normal(V, F, vec_v):
    V = np.asarray(V, float); F = np.asarray(F, int)
    vec_v = np.asarray(vec_v, float)
    nF = F.shape[0]
    n_hat, Af = face_normals_and_areas(V, F)
    curl_f = np.full(nF, np.nan, float)

    # per-edge circulation around each face
    for f,(i,j,k) in enumerate(F):
        if Af[f] <= 1e-30: continue
        if (not np.all(np.isfinite(vec_v[i])) or
            not np.all(np.isfinite(vec_v[j])) or
            not np.all(np.isfinite(vec_v[k]))):
            continue
        # oriented boundary (i->j), (j->k), (k->i)
        s = 0.0
        for (a,b) in ((i,j),(j,k),(k,i)):
            E = V[b] - V[a]
            t_hat = np.cross(n_hat[f], E)
            ln = np.linalg.norm(t_hat)
            if ln < 1e-30: continue
            t_hat /= ln
            v_ab = 0.5*(vec_v[a] + vec_v[b])
            s += np.dot(v_ab, t_hat) * np.linalg.norm(E)
        curl_f[f] = s / Af[f]

    # to vertices (area-weighted)
    nV = V.shape[0]
    curl_v = np.full(nV, np.nan, float)
    acc = np.zeros(nV, float)
    wts = np.zeros(nV, float)
    valid = np.isfinite(curl_f)
    for f,(i,j,k) in enumerate(F):
        if not valid[f]: continue
        w = Af[f]
        val = curl_f[f]
        for vtx in (i,j,k):
            acc[vtx]+=w*val; wts[vtx]+=w
    nz = wts>0
    curl_v[nz] = acc[nz]/wts[nz]
    return curl_v

# ---- Hessian via local quadratic fit on 1-ring ----
def hessian_quadratic_fit(V, F, t, rho_scale=1.5):
    V = np.asarray(V, float); F = np.asarray(F, int)
    t = np.asarray(t, float).ravel()
    n = V.shape[0]
    H11 = np.full(n, np.nan); H12 = np.full(n, np.nan); H22 = np.full(n, np.nan)
    # tangent frames
    nrm = vertex_normals_area_weighted(V, F)

    # build 1-ring adjacency
    nbr = [[] for _ in range(n)]
    for (i,j,k) in F:
        nbr[i].extend([j,k]); nbr[j].extend([i,k]); nbr[k].extend([i,j])
    for i in range(n):
        if not np.isfinite(t[i]): continue
        nb = np.unique(nbr[i])
        if nb.size < 3:
            continue
        # orthonormal frame
        z = nrm[i]
        # pick ex from any neighbor direction not parallel to z
        ex = V[nb[0]] - V[i]
        ex -= np.dot(ex, z)*z
        if np.linalg.norm(ex) < 1e-12:
            continue
        ex /= np.linalg.norm(ex)
        ey = np.cross(z, ex); ey /= np.maximum(np.linalg.norm(ey), 1e-12)
        # local coords
        Pi = V[nb] - V[i]
        x = Pi @ ex; y = Pi @ ey
        dt = t[nb] - t[i]
        ok = np.isfinite(dt)
        x = x[ok]; y = y[ok]; dt = dt[ok]
        if x.size < 5:  # need enough equations
            continue
        # weights
        ell = np.sqrt(x*x + y*y)
        if ell.size==0: continue
        h = rho_scale * np.median(ell)
        h = max(h, 1e-6)
        w = np.exp(-(ell**2)/(h*h))
        # quadratic model: c + a x + b y + 0.5*(h11 x^2 + 2 h12 x y + h22 y^2)
        # We fit parameters p=[a,b,h11,h12,h22] because c cancels with dt (centered at i)
        X = np.column_stack([x, y, 0.5*x*x, x*y, 0.5*y*y])
        W = diags(w)
        # solve weighted least squares
        try:
            A = X.T @ W @ X
            b = X.T @ W @ dt
            p = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
        H11[i], H12[i], H22[i] = p[2], p[3], p[4]
    return H11, H12, H22