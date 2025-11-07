import os, sys, numpy as np
import moderngl


# --- sibling repo paths ---
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [
    os.path.abspath(os.path.join(here, "..", "Carto_tool")),
    os.path.abspath(os.path.join(here, "..", "Engine")),
    os.path.abspath(os.path.join(here))
]

from Eng.Engine.Engine_main import Engine
from CAR_TOOL.CARTO_Tool import Carto
from shader import vertex_shader,fragment_shader

# ---------------- Shaders (WORLD SPACE) ----------------


# ------------- LUT texture (discrete) -------------
def make_lut_texture(ctx, cmap_name="viridis", n_colors=8):
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(cmap_name, n_colors)
        lut = (cmap(np.linspace(0, 1, n_colors))[:, :3] * 255).astype(np.uint8)
    except Exception:
        t = np.linspace(0, 1, n_colors, dtype=np.float32)
        lut = np.stack([t, np.zeros_like(t), 1.0 - t], axis=1)
        lut = (lut * 255).astype(np.uint8)
    tex = ctx.texture((n_colors, 1), 3, lut.tobytes(), alignment=1)
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)  # crisp bands
    tex.repeat_x = False
    tex.repeat_y = False
    return tex

from smooth import *

from smooth_z import *
# ------------- Main -------------
if __name__ == "__main__":
    carto = Carto()
    verts, faces, scals1, scals2,LAT = carto.pars_mesh_file_with_electrode()  # verts:(N,3), faces:(M,3)
    eng = Engine()

    # choose scalar
    s = LAT
    target_radius = 2  # e.g., mm — tune to your mesh scale
    V = np.asarray(verts, np.float64).copy()
    F = np.asarray(faces, np.int32)

    # V: (n,3) float64; F: (m,3) int32
    Lmean = mean_edge_length(V, F)         # you already have this helper
    sample_radius  = 5 * Lmean           # geodesic neighborhood for measuring z
    sample_sigma   = 5 * Lmean   # Gaussian width on the geodesic ball
    diffuse_radius = 5       # heat-kernel radius for z denoising
    alpha          = 0.1                   # smaller step; use multiple iterations
    gamma          = 0.0                   # ignore normals (remove sharp sections)
    nsteps         = 2

    iters = 5
    for _ in range(iters):
        V, z_raw, z_s = curvature_smooth_mesh_normals_geodesic(
            V, F,
            sample_radius=sample_radius,
            sample_sigma=sample_sigma,
            diffuse_radius=diffuse_radius,
            alpha=alpha,
            gamma=gamma,
            nsteps=nsteps
        )


    s_smooth = heat_kernel_smooth_scalar(verts, faces, s, radius=target_radius, nsteps=2)


    verts = np.asarray(V, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    s     = np.asarray(s,     dtype=np.float32).reshape(-1)

    # optional: center camera on mesh
    mesh_center = verts.mean(axis=0).astype(np.float32)

    eng.player.center = mesh_center


    # interleave [pos.xyz, scalar]
    s_final=s[:, None]-s_smooth[:, None]
    print(s_final.min(),s_final.max(),s_final.mean(),s_final.std())
    vertex_data = np.concatenate([verts, s_smooth[:, None]], axis=1).astype("f4").tobytes()
    vbo_format  = "3f 1f"
    attrs       = ("aPosition", "aScalar")

    # program + buffers
    program = eng.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vbo = eng.ctx.buffer(vertex_data)
    ibo = eng.ctx.buffer(faces.astype("i4").tobytes())
    vao = eng.ctx.vertex_array(program, [(vbo, vbo_format, *attrs)], index_buffer=ibo)

    # required matrices (engine updates m_view/m_proj every frame)
    program["m_model"].write(np.eye(4, dtype="f4").tobytes())

    # lighting defaults (no attenuation)
    program["uAmbient"].value   = (0.58, 0.58, 0.58)
    program["uSpecColor"].value = (0.4, 0.4, 0.4)
    program["uShininess"].value = 64.0

    # scalar controls
    program["uSMin"].value   = float(np.min(s))
    program["uSMax"].value   = float(np.max(s))
    program["uGamma"].value  = 1
    num_colors =50 # set discrete bands
    program["uNumColors"].value = int(num_colors)

    # LUT: discrete N colors
    lut_tex = make_lut_texture(eng.ctx, cmap_name='rainbow_r', n_colors=num_colors)
    lut_tex.use(location=0)
    program["uLUT"].value = 0

    # initial light/camera positions (world space)
    def _get_center():
        # prefer engine player center if present; fallback to mesh center
        try:
            c = np.array(eng.player.position, dtype=np.float32).reshape(3)
            if np.all(np.isfinite(c)): return c
        except Exception:
            pass
        return mesh_center

    cam_pos = _get_center()
    light_pos = cam_pos.copy()            # light at the camera by default
    program["uCameraPosWorld"].value = tuple(cam_pos)
    program["uLightPosWorld"].value  = tuple(light_pos)

    # object the engine expects
    class Drawable: pass
    cmesh = Drawable()
    cmesh.vao = vao
    cmesh.program = program
    cmesh.opaque = True
    def render(self): self.vao.render()
    cmesh.render = render.__get__(cmesh)

    eng.scene.objects["cmesh"] = cmesh

    # --- per-frame updater: keep light at camera (both world space) ---
    def update_light_and_camera():
        c = _get_center()
        program["uCameraPosWorld"].value = (float(c[0]), float(c[1]), float(c[2]))
        program["uLightPosWorld"].value  = (float(c[0]), float(c[1]), float(c[2]))

    # run
    eng.run(render_func=[update_light_and_camera])
