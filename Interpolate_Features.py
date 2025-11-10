from Json_aux.EXTRACT_FROM_JASON import LAT_points
import json,os,sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
from batch_vtk_features import export_lat_to_vtk
here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [
    os.path.abspath(os.path.join(here, "..", "Carto_tool")),
    os.path.abspath(os.path.join(here, "..", "Engine")),
    os.path.abspath(os.path.join(here))
]
from CAR_TOOL.CARTO_Tool import Carto
from cartopoints import Carto_points
from smooth import *
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from Eng.Engine.Engine_main import Engine
from shader import vertex_shader,fragment_shader
from Cmesh import make_lut_texture



if __name__=="__main__":
    root = tk.Tk()
    root.withdraw()
    file_path=None
    # make sure dialog appears on top
    if not file_path:
        root.attributes("-topmost",True)
        file_path = filedialog.askopenfilename(parent=root)
        root.destroy()
        print(file_path)
    with open(file_path,"r") as f:
        a=json.load(f)
    lat=LAT_points(a)
    a=np.array(lat.p_numbers,dtype="int")
    V1=np.array(lat.First_Voltage,dtype="float64")
    V2=np.array(lat.Second_Voltage,dtype="float64")
    V3=np.array(lat.Third_Voltage,dtype="float64")
    carto=Carto()
    verts, faces, scals1, scals2,LAT = carto.pars_mesh_file_with_electrode()  # verts:(N,3), faces:(M,3)

    cp=Carto_points(carto)
    cp.extract_all()
    projection=cp.get_projection(cp.points,verts)
    b=np.array(cp.p_number,dtype="int")
    print(cp.p_number,projection)
    coords,mesh_index=projection
    mesh_index=np.array(mesh_index,dtype="int")
    a_index = {v: i for i, v in enumerate(a)}
    b_index = {v: i for i, v in enumerate(b)}

    # find common point numbers
    common = sorted(set(a_index.keys()) & set(b_index.keys()), key=lambda x: (a_index[x], b_index[x]))

    # now extract index arrays
    ai = np.array([a_index[v] for v in common], dtype=int)
    bi = np.array([b_index[v] for v in common], dtype=int)
    assert np.all(a[ai] == b[bi])
    out_dir = os.path.join(os.path.dirname(file_path), "VTK_exports")
    export_lat_to_vtk(
        lat=lat,
        verts=verts, faces=faces,
        coords=coords, mesh_index=mesh_index,
        cp_pnums=np.array(cp.p_number, dtype=int),
        out_dir=out_dir,
        lam=1e-6
    )
    # verify
    assert np.all(a[ai] == b[bi])
    print(f"{len(common)} common point numbers")
    print("ai[:10] =", a[ai[:]])
    print("bi[:10] =", b[bi[:]])
    print(a[ai[:]]==b[bi[:]])
    mesh_index[bi],V1[ai]
    eng = Engine()
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)    # optional: center camera on mesh
    mesh_center = verts.mean(axis=0).astype(np.float32)
    known_idx = mesh_index[bi]   # indices on the mesh
    known_val = V1[ai]            # voltages aligned

    # Optional tiny Tikhonov (screened Laplacian) to stabilize if constraints are very sparse/noisy
    s = laplace_interpolate_from_sparse(verts, faces, known_idx, known_val, lam=1e-6)
    eng.player.center = mesh_center

    vertex_data = np.concatenate([verts, s[:, None]], axis=1).astype("f4").tobytes()
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
    num_colors =20 # set discrete bands
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

