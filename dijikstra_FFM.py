from CARTO_Tool import Carto
from cartopoints import Carto_points,np
import heapq
import numpy as np
from engine.decorate.Line import Line
from engine.Engine import Engine
from engine.decorate.Texture.adding_texture import add_color_bar
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib
import time
from engine import shader_program,mesh
from Cmesh import *

def euclidean_distance(p, q):
    """Calculate the Euclidean distance between two 3D points."""
    return np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2 + (p[2] - q[2])**2)

def build_graph(nodes:any, triangles:any,complete=False):
    """
    Constructs an adjacency list for the mesh.
    
    nodes: List of tuples (x, y, z) representing 3D coordinates.
    triangles: List of tuples (i, j, k) with  indices into the nodes list.
    returns dictionaries of graph, edges, triangles graph if you set complete as True
    """
    # Initialize an empty graph: each node index maps to a list of (neighbor_index, weight)
    graph = {i: [] for i in range(len(nodes))}
    edges = {}
    triangles_graph={i:[] for i in range(len(nodes))}
    
    # For each triangle, add its three edges
    for tri in triangles:
        # Each triangle gives us three edges: (i, j), (j, k), (k, i)
        for a, b,c in [(tri[0], tri[1],tri[2]), (tri[1], tri[2],tri[0]), (tri[2], tri[0],tri[1])]:
            # Order the edge vertices to avoid duplicate entries (e.g., (1,2) same as (2,1))
            if a > b:
                a, b = b, a
            if (a,b) not in triangles_graph[c]:
                triangles_graph[c].append((a,b))

            if (a, b) not in edges:
                dist = euclidean_distance(nodes[a], nodes[b])
                edges[(a, b)] = dist
    
    # Populate the graph (edges are bidirectional)
    for (a, b), dist in edges.items():
        graph[a].append((b, dist))
        graph[b].append((a, dist))
    if complete:
        return graph,edges,triangles_graph
    return graph
def dijkstra_FMM(graph,source,triangles_graph,vertices):
    T={node:np.inf for node in graph}
    pq=[]
    T[source]=0
    visited = set()
    heapq.heappush(pq,(0,source))
    t,u=heapq.heappop(pq)
    visited.add(u)
    for v, weight in graph[u]:
        new_T = t + weight
        T[v]=new_T
        
        heapq.heappush(pq,(new_T,v))
    while pq:
        t,u=heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        # Explore each neighbor of u
        for v, weight in graph[u]:
            for A,B in triangles_graph[v]:
                if A in visited and B in visited:
                    new_T=solve_eikonal_triangle(T[A],T[B],vertices[A],vertices[B],vertices[v])
                    if new_T is not None and new_T < T[v]:
                        T[v] = new_T
                        #if new_T>50:
                        #    continue
                        heapq.heappush(pq, (new_T, v))
                    #if T[u]+weight<T[v]:
                    #    pass
                    #    T[v]=T[u]+weight
                    #    heapq.heappush(pq, (T[v], v))

            
    return T

import random
def solve_eikonal_triangle(TA, TB, a, b, c):
    # Vector from A to B
    AB = b - a
    u = np.linalg.norm(AB)
    e1 = AB / u  # unit vector along AB

    # Vector from A to C
    AC = c - a
    v = np.dot(AC, e1)
    w = np.linalg.norm(AC - v * e1)
    p = (TB - TA) / u

    discrim = (w**2 )* (1 - p**2)
    if discrim < 0:
        return None  # No real soution; fallback needed
    #s=random.choices(np.linspace(0.9,4.25,5),[0.6,0.2,0.05,0.05,0.1])[0]
    sqrt_discrim = np.sqrt(discrim)
    TC1 = TA + p * v + sqrt_discrim
    TC2 = TA + p * v - sqrt_discrim

    TC = max(TC1, TC2)  # Causal solution
    return TC


















def k_nearest_samples(graph, samples, k=3,eng=None):
    """
    For each node in 'graph', finds the k (default 3) closest sample nodes.
    
    graph: A dictionary mapping each node to a list of (neighbor, weight) pairs.
    samples: An iterable of nodes that are the sample points.
    k: Number of nearest samples to find per node.
    
    Returns a dictionary mapping each node to a list of tuples (distance, sample)
    representing the k smallest distances from that node to a sample.
    """
    # For every node, we store a list of (distance, sample) pairs.
    nearest = {node: [] for node in graph}
    
    # Priority queue: items are (distance, current_node, originating_sample)
    pq = []
    visited={}
    # Initialize the queue with all sample nodes (distance 0 from themselves)
    for s in samples:
        visited[s]=set()
        heapq.heappush(pq, (0, s, s))
    for index in samples:
        nearest[s].append((0, s))
        t, u, origin= (0,s,s)
        visited[origin].add(u)
        for v, weight in graph[u]:
            new_T = t + weight
            
            heapq.heappush(pq,(new_T,v,s))
    
    while pq:
        d, u, origin = heapq.heappop(pq)
        # Check if this entry is still valid (it might be outdated if a better distance was found)
        if (d, origin) not in nearest[u]:
            continue
        if u in visited[origin]:
            continue
        visited[origin].add(u)
        # Explore each neighbor of u
        for v, weight in graph[u]:
            for A,B in triangles_graph[v]:
                if A in visited[origin] and B in visited[origin]:
                    new_T=solve_eikonal_triangle(T[A],T[B],vertices[A],vertices[B],vertices[v])
                    update = True
                    for i,(existing_T, existing_origin) in enumerate(nearest[v]):
                        if existing_origin == origin:
                            if new_T >= existing_T:
                                update = False
                            else:
                                # Found a better route from the same sample; remove the old one.
                                nearest[v].pop(i)
                            break
            
            if update:
                # Only add if we have fewer than k routes or the new one is better than the worst we have.
                if new_T is not None and (len(nearest[v]) < k or new_T < max(nearest[v], key=lambda x: x[0])[0]):
                    if new_T>50:
                        continue
                    heapq.heappush(pq, (new_T, v, origin))
                    nearest[v].append((new_T, origin))
                    # Keep the list sorted (smallest distances first) and trim to k items.
                    nearest[v].sort(key=lambda x: x[0])
                    if len(nearest[v]) > k:
                        nearest[v] = nearest[v][:k]
                    
    #return {m:nearest[m] for m in samples}
    return nearest




def write_vtk_with_scalars(filename, verts, triangles, scalars,normals):
    with open(filename, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('VTK from Python\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')

        # Write vertices
        f.write(f'POINTS {len(verts)} float\n')
        for v in verts:
            f.write(f'{v[0]} {v[1]} {v[2]}\n')

        # Write triangles
        f.write(f'POLYGONS {len(triangles)} {len(triangles) * 4}\n')
        for tri in triangles:
            f.write(f'3 {tri[0]} {tri[1]} {tri[2]}\n')

        # Write scalar data (per-vertex)
        f.write(f'POINT_DATA {len(scalars)}\n')
        f.write('SCALARS vertex_scalar float 1\n')
        f.write('LOOKUP_TABLE default\n')
        for s in scalars:
            f.write(f'{s}\n')
        # Normals
        f.write('VECTORS normals float\n')
        for n in normals:
            f.write(f'{n[0]} {n[1]} {n[2]}\n')







if __name__=="__main__":

    carto = Carto()
    cp=Carto_points(carto,triple=False)
    vertices,faces,_,_,_=carto.pars_mesh_file_with_electrode() 
    verts = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    
    # Build the graph from the mesh
    graph ,edges,triangles_graph= build_graph(vertices, faces,True)
  
    cp.extract_all()
    projected_points,indices=cp.get_projection(cp.points,vertices)


    from engine.decorate.Sphere import Sphere,glm
    from engine.decorate.Line import Line
    from engine.Engine import Engine
    eng=Engine()
    sp=Sphere(20,20,eng)
    sp.inverse_normals()

    
    for p in projected_points:
        sp.translate(glm.vec3(p),1)
        sp.draw(eng,colors=[glm.vec3(0.1,0.7,0)])
        sp.translate(-glm.vec3(p),1)
    sp.translate(glm.vec3(0,0,0),20)

    
    prev=[]
    prev_all=[]
    from itertools import count

    id_gen = count()
    ll=Line(eng)

    T=dijkstra_FMM(graph,120,triangles_graph,vertices)
    scalars=[t if t is not np.inf else None for t in T.values()]
    vmin=np.min([sc for sc in scalars if sc is not None])
    vmax=np.max([sc for sc in scalars if sc is not None])



    eng.player.center=mesh.Mesh.center(vertices)
    max_T=np.max([t if t is not None else -np.inf for t in scalars])
    s     = np.asarray(scalars,     dtype=np.float32).reshape(-1)
    vertex_data = np.concatenate([vertices, s[:, None]], axis=1).astype("f4").tobytes()
    vbo_format  = "3f 1f"
    attrs       = ("aPosition", "aScalar")

    # program + buffers
    program = eng.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vbo = eng.ctx.buffer(vertex_data)
    ibo = eng.ctx.buffer(faces.astype("i4").tobytes())
    vao = eng.ctx.vertex_array(program, [(vbo, vbo_format, *attrs)], index_buffer=ibo)    

    #write_vtk_with_scalars("saman_test_dijkstra_1.vtk",vertices,faces,scalars,vert_normals)
    # required matrices (engine updates m_view/m_proj every frame)
    program["m_model"].write(np.eye(4, dtype="f4").tobytes())

    # lighting defaults (no attenuation)
    program["uAmbient"].value   = (0.58, 0.58, 0.58)
    program["uSpecColor"].value = (0.4, 0.4, 0.4)
    program["uShininess"].value = 64.0

    # scalar controls
    program["uSMin"].value   = float(np.min(s))
    program["uSMax"].value   = float(np.max(s))
    program["uGamma"].value  = 0.8
    num_colors = 40 # set discrete bands
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





    