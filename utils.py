import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
import gmsh
import numpy as np
import torch

SU2_SHAPE_IDS = {
    'line': 3,
    'triangle': 5,
    'quad': 9,
}

def generate_mesh(profile, save_path, visualize=False):
    """
    This function generates mesh for an airfoil from specified options.
    TODO: add more options and finer control.
    """
    gmsh.initialize()
    gmsh.model.add('new model')
    lc = 0.002
    
    airfoil_points = []
    for point in profile:
        x, y = point
        airfoil_points.append(gmsh.model.geo.add_point(x, y, 0, lc))
    airfoil_points.append(airfoil_points[0])
    
    top = gmsh.model.geo.add_point(0, 10, 0, 500*lc)
    center = gmsh.model.geo.add_point(0, 0, 0, 500*lc)
    bottom = gmsh.model.geo.add_point(0, -10, 0, 500*lc)
    arc = gmsh.model.geo.add_circle_arc(top, center, bottom)

    top_right = gmsh.model.geo.add_point(10, 10, 0, 1000*lc)
    bottom_right = gmsh.model.geo.add_point(10, -10, 0, 1000*lc)
    rec = gmsh.model.geo.add_polyline([top, top_right, bottom_right, bottom])
    
    airfoil = gmsh.model.geo.add_spline(airfoil_points)
    
    surface = gmsh.model.geo.add_plane_surface([
        gmsh.model.geo.add_curve_loop([arc, -rec]), # farfield
        gmsh.model.geo.add_curve_loop([airfoil]) 
    ])
    gmsh.model.geo.synchronize()

    airfoil_tag = gmsh.model.add_physical_group(1, [airfoil])
    gmsh.model.set_physical_name(1, airfoil_tag, 'airfoil')
    farfield_tag = gmsh.model.add_physical_group(1, [arc, -rec])
    gmsh.model.set_physical_name(1, farfield_tag, 'farfield')
    surface_tag = gmsh.model.add_physical_group(2, [surface])
    gmsh.model.set_physical_name(2, surface_tag, 'surface')

    gmsh.model.mesh.generate(2)
    gmsh.write(save_path)

    if visualize:
        gmsh.fltk.run()
    gmsh.finalize()

def read_dat(file_name):
    profile = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if not line.startswith('#'):
                x, y = line.strip().split()
                profile.append((float(x), float(y)))
    return profile

def get_mesh_graph(mesh_filename):
    def get_rhs(s: str) -> str:
        return s.split('=')[-1]

    marker_dict = {}
    with open(mesh_filename) as f:
        for line in f:
            if line.startswith('NPOIN'):
                num_points = int(get_rhs(line))
                mesh_points = [[float(p) for p in f.readline().split()[:2]]
                               for _ in range(num_points)]
                nodes = torch.tensor(mesh_points, dtype=torch.float)

            if line.startswith('NMARK'):
                num_markers = int(get_rhs(line))
                for _ in range(num_markers):
                    line = f.readline()
                    assert line.startswith('MARKER_TAG')
                    marker_tag = get_rhs(line).strip()
                    num_elems = int(get_rhs(f.readline()))
                    marker_elems = [[int(e) for e in f.readline().split()[-2:]]
                                    for _ in range(num_elems)]
                    marker_dict[marker_tag] = marker_elems

            if line.startswith('NELEM'):
                edges = []
                triangles = []
                quads = []
                num_edges = int(get_rhs(line))
                for _ in range(num_edges):
                    elem = [int(p) for p in f.readline().split()]
                    if elem[0] == SU2_SHAPE_IDS['triangle']:
                        n = 3
                        triangles.append(elem[1:1+n])
                    elif elem[0] == SU2_SHAPE_IDS['quad']:
                        n = 4
                        quads.append(elem[1:1+n])
                    else:
                        raise NotImplementedError
                    elem = elem[1:1+n]
                    edges += [[elem[i], elem[(i+1) % n]] for i in range(n)]
                edges = torch.tensor(edges, dtype=torch.long).T
                elems = [triangles, quads]

    return nodes, edges, elems, marker_dict

def plot_scalar_field(fields, X=None, tri=None, show=False, title=None, shading='gouraud'):
    """
    fields should be of shape (n, m, *) for triangular mesh or (n, m, *, *) for regular grid
    """
    assert len(fields.shape) > 2
    nrows, ncols = fields.shape[:2]
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols*5, nrows*5))
    
    if X!=None and tri !=None: # triangular mesh
        for i, j in np.ndindex(fields.shape[:2]):
            axes[i, j].tripcolor(X[0], X[1], tri, fields[i, j], shading=shading)
            axes[i, j].set_xlim(left=-0.5, right=1.5)
            axes[i, j].set_ylim(bottom=-1, top=1)
    else: # regular grid
        for i, j in np.ndindex(fields.shape[:2]):
            axes[i, j].imshow(fields[i, j])        
    
    if title:fig.suptitle(title)
    if show: fig.show()
    return fig

def interp_scalar_field(tri: Triangulation, field:np.ndarray, size=(512, 512), xlim=(-0.5, 1.5), ylim=(-1, 1)) -> np.ndarray:
    """

    """
    x = np.linspace(xlim[0], xlim[1], size[0])
    y = np.linspace(ylim[0], ylim[1], size[1])
    XX, YY = np.meshgrid(x,  y)
    interp = LinearTriInterpolator(tri, field)
    return interp(XX, -YY)

def generate_training_data(airfoil_dir, output_dir, config_file='solve.cfg'):
    
    def modify_config(Mach, AoA):
        with open(config_file, 'r') as f:
            lines = f.readlines()
        
        with open(config_file, 'w') as f:
            new_lines = []
            for line in lines:
                if line.startswith('MACH_NUMBER'):
                    line = f'MACH_NUMBER= {Mach}\n'
                if line.startswith('AOA'):
                    line = f'AOA= {AoA}\n'
                new_lines.append(line)
            f.writelines(new_lines)

    import os, shutil
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(airfoil_dir):
        name = file_name.split('.')[0]
        save_path = os.path.join(output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        profile = read_dat(os.path.join(airfoil_dir, file_name))
        generate_mesh(profile, 'mesh.su2')
        for Mach in np.linspace(0.2, 0.5, 5):
            for AoA in np.linspace(0, 10, 11):
                modify_config(Mach, AoA)
                os.system(f'SU2_CFD {config_file}')
                
                shutil.move('solution.csv', os.path.join(save_path, f'solution_{Mach}_{AoA}.csv'))
                shutil.move('flow.szplt', os.path.join(save_path, f'flow_{Mach}_{AoA}.szplt'))
        os.rename('mesh.su2' , os.path.join(save_path, 'mesh.su2'))
    os.remove('history.dat')