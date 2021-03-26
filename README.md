# Deep Learning for Fluid Modeling

This project provides a framework for leaning prediction of flow fields with different data formats.

### Requirements:

- Pytorch
- [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html)
- [SU2](https://su2code.github.io/)
- [gmsh](http://www.gmsh.info/) (can be alternatively installed by  [pygmsh](https://pypi.org/project/pygmsh/))
- Tecplot (optional, only for post processing)

### Spatial Discretization (Data Format) Support:

- Regular Grids
- Unstructured (Triangular) Mesh
- (#TODO) Structured (Quadrilateral) Mesh

### Utilities:

- Generate mesh for airfoil specified by coordinate files.
- Perform simulation using SU2 for data generation.
- Functions for interpolating and visualizing fields.
- Convenient training and logging with  [Pytorch Ligntning](https://pytorchlightning.ai/).