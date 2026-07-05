# Boundary conditions
## Differential equations
The differential equations that we want to solve are of the form:
$$

-\frac{\partial}{\partial x}\left(\alpha_1 \frac{\partial \Phi(x,y)}{\partial x}\right)
-\frac{\partial}{\partial y}\left(\alpha_2 \frac{\partial \Phi(x,y)}{\partial y}\right)
+\beta\Phi
=f(x,y)

$$
or 
$$
-div(\alpha \, grad(\Phi)) - \beta \Phi = f(x,y)
$$

## Dirichlet boundary conditions
Dirchlet boundary conditions are of the form:
$$
\Phi\big|_{\Gamma}=[some \, number]
$$

It basically specifies the value of the solution at the boundary.
## Robin boundary conditions
Robin boundary conditions are of the form
$$
\left(\alpha_1 \frac{\partial \Phi}{\partial x},
\alpha_2 \frac{\partial \Phi}{\partial y}\right)\cdot\vec{n}
+\gamma\Phi\big|_{\Gamma}=q
$$
It is basically the derivative of the solution at the boundary.


# Mesh generation
Mesh generation using gmsh and gmshtools.
## Generating mesh
### Creating the geometry 
Create all forms of the geometry (individual circles, rectangles,...) even if they overlap.
### Creating the surfaces
Create a main surface that contains all the other surfaces with
```python
main_surf = gmsh.model.occ.addPlaneSurface([plate_loop, other_loops])
```
`plate_loop` is the loop of the main surface and `other_loops` is a list of loops of the other surfaces. These will be subtracted from the main surface.
For more info, check the [`2D validation script`](./validations/validate_2d_cpp.py).
### Fragment the geometry
> See [`WS1920`](./uebungen/klausuren/WS1920/main.py) for an example of how to use the fragment function.

Fragment to solve the problem of overlapping surfaces. Kind of like embed, but you don't have to define each single point.

First create a list of all surfaces/lines that need to be fragmented. The first element of the tuple is the dimension of the object (1 for lines, 2 for surfaces) and the second element is the tag of the object.
```python
stuff_to_fragment = [
    (1, line1),
    (2, surf0),
    (1, line2),
    (2, surfXX),
]
```
Then we just fragment the geometry with the main surface and all other surfaces.
```python
out_dimtags, out_map = gmsh.model.occ.fragment([(2, main_surf)], stuff_to_fragment)
```

Now, remap old tags to new tags. The `out_map` is a dictionary that maps the old tags to the new tags. The keys are tuples of the form `(dim, old_tag)` and the values are lists of tuples of the form `(dim, new_tag)`. For example, if you have a surface with tag 1 that was fragmented into two surfaces with tags 2 and 3, then `out_map[(2, 1)]` will be `[(2, 2), (2, 3)]`.
```python
def new_tags(tool_idx):
    return [t for _, t in out_map[1 + tool_idx]]
```

Then, we can get the new tags of the surfaces and lines that we put in the `stuff_to_fragmant` list. For example, if we want to get the new tags of the first surface in the `stuff_to_fragmant` list, we can do:
```python
new_line1 = new_tags(0) # 0 because it is in the top of the stuff_to_fragmant list
new_surf0 = new_tags(1)
# and so on...
```

If we want the boundaries of the new surfaces, we can use the `get_boundary` function.
```python
def boundary_curves(surf_tags):
    """Get all unique 1D boundary curve tags for a list of surface tags."""
    curves = set()
    for s in surf_tags:
        for _, ct in gmsh.model.getBoundary([(2, s)], oriented=False, combined=True):
            curves.add(abs(ct))
    return list(curves)

```
Then you can get the boundaries of the new surfaces like this to e.g. add a physical group to them:
```python
gmsh.model.addPhysicalGroup(1, boundary_curves(new_surf0), tag=20, name="Surf0Boundary")
```