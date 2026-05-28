import gmsh
import numpy as np

def get_plist_tlist_from_gmsh(meshname):
    gmsh.initialize()

    gmsh.open(meshname)
    nodeTags, coords, parametricCoord = gmsh.model.mesh.getNodes()
    plist = np.array(coords).reshape(-1, 3)[:, :2]
    tp, nm, el = gmsh.model.mesh.getElements(2, -1)
    tlist = np.array(el[0].reshape(-1, 3)) - 1

    gmsh.finalize()
    return plist, tlist


def get_boundaries(meshname):
    gmsh.initialize()
    gmsh.open(meshname)
    # Get all boundary nodes (those on the edges of the surface)
    entities = gmsh.model.getEntitiesForPhysicalGroup(1, 99)
    # print("Entities in Physical Group 99 (Boundary Edges):", entities)
    # Get nodes from those entities
    boundary_node_tags = set()
    for edge_tag in entities:
        nodes = gmsh.model.mesh.getNodes(1, edge_tag)[0]
        boundary_node_tags.update(nodes)

    dr = np.array(sorted(boundary_node_tags)).astype(int) - 1

    gmsh.finalize()
    return dr
