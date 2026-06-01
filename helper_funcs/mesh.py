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


def get_boundaries(meshname, group_tag=99):
    gmsh.initialize()
    gmsh.open(meshname)
    # Get all boundary nodes (those on the edges of the surface)
    entities = gmsh.model.getEntitiesForPhysicalGroup(1, group_tag)
    # print(f"Entities in Physical Group {group_tag} (Boundary Edges):", entities)
    # Get nodes from those entities
    boundary_node_tags = set()
    for edge_tag in entities:
        nodes = gmsh.model.mesh.getNodes(1, edge_tag)[0]
        boundary_node_tags.update(nodes)

    dr = np.array(sorted(boundary_node_tags)).astype(int) - 1

    gmsh.finalize()
    return dr


def get_boundary_edges(meshname, group_tag=99):
    gmsh.initialize()
    gmsh.open(meshname)

    entities = gmsh.model.getEntitiesForPhysicalGroup(1, group_tag)
    edges = []

    for edge_tag in entities:
        element_types, _, element_nodes = gmsh.model.mesh.getElements(1, edge_tag)
        for elem_type, nodes in zip(element_types, element_nodes):
            _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(elem_type)
            nodes = np.array(nodes, dtype=int).reshape(-1, num_nodes)
            # Use the first two nodes as the line endpoints (works for linear and higher-order lines).
            for n0, n1 in nodes[:, :2]:
                edges.append([n0 - 1, n1 - 1])

    gmsh.finalize()
    return np.array(edges, dtype=int)