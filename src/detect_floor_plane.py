import open3d as o3d
from open3d.visualization import draw_geometries
import numpy as np
import copy
import matplotlib.pyplot as plt



def detect_planar_patch(pcd):
    """Detects planes in pointcloud. Works quite well"""
    
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=30 ))

    # using all defaults
    oboxes = pcd.detect_planar_patches(
        normal_variance_threshold_deg=60,
        coplanarity_deg=75,
        outlier_ratio=0.75,
        min_plane_edge_length=2000,
        min_num_points=1000,
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    print("Detected {} patches".format(len(oboxes)))

    geometries = []
    largest_area = 0
    largest_plane = None
    box = None

    all_geometries = {"planes":[], "boxes":[], "pcd":[]}
    for obox in oboxes:
        mesh = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obox, scale=[1, 1, 0.0001])
        mesh.paint_uniform_color(obox.color)
        if mesh.get_surface_area()>largest_area:
            largest_area = mesh.get_surface_area()
            print(largest_area)
            largest_plane = copy.deepcopy(mesh)
            box = copy.deepcopy(obox)
        all_geometries["planes"].append(mesh)
        all_geometries["boxes"].append(obox)
    all_geometries["pcd"].append(pcd)
    geometries.append(largest_plane)
    geometries.append(box)
    geometries.append(pcd)

    o3d.io.write_triangle_mesh("C:/Users/adamf/Codes/Mocap_process/Alligned_clouds/floor_plane.obj", largest_plane, write_ascii = True)
    o3d.io.write_line_set("C:/Users/adamf/Codes/Mocap_process/Alligned_clouds/floor_box.ply", box, write_ascii = True)


    draw_geometries(geometries)


def expand_floor_mesh(floor_mesh, z_thickness=0.1, visualize=True):
    """
    Expand an existing floor mesh to enclose more floor points.
    
    Args:
        floor_mesh: Existing floor triangle mesh
        z_thickness: Additional thickness to add in z-direction (in meters)
        visualize: Whether to visualize before and after
        
    Returns:
        Expanded floor mesh
    """
    # Make a copy of the input mesh
    expanded_mesh = copy.deepcopy(floor_mesh)
    
    vertices = np.asarray(expanded_mesh.vertices)
    top_vertices_mask = np.zeros((vertices.shape[0],), dtype=bool)
    bottom_vertices_mask = np.zeros((vertices.shape[0],), dtype=bool)
    
    corners = []
    z_list = []
    for i,vertice in enumerate(vertices):
        xy = (vertice[0], vertice[1])
        z = vertice[2]
        if not xy in corners:
            print(xy)
            corners.append(xy)
            z_list.append(z)
        else:
            index = corners.index(xy)
            if z>z_list[index]:
                top_vertices_mask[i] = True
                bottom_vertices_mask[index] = True
            else:
                bottom_vertices_mask[i] = True
                top_vertices_mask[index] = True

    print(top_vertices_mask)
    print(bottom_vertices_mask)
    print(vertices)
            
    
    print(corners)
    # Get mesh center and bounds
    center = expanded_mesh.get_center()
    min_bound = expanded_mesh.get_min_bound()
    max_bound = expanded_mesh.get_max_bound()
    

    # Now adjust z-bounds by moving vertices
    
    tol = 100
    
    # Find vertices near the top and bottom z-bounds
    #top_vertices_mask = np.isclose(vertices[:, 2], max_bound[0], atol=tol)
    #bottom_vertices_mask = np.isclose(vertices[:, 2], min_bound[1], atol=tol)
    print(top_vertices_mask)
    print(bottom_vertices_mask)
    # Move top vertices up and bottom vertices down
    vertices[top_vertices_mask, 2] += z_thickness
    vertices[bottom_vertices_mask, 2] -= z_thickness
    
    # Update mesh vertices
    expanded_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Recompute normals
    expanded_mesh.compute_vertex_normals()
    
    if visualize:
        # Visualize before and after
        original_mesh = copy.deepcopy(floor_mesh)
        original_mesh.paint_uniform_color([1, 0, 0])  # Red
        expanded_mesh.paint_uniform_color([0, 1, 0])  # Green
        
        print("Red: Original mesh, Green: Expanded mesh")
        o3d.visualization.draw_geometries([original_mesh, expanded_mesh])
    expanded_mesh.paint_uniform_color([0,0,0])
    return expanded_mesh

cloud = o3d.io.read_point_cloud("/home/adamfi/Codes/Mocap_process/Alligned_clouds/ICP_reged.ply")
plane = o3d.io.read_triangle_mesh("/home/adamfi/Codes/Mocap_process/Alligned_clouds/floor_plane.obj")

expanded_mesh = expand_floor_mesh(plane, 50)

o3d.visualization.add_point_light(position=[0, 0, 10], color=[1, 1, 1], intensity=1000)

o3d.visualization.draw_geometries([cloud, expanded_mesh])
#detect_planar_patch(cloud)