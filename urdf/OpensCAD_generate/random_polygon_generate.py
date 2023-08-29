import numpy as np
from scipy.spatial import ConvexHull
import subprocess
import os
from urdfpy import URDF
import shutil

def generate(path, start_evaluation, end_evaluation):
    # Generate OpenSCAD code dynamically
    num_vertices_max = 40
    x_range = np.array([-0.025, 0.025])
    y_range = np.array([-0.02, 0.02])


    total_point = []
    for i in range(start_evaluation, end_evaluation):
        num_vertices = 6
        x_data = np.random.uniform(x_range[0], x_range[1], num_vertices_max)
        y_data = np.random.uniform(y_range[0], y_range[1], num_vertices_max)
        points = np.concatenate((x_data.reshape(num_vertices_max, 1), y_data.reshape(num_vertices_max, 1)), axis=1)
        hull = ConvexHull(points)
        convex_points = points[hull.vertices[:num_vertices]]
        total_point = np.append(total_point, convex_points).reshape(-1, num_vertices * 2)
        openscad_code = f"""
                        p1 = [{convex_points[0, 0]}, {convex_points[0, 1]}];
                        p2 = [{convex_points[1, 0]}, {convex_points[1, 1]}];
                        p3 = [{convex_points[2, 0]}, {convex_points[2, 1]}];
                        p4 = [{convex_points[3, 0]}, {convex_points[3, 1]}];
                        p5 = [{convex_points[4, 0]}, {convex_points[4, 1]}];
                        p6 = [{convex_points[5, 0]}, {convex_points[5, 1]}];
                        points = [p1, p2, p3, p4, p5, p6];
                        linear_extrude(height=0.01)
                        polygon(points);
                        """
        with open("random_polygon.scad", "w") as file:
            file.write(openscad_code)
        output_file = path + f"polygon_{i}.stl"
        print(f'this is num{i}')
        command = ['openscad', '-o', output_file, '--export-format=binstl', 'random_polygon.scad']
        subprocess.run(command)
    np.savetxt(path + 'points_%s_%s.txt' % (start_evaluation, end_evaluation), total_point)

def stl2urdf(start, end, tar_path):

    total_data = np.loadtxt(tar_path + 'points_%s_%s.txt' % (start, end))

    for i in range(start, end):
        points_data = total_data[i - start]
        # temp = URDF.load('../OpensCAD_generate/template.urdf')
        # temp.name = 'polygon_%s' % i
        # temp.base_link.visuals[0].geometry.mesh.filename = 'polygon_%s.stl' % i
        # temp.base_link.collisions[0].geometry.mesh.filename = 'polygon_%s.stl' % i
        # temp.save(tar_path + 'polygon_%s.urdf' % i)
        shutil.copy('../OpensCAD_generate/template.urdf', tar_path + 'polygon_%s.urdf' % i)
        pass
    for i in range(start, end):
        with open(tar_path + 'polygon_%s.urdf' % i, "r") as file:
            data = file.read()
            new_data = data.replace('polygon_0.stl', 'polygon_%d.stl' % i)
        with open(tar_path + 'polygon_%s.urdf' % i, "w") as file:
            file.write(new_data)

if __name__ == '__main__':

    path = '../../../knolling_dataset/random_polygon/'
    os.makedirs(path, exist_ok=True)
    start_evaluation = 400
    end_evaluation = 600
    generate(path, start_evaluation, end_evaluation)

    tar_path = '../../../knolling_dataset/random_polygon/'
    os.makedirs(tar_path, exist_ok=True)
    start = 400
    end = 600

    stl2urdf(start, end, tar_path)