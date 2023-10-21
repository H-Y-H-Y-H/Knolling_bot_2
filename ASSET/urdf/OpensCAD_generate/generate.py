import numpy as np
from collections import Counter

# with open('data.txt', 'r') as file:
#     data = file.read()
#     data = data.replace(',', ' ')
#     data = list(data.split())
#     data = np.array([float(i) for i in data]).reshape(-1, 5)
# print(data)
# np.savetxt('data.txt', data)

data = np.load('data.txt', allow_pickle=True)

occurrences = Counter(data)
for cube, count in occurrences.items():
    print(cube, ":", count)

import subprocess

# Read the text file and parse cube sizes
with open("data.txt", "r") as file:
    cube_sizes = [list(map(float, line.strip().split())) for line in file]

# Generate OpenSCAD code dynamically
openscad_code = ""
for i, size in enumerate(cube_sizes):
    size = np.round(size[2:4], decimals=3)
    length = size[0] * 1000
    width = size[1] * 1000
    height = 12
    print(int(length), int(width), int(height))
    openscad_code = f"""
        cube([{int(length)}, {int(width)}, {int(height)}], center = true);
    """

    # Save the generated OpenSCAD code to a file
    with open("batch.scad", "w") as file:
        file.write(openscad_code)

    # Execute OpenSCAD command to generate STL files
    output_file = f"cube_{int(length)}_{int(width)}_{int(height)}.stl"
    print(f'this is num{i}, size is {size}')
    subprocess.run(["C:/Program Files (x86)/openscad/openscad.exe", "-o", output_file, "batch.scad"])
