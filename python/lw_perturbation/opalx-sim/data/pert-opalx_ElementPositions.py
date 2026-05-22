import os, sys, argparse, math, base64, zlib, struct

if sys.version_info < (3,0):
    range = xrange

vertices = []
numVertices = [74]
vertices_base64 = "eJxtk09IVFEUxicCR/EpiWdeMo2oCANhCpL9k6F7iWpTpM2mgoowWxREYdOigmCaTTUDFQW1E1tkJO1iApFuRH9wIiYosihKkMhNMAtrIBkqvN95zLn4NsP73p1zv9853wmFlntKxv7OKldfHUvnsplv6sqZ+ovR/rxiPdad3Pa76av6VX81dX/uRaBP7N4/fz7/Wf1deoqBvmWwePdU5qOK+Ynk4xvvA70V7/yd9RVLv2/VA9RjvYz71uB+1nPwx35ZDx+k9fF76YTkK5nL9rxxz4Mn4GO9YnkCPtaj1r9JOP432HfT4fCO2/+bMPhYb7H3mbAn+xmx/kze6T94TDo79PJ5XyER+MH5AfCx3oH6H8DH+iP4mQUf6zvhvwF8rHfhvRvfWW+yPGYM9VhvszyGcD/rP+CvGX5ZH5ufvnVz/NMU8wU5AX/MOf8EeYg79QvI36TjZyXmscfx34n5bXJ472De7eBjvRH52OrJfq5CntamZP9Dyz6kwame7ug9OhT9PsV6G+p8GameO+ku3Dsq9o70M/h8KPaO9AFwlSPVOSS9HX1oEbkl3WM51XWRc9KHkMs6sRekT1hO1SD2iHTv7crinyM/nb0jPWl5TLtz/hpy2ePUH0UuXzl+SpbHHHb8N2Ouu3zJm0UONoKP9Rrk5qQn++khZ7mU7D94zODCcGH6XSnBeifOnxupnjvpPtQ/LfaOdBF+Lom9I30c/uci1TkkvQ+5rBW5/T9Hy2MyIuekc5xLsRek3yCXdWKPSNe2vvb98oKzd6Q3I39x5/xe5K/fqX8M+Ztx/MwgfynHf2XAzm/Yl7wXMO8k+FhfXGfzMeHJftYgT41nZf//AScGaG8="
triangles = [[ 0, 2, 1, 1, 2, 38, 2, 39, 38, 0, 3, 2, 2, 3, 39, 3, 40, 39, 0, 4, 3, 3, 4, 40, 4, 41, 40, 0, 5, 4, 4, 5, 41, 5, 42, 41, 0, 6, 5, 5, 6, 42, 6, 43, 42, 0, 7, 6, 6, 7, 43, 7, 44, 43, 0, 8, 7, 7, 8, 44, 8, 45, 44, 0, 9, 8, 8, 9, 45, 9, 46, 45, 0, 10, 9, 9, 10, 46, 10, 47, 46, 0, 11, 10, 10, 11, 47, 11, 48, 47, 0, 12, 11, 11, 12, 48, 12, 49, 48, 0, 13, 12, 12, 13, 49, 13, 50, 49, 0, 14, 13, 13, 14, 50, 14, 51, 50, 0, 15, 14, 14, 15, 51, 15, 52, 51, 0, 16, 15, 15, 16, 52, 16, 53, 52, 0, 17, 16, 16, 17, 53, 17, 54, 53, 0, 18, 17, 17, 18, 54, 18, 55, 54, 0, 19, 18, 18, 19, 55, 19, 56, 55, 0, 20, 19, 19, 20, 56, 20, 57, 56, 0, 21, 20, 20, 21, 57, 21, 58, 57, 0, 22, 21, 21, 22, 58, 22, 59, 58, 0, 23, 22, 22, 23, 59, 23, 60, 59, 0, 24, 23, 23, 24, 60, 24, 61, 60, 0, 25, 24, 24, 25, 61, 25, 62, 61, 0, 26, 25, 25, 26, 62, 26, 63, 62, 0, 27, 26, 26, 27, 63, 27, 64, 63, 0, 28, 27, 27, 28, 64, 28, 65, 64, 0, 29, 28, 28, 29, 65, 29, 66, 65, 0, 30, 29, 29, 30, 66, 30, 67, 66, 0, 31, 30, 30, 31, 67, 31, 68, 67, 0, 32, 31, 31, 32, 68, 32, 69, 68, 0, 33, 32, 32, 33, 69, 33, 70, 69, 0, 34, 33, 33, 34, 70, 34, 71, 70, 0, 35, 34, 34, 35, 71, 35, 72, 71, 0, 36, 35, 35, 36, 72, 36, 73, 72, 0, 1, 36, 36, 1, 73, 1, 38, 73, 37, 38, 39, 37, 39, 40, 37, 40, 41, 37, 41, 42, 37, 42, 43, 37, 43, 44, 37, 44, 45, 37, 45, 46, 37, 46, 47, 37, 47, 48, 37, 48, 49, 37, 49, 50, 37, 50, 51, 37, 51, 52, 37, 52, 53, 37, 53, 54, 37, 54, 55, 37, 55, 56, 37, 56, 57, 37, 57, 58, 37, 58, 59, 37, 59, 60, 37, 60, 61, 37, 61, 62, 37, 62, 63, 37, 63, 64, 37, 64, 65, 37, 65, 66, 37, 66, 67, 37, 67, 68, 37, 68, 69, 37, 69, 70, 37, 70, 71, 37, 71, 72, 37, 72, 73, 37, 73, 38]]
decoration = [[ 0.000000, 0.000000, -1.000000, 0.000000, 0.000000, 5.000000]]

color = [0]

index_base64 = 'eJylVlGL4zYQft9foXMoeCGRkyulx24SyOYWbiHbW/aWwrX0QZHHsQ5bSiU5ubTcf+9I8ia2Ex8tNcSWNDPffDMjjTJ98/7j8uXz0z3JbVnMr6bhQ/CZ5sDSMPTTEixDLbsdwZ+V2M2ipZIWpB29HLYQER5ms8jCV5s4mFvCc6YN2Flls9G7iCTzqxOcFbaA+R1bHwol6RdDDCu3BSBOCtMkSE/ab0Yj0tAdjRoyw7XYWmI0n0WOn7lJEp5Kug76Xwzlqkw4k9IbR/NpEkz+M8b6yOAyiD00SbvH5WFI1io9kL9bAveoHeisUPsbkos0BXl7prEXqc1vyGQ8/uFcmIPY5LZPWjK9EfKGjM9FW5amQm7OZN+uWtOBBpmCXjK5Y+YC/f9BzqqK5yPGrVBIEesCXSavI0zzKavTJOzJqUtovcQDPZHOoiZfV6AgqvW61doxTWrbGUkVr0rcvXQD9r4AN7w7PKRxG/L6tmUNEhMMaC1hT+4Wd59XH3+h934xDshDYnUFaNb2qoFZ+MTBG2eV9Gkg8XUnx07X1FpNF94yDt6b2O5JkoWvLWGkcAU4y7wD9ZIO6AcohdnmoAVfOXEcuZVo2FL6FbhV+sd4PCSTIRlfDwO/6/MCexdUuJZghD2gszF916eWiiyrTDfMpSqcrwl13upXr6eNVpVMvcllmDH9GRnXr14YswVeFawXw0Vdv/ozL8lCc/KsLFaZLFkJml0sA/eijis0DZbBMI7CN3I+MQePzOb06WFIfsJJpyz0N9Aq7ilKcEaZtYznrnNrVRx3acYKc6mKtdE+ByieNHBh3EadkbfjTvAunPJQgslROhi8X7wsBoPbTn7ISw4kE9pY7EAaoS34I0jWQLD8KXYF4gogsgPZ54LnxAPiqii3SltKPoAGsod6TlhReBUwLU+tc7JSDA8wffAWj6gcR5hL9wt862wNG+cQy/HoUbsHsg7jE1hiMRSLHRaHKvOzuprI1h7DDDwxMuepL7s1jN8Gwe/v4z86+T3LNW4QLVgx6bYGy2TKdPpYy2N/G1caJlH/WT2FdoSle6EhczVCB66HnVs12dJXQ9Q+YvxLE+oajxE7+O7hfev2f3h1IvjWPYgaMGAZoj2p4t3WasKvjbXRjOMmUmivVFfy2V8BK6W2cX+v9nA03BZxg2GLHe6dZzDiLziu7IVM1Z7ifXy/wztnJQw2TETAe8fpRcPvXA+vDL1m16f7nv6hTJNwYU79/7L5P+Qvs/k='

def decodeVertices():
    vertices_binary = zlib.decompress(base64.b64decode(vertices_base64))
    k = 0
    for i in range(len(numVertices)):
        current = []
        for j in range(numVertices[i] * 3):
            current.append(float(struct.unpack('=d', vertices_binary[k:k+8])[0]))
            k += 8
        vertices.append(current)

def showVTK():
    try:
        import numpy as np
        import pyvista as pv
    except ModuleNotFoundError:
        sys.stderr.write("PyVista is not installed.\n")
        sys.stderr.write("Install it in the local virtual environment with:\n")
        sys.stderr.write("  source /Users/adelmann/git/opalx/.venv-h6/bin/activate\n")
        sys.stderr.write("  python -m pip install pyvista\n")
        return 1

    vtk_file = "pert-opalx_ElementPositions.vtk"
    script_file = os.path.abspath(__file__)
    needs_export = not os.path.exists(vtk_file)
    if not needs_export:
        needs_export = os.path.getmtime(script_file) > os.path.getmtime(vtk_file)
    if needs_export:
        exportVTK()

    mesh = pv.read(vtk_file)
    plotter = pv.Plotter()
    add_mesh_kwargs = {}
    active_scalars_name = mesh.active_scalars_name
    if active_scalars_name is not None:
        active_scalars = mesh.active_scalars
        if getattr(active_scalars, 'ndim', 0) == 2 and active_scalars.shape[1] in (3, 4):
            add_mesh_kwargs['scalars'] = active_scalars_name
            add_mesh_kwargs['rgb'] = True
            add_mesh_kwargs['preference'] = 'cell'
            unique_colors = active_scalars[:, :3].astype(float)
            if np.nanmax(unique_colors) > 1.0:
                unique_colors /= 255.0
            unique_colors = np.unique(unique_colors, axis=0)
            legend = [
                ('Other', (0.5, 0.5, 0.5)),
                ('Dipole', (1.0, 0.847, 0.0)),
                ('Quadrupole', (1.0, 0.0, 0.0)),
                ('Sextupole', (0.537, 0.745, 0.525)),
                ('Octupole', (0.5, 0.5, 0.0)),
                ('Solenoid', (1.0, 138.0 / 255.0, 0.0)),
                ('RFCavity', (1.0, 1.0, 0.0)),
                ('TravelingWave', (0.0, 0.6, 0.0)),
                ('Drift', (0.0, 0.0, 1.0)),
            ]
            present = []
            for label, color in legend:
                color_arr = np.asarray(color, dtype=float)
                if np.any(np.all(np.isclose(unique_colors, color_arr, atol=5e-3), axis=1)):
                    present.append((label, color))
            if present:
                plotter.add_legend(present, bcolor=(1.0, 1.0, 1.0), face='circle', border=True, size=(0.2, 0.24))
    plotter.add_mesh(mesh, **add_mesh_kwargs)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
    return 0

def dot(a, b):
    return sum(a[i]*b[i] for i in range(len(a)))

def normalize(a):
    length = math.sqrt(dot(a, a))
    for i in range(3):
        a[i]/=length
    return a

def distance(a, b):
    c = [0] * 2
    for i in range(len(a)):
        c[i] = a[i] - b[i]
    return math.sqrt(dot(c, c))

def cross(a, b):
    c = [0, 0, 0]
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

class Quaternion:
    def __init__(self, values):
        self.R = values

    def __str__(self):
        return str(self.R)

    def __mul__(self, other):
        imagThis = self.imag()
        imagOther = other.imag()
        cro = cross(imagThis, imagOther)

        ret = ([self.R[0] * other.R[0] - dot(imagThis, imagOther)] +
               [self.R[0] * imagOther[0] + other.R[0] * imagThis[0] + cro[0],
                self.R[0] * imagOther[1] + other.R[0] * imagThis[1] + cro[1],
                self.R[0] * imagOther[2] + other.R[0] * imagThis[2] + cro[2]])

        return Quaternion(ret)

    def imag(self):
        return self.R[1:]

    def conjugate(self):
        ret = [0] * 4
        ret[0] = self.R[0]
        ret[1] = -self.R[1]
        ret[2] = -self.R[2]
        ret[3] = -self.R[3]

        return Quaternion(ret)

    def inverse(self):
        return self.conjugate()

    def rotate(self, vec3D):
        quat = Quaternion([0.0] + vec3D)
        conj = self.conjugate()

        ret = self * (quat * conj)
        return ret.R[1:]

def getQuaternion(u, ref):
    u = normalize(u)
    ref = normalize(ref)
    axis = cross(u, ref)
    normAxis = math.sqrt(dot(axis, axis))

    if normAxis < 1e-12:
        if math.fabs(dot(u, ref) - 1.0) < 1e-12:
            return Quaternion([1.0, 0.0, 0.0, 0.0])

        return Quaternion([0.0, 1.0, 0.0, 0.0])

    axis = normalize(axis)
    cosAngle = math.sqrt(0.5 * (1 + dot(u, ref)))
    sinAngle = math.sqrt(1 - cosAngle * cosAngle)

    return Quaternion([cosAngle, sinAngle * axis[0], sinAngle * axis[1], sinAngle * axis[2]])

def exportVTK():
    vertices_str = ""
    triangles_str = ""
    color_str = ""
    cellTypes_str = ""
    startIdx = 0
    vertexCounter = 0
    cellCounter = 0
    lookup_table = []
    lookup_table.append([0.5, 0.5, 0.5, 0.5])
    lookup_table.append([1.0, 0.847, 0.0, 1.0])
    lookup_table.append([1.0, 0.0, 0.0, 1.0])
    lookup_table.append([0.537, 0.745, 0.525, 1.0])
    lookup_table.append([0.5, 0.5, 0.0, 1.0])
    lookup_table.append([1.0, 0.5411764706, 0.0, 1.0])
    lookup_table.append([1.0, 1.0, 0.0, 1.0])
    lookup_table.append([0.0, 0.6, 0.0, 1.0])
    lookup_table.append([0.0, 0.0, 1.0, 1.0])

    decodeVertices()

    for i in range(len(vertices)):
        for j in range(0, len(vertices[i]), 3):
            vertices_str += ("%f %f %f\n" %(vertices[i][j], vertices[i][j+1], vertices[i][j+2]))
            vertexCounter += 1

        for j in range(0, len(triangles[i]), 3):
            triangles_str += ("3 %d %d %d\n" % (triangles[i][j] + startIdx, triangles[i][j+1] + startIdx, triangles[i][j+2] + startIdx))
            cellTypes_str += "5\n"
            tmp_color = lookup_table[color[i]]
            color_str += ("%f %f %f %f\n" % (tmp_color[0], tmp_color[1], tmp_color[2], tmp_color[3]))
            cellCounter += 1
        startIdx = vertexCounter

    fh = open('pert-opalx_ElementPositions.vtk','w')
    fh.write("# vtk DataFile Version 2.0\n")
    fh.write("test\nASCII\n\n")
    fh.write("DATASET UNSTRUCTURED_GRID\n")
    fh.write("POINTS " + str(vertexCounter) + " float\n")
    fh.write(vertices_str)
    fh.write("CELLS " + str(cellCounter) + " " + str(cellCounter * 4) + "\n")
    fh.write(triangles_str)
    fh.write("CELL_TYPES " + str(cellCounter) + "\n")
    fh.write(cellTypes_str)
    fh.write("CELL_DATA " + str(cellCounter) + "\n")
    fh.write("COLOR_SCALARS type 4\n")
    fh.write(color_str + "\n")
    fh.close()

def getNormal(tri_vertices):
    vec1 = [tri_vertices[1][0] - tri_vertices[0][0],
            tri_vertices[1][1] - tri_vertices[0][1],
            tri_vertices[1][2] - tri_vertices[0][2]]
    vec2 = [tri_vertices[2][0] - tri_vertices[0][0],
            tri_vertices[2][1] - tri_vertices[0][1],
            tri_vertices[2][2] - tri_vertices[0][2]]
    return normalize(cross(vec1,vec2))

def exportWeb(bgcolor):
    lookup_table = []
    lookup_table.append([0.5, 0.5, 0.5])
    lookup_table.append([1.0, 0.847, 0.0])
    lookup_table.append([1.0, 0.0, 0.0])
    lookup_table.append([0.537, 0.745, 0.525])
    lookup_table.append([0.5, 0.5, 0.0])
    lookup_table.append([1.0, 0.5411764706, 0.0])
    lookup_table.append([1.0, 1.0, 0.0])
    lookup_table.append([0.0, 0.6, 0.0])
    lookup_table.append([0.0, 0.0, 1.0])

    decodeVertices()

    mesh = "'data:"
    mesh += "{"
    mesh += "\"autoClear\":true,"
    mesh += "\"clearColor\":[0.0000,0.0000,0.0000],"
    mesh += "\"ambientColor\":[0.0000,0.0000,0.0000],"
    mesh += "\"gravity\":[0.0000,-9.8100,0.0000],"
    mesh += "\"cameras\":["
    mesh += "{"
    mesh += "\"name\":\"Camera\","
    mesh += "\"id\":\"Camera\","
    mesh += "\"position\":[21.7936,2.2312,-85.7292],"
    mesh += "\"rotation\":[0.0432,-0.1766,-0.0668],"
    mesh += "\"fov\":0.8578,"
    mesh += "\"minZ\":10.0000,"
    mesh += "\"maxZ\":10000.0000,"
    mesh += "\"speed\":1.0000,"
    mesh += "\"inertia\":0.9000,"
    mesh += "\"checkCollisions\":false,"
    mesh += "\"applyGravity\":false,"
    mesh += "\"ellipsoid\":[0.2000,0.9000,0.2000]"
    mesh += "}],"
    mesh += "\"activeCamera\":\"Camera\","
    mesh += "\"lights\":["
    mesh += "{"
    mesh += "\"name\":\"Lamp\","
    mesh += "\"id\":\"Lamp\","
    mesh += "\"type\":0.0000,"
    mesh += "\"position\":[4.0762,34.9321,-63.5788],"
    mesh += "\"intensity\":1.0000,"
    mesh += "\"diffuse\":[1.0000,1.0000,1.0000],"
    mesh += "\"specular\":[1.0000,1.0000,1.0000]"
    mesh += "}],"
    mesh += "\"materials\":[],"
    mesh += "\"meshes\":["

    for i in range(len(triangles)):
        vertex_list = []
        indices_list = []
        normals_list = []
        color_list = []
        for j in range(0, len(triangles[i]), 3):
            tri_vertices = []
            idcs = triangles[i][j:j + 3]
            tri_vertices.append(vertices[i][3 * idcs[0]:3 * (idcs[0] + 1)])
            tri_vertices.append(vertices[i][3 * idcs[1]:3 * (idcs[1] + 1)])
            tri_vertices.append(vertices[i][3 * idcs[2]:3 * (idcs[2] + 1)])
            indices_list.append(','.join(str(n) for n in range(len(vertex_list),len(vertex_list) + 3)))
            # left hand order!
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[0]))
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[2]))
            vertex_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in tri_vertices[1]))
            normal = getNormal(tri_vertices)
            normals_list.append(','.join("%.5f" % (round(n,5) + 0.0) for n in normal * 3))
            color_list.append(','.join([str(n) for n in lookup_table[color[i]]] * 3))
        mesh += "{"
        mesh += "\"name\":\"element_" + str(i) + "\","
        mesh += "\"id\":\"element_" + str(i) + "\","
        mesh += "\"position\":[0.0,0.0,0.0],"
        mesh += "\"rotation\":[0.0,0.0,0.0],"
        mesh += "\"scaling\":[1.0,1.0,1.0],"
        mesh += "\"isVisible\":true,"
        mesh += "\"isEnabled\":true,"
        mesh += "\"useFlatShading\":false,"
        mesh += "\"checkCollisions\":false,"
        mesh += "\"billboardMode\":0,"
        mesh += "\"receiveShadows\":false,"
        mesh += "\"positions\":[" + ','.join(vertex_list) + "],"
        mesh += "\"normals\":[" + ','.join(normals_list) + "],"
        mesh += "\"indices\":[" + ','.join(indices_list) + "],"
        mesh += "\"colors\":[" + ','.join(color_list) + "],"
        mesh += "\"subMeshes\":["
        mesh += "{"
        mesh += "\"materialIndex\":0,"
        mesh += " \"verticesStart\":0,"
        mesh += " \"verticesCount\":" + str(len(triangles[i])) + ","
        mesh += " \"indexStart\":0,"
        mesh += " \"indexCount\":" + str(len(triangles[i])) + ""
        mesh += "}]"
        mesh += "}"
        mesh += ","

        del normals_list[:]
        del vertex_list[:]
        del color_list[:]
        del indices_list[:]

    mesh = mesh[:-1] + "]"
    mesh += "}'"
    index_compressed = base64.b64decode(index_base64)
    index = zlib.decompress(index_compressed).decode('utf-8')
    if (len(bgcolor) == 3):
        mesh += ";\n            "
        mesh += "scene.clearColor = new BABYLON.Color3(%f, %f, %f)" % (bgcolor[0], bgcolor[1], bgcolor[2])

    index = index.replace('##DATA##', mesh)
    fh = open('pert-opalx_ElementPositions.html','w')
    fh.write(index)
    fh.close()

def computeMinAngle(idx, curAngle, positions, connections, check):
    matrix = [[-math.cos(curAngle), -math.sin(curAngle)], [math.sin(curAngle), -math.cos(curAngle)]]

    minAngle = 2 * math.pi
    nextIdx = -1

    for j in connections[idx]:
        direction = [positions[j][0] - positions[idx][0],
                     positions[j][1] - positions[idx][1]]
        direction = [dot(matrix[0],direction), dot(matrix[1],direction)]

        if math.fabs(dot([1.0, 0.0], direction) / distance(positions[j], positions[idx]) - 1.0) < 1e-6: continue

        angle = math.fmod(math.atan2(direction[1],direction[0]) + 2 * math.pi, 2 * math.pi)

        if angle < minAngle:
            nextIdx = j
            minAngle = angle
        if angle == minAngle and check:
            dire =  [positions[j][0] - positions[idx][0],
                     positions[j][1] - positions[idx][1]]
            minA0 = math.atan2(dire[1], dire[0])
            minA1 = computeMinAngle(nextIdx, minA0, positions, connections, False)
            minA2 = computeMinAngle(j, minA0, positions, connections, False)
            if minA2[1] < minA2[1]:
                nextIdx = j

    if nextIdx == -1:
        nextIdx = connections[idx][0]

    return (nextIdx, minAngle)

def squashVertices(positionDict, connections):
    removedItems = []
    indices = [int(k) for k in positionDict.keys()]
    idxChanges = indices
    for i in indices:
        if i in removedItems:
            continue
        for j in indices:
            if j in removedItems or j == i:
                continue

            if distance(positionDict[i], positionDict[j]) < 1e-6:
                connections[j] = list(set(connections[j]))
                if i in connections[j]:
                    connections[j].remove(i)
                if j in connections[i]:
                    connections[i].remove(j)

                connections[i].extend(connections[j])
                connections[i] = list(set(connections[i]))
                connections[i].sort()

                for k in connections.keys():
                    if k == i: continue
                    if j in connections[k]:
                        idx = connections[k].index(j)
                        connections[k][idx] = i

                idxChanges[j] = i
                removedItems.append(j)

    for i in removedItems:
        del positionDict[i]
        del connections[i]

    for i in connections.keys():
        connections[i] = list(set(connections[i]))
        connections[i].sort()

    return idxChanges

def computeLineEquations(positions, connections):
    cons = set()
    for i in connections.keys():
        for j in connections[i]:
            cons.add((min(i, j), max(i, j)))

    lineEquations = {}
    for item in cons:
        a = (positions[item[1]][1] - positions[item[0]][1])
        b = -(positions[item[1]][0] - positions[item[0]][0])

        xm = 0.5 * (positions[item[0]][0] + positions[item[1]][0])
        ym = 0.5 * (positions[item[0]][1] + positions[item[1]][1])
        c = -(a * xm +  b * ym)

        lineEquations[item] = (a, b, c)

    return lineEquations

def checkPossibleSegmentIntersection(segment, positions, lineEquations, minAngle, lastIntersection):
    item1 = (min(segment), max(segment))

    (a1, b1, c1) = (0,0,0)
    A = [0]*2
    B = A

    if segment[0] == None:
        A = lastIntersection
        B = positions[segment[1]]

        a1 = B[1] - A[1]
        b1 = -(B[0] - A[0])
        xm = 0.5 * (A[0] + B[0])
        ym = 0.5 * (A[1] + B[1])
        c1 = -(a1 * xm + b1 * ym)

    else:
        A = positions[segment[0]]
        B = positions[segment[1]]

        (a1, b1, c1) = lineEquations[item1]

        if segment[1] < segment[0]:
            (a1, b1, c1) = (-a1, -b1, -c1)

    curAngle = math.atan2(a1, -b1)
    matrix = [[-math.cos(curAngle), -math.sin(curAngle)], [math.sin(curAngle), -math.cos(curAngle)]]

    origMinAngle = minAngle

    segment1 = [B[0] - A[0], B[1] - A[1], 0.0]

    intersectingSegments = []
    distanceAB = distance(A, B)
    for item2 in lineEquations.keys():
        item = item2
        C = positions[item[0]]
        D = positions[item[1]]

        if segment[0] == None:
            if (segment[1] == item[0] or
                segment[1] == item[1]): continue
        else:
            if (item1[0] == item[0] or
                item1[1] == item[0] or
                item1[0] == item[1] or
                item1[1] == item[1]): continue

        nodes = set([item1[0],item1[1],item[0],item[1]])
        if len(nodes) < 4: continue       # share same vertex

        (a2, b2, c2) = lineEquations[item]

        segment2 = [C[0] - A[0], C[1] - A[1], 0.0]
        segment3 = [D[0] - A[0], D[1] - A[1], 0.0]

        # check that C and D aren't on the same side of AB
        if cross(segment1, segment2)[2] * cross(segment1, segment3)[2] > 0.0: continue

        if cross(segment1, segment2)[2] < 0.0 or cross(segment1, segment3)[2] > 0.0:
            (C, D, a2, b2, c2) = (D, C, -a2, -b2, -c2)
            item = (item[1], item[0])

        denominator = a1 * b2 - b1 * a2
        if math.fabs(denominator) < 1e-9: continue

        px = (b1 * c2 - b2 * c1) / denominator
        py = (a2 * c1 - a1 * c2) / denominator

        distanceCD = distance(C, D)

        distanceAP = distance(A, [px, py])
        distanceBP = distance(B, [px, py])
        distanceCP = distance(C, [px, py])
        distanceDP = distance(D, [px, py])

        # check intersection is between AB and CD
        check1 = (distanceAP - 1e-6 < distanceAB and distanceBP - 1e-6 < distanceAB)
        check2 = (distanceCP - 1e-6 < distanceCD and distanceDP - 1e-6 < distanceCD)
        if not check1 or not check2: continue

        if math.fabs(dot(segment1, [D[0] - C[0], D[1] - C[1], 0.0]) / (distanceAB * distanceCD) + 1.0) < 1e-9: continue

        if ((distanceAP < 1e-6) or
            (distanceBP < 1e-6 and distanceCP < 1e-6) or
            (distanceDP < 1e-6)):
            continue

        direction = [D[0] - C[0], D[1] - C[1]]
        direction = [dot(matrix[0], direction), dot(matrix[1], direction)]
        angle = math.fmod(math.atan2(direction[1], direction[0]) + 2 * math.pi, 2 * math.pi)

        newSegment = ([px, py], item[1], distanceAP, angle)

        if distanceCP < 1e-6 and angle > origMinAngle: continue
        if distanceBP > 1e-6 and angle > math.pi: continue

        if len(intersectingSegments) == 0:
            intersectingSegments.append(newSegment)
            minAngle = angle
        else:
            if intersectingSegments[0][2] - 1e-9 > distanceAP:
                intersectingSegments[0] = newSegment
                minAngle = angle
            elif intersectingSegments[0][2] + 1e-9 > distanceAP and angle < minAngle:
                intersectingSegments[0] = newSegment
                minAngle = angle

    return intersectingSegments

def projectToPlane(normal):
    fh = open('pert-opalx_ElementPositions.gpl','w')
    normal = normalize(normal)
    ori = getQuaternion(normal, [0, 0, 1])

    left2Right = [0, 0, 1]
    if math.fabs(math.fabs(dot(normal, left2Right)) - 1) < 1e-9:
        left2Right = [1, 0, 0]
    rotL2R = ori.rotate(left2Right)
    deviation = math.atan2(rotL2R[1], rotL2R[0])
    rotBack = Quaternion([math.cos(0.5 * deviation), 0, 0, -math.sin(0.5 * deviation)])
    ori = rotBack * ori

    decodeVertices()

    for i in range(len(vertices)):
        positions = {}
        connections = {}
        for j in range(0, len(vertices[i]), 3):
            nextPos3D = ori.rotate(vertices[i][j:j+3])
            nextPos2D = nextPos3D[0:2]
            positions[j/3] = nextPos2D

        if len(positions) == 0:
            continue

        idx = 0
        maxX = positions[0][0]
        for j in positions.keys():
            if positions[j][0] > maxX:
                maxX = positions[j][0]
                idx = j
            if positions[j][0] == maxX and positions[j][1] > positions[idx][1]:
                idx = j

        for j in range(0, len(triangles[i]), 3):
            for k in range(0, 3):
                vertIdx = triangles[i][j + k]
                if not vertIdx in connections:
                    connections[vertIdx] = []
                for l in range(1, 3):
                    connections[vertIdx].append(triangles[i][j + ((k + l) % 3)])

        numConnections = 0
        for j in connections.keys():
            connections[j] = list(set(connections[j]))
            numConnections += len(connections[j])

        numConnections /= 2
        idChanges = squashVertices(positions, connections)

        lineEquations = computeLineEquations(positions, connections)

        idx = idChanges[int(idx)]
        fh.write("%.6f    %.6f\n" % (positions[idx][0], positions[idx][1]))

        curAngle = math.pi
        origin = idx
        count = 0
        passed = []
        while (count == 0 or distance(positions[origin], positions[idx]) > 1e-9) and not count > numConnections:
            nextGen = computeMinAngle(idx, curAngle, positions, connections, False)
            nextIdx = nextGen[0]
            direction = [positions[nextIdx][0] - positions[idx][0],
                         positions[nextIdx][1] - positions[idx][1]]
            curAngle = math.atan2(direction[1], direction[0])

            intersections = checkPossibleSegmentIntersection((idx, nextIdx), positions, lineEquations, nextGen[1], [])
            if len(intersections) > 0:
                while len(intersections) > 0 and not count > numConnections:
                    fh.write("%.6f    %.6f\n" %(intersections[0][0][0], intersections[0][0][1]))
                    count += 1
    
                    idx = intersections[0][1]
                    direction = [positions[idx][0] - intersections[0][0][0],
                                 positions[idx][1] - intersections[0][0][1]]
                    curAngle = math.atan2(direction[1], direction[0])
                    nextGen = computeMinAngle(idx, curAngle, positions, connections, False)
    
                    intersections = checkPossibleSegmentIntersection((None, idx), positions, lineEquations, nextGen[1], intersections[0][0])
            else:
                idx = nextIdx
    
            fh.write("%.6f    %.6f\n" % (positions[idx][0], positions[idx][1]))
    
            if idx in passed:
                direction1 = [positions[idx][0] - positions[passed[-1]][0],
                              positions[idx][1] - positions[passed[-1]][1]]
                direction2 = [positions[origin][0] - positions[passed[-1]][0],
                              positions[origin][1] - positions[passed[-1]][1]]
                dist1 = distance(positions[idx], positions[passed[-1]])
                dist2 = distance(positions[origin], positions[passed[-1]])
                if dist1 * dist2 > 0.0 and math.fabs(math.fabs(dot(direction1, direction2) / (dist1 * dist2)) - 1.0) > 1e-9:
                    sys.stderr.write("error: projection cycling on element id: %d, vertex id: %d\n" %(i, idx))
                break

            passed.append(idx)
            count += 1
        fh.write("\n")

        if count > numConnections:
            sys.stderr.write("error: projection cycling on element id: %d\n" % i)

        for j in range(0, len(decoration[i]), 6):
            for k in range(j, j + 6, 3):
                nextPos3D = ori.rotate(decoration[i][k:k+3])
                fh.write("%.6f    %.6f\n" % (nextPos3D[0], nextPos3D[1]))
            fh.write("\n")

    fh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-vtk', action='store_true')
    parser.add_argument('--export-web', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--background', nargs=3, type=float)
    parser.add_argument('--project-to-plane', action='store_true')
    parser.add_argument('--normal', nargs=3, type=float)
    args = parser.parse_args()

    if (args.export_vtk):
        exportVTK()
        sys.exit()

    if (args.export_web):
        bgcolor = []
        if (args.background):
            validBackground = True
            for comp in bgcolor:
                if comp < 0.0 or comp > 1.0:
                    validBackground = False
                    break
            if (validBackground):
                bgcolor = args.background
        exportWeb(bgcolor)
        sys.exit()

    if (args.show):
        sys.exit(showVTK())

    if (args.project_to_plane):
        normal = [0.0, 1.0, 0.0]
        if (args.normal):
            normal = args.normal
        projectToPlane(normal)
        sys.exit()

    parser.print_help()
