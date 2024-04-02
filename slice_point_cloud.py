import os.path

import shapefile
import laspy
import time
from laxpy import IndexedLAS
from shapely import Polygon
import numpy as np
import argparse

PROGRAM_DESCRIPTION = "Slice individual plots, defined in a shapefile, " \
                      "from a LAS pointcloud into individual point clouds."
PROGRAM_EPILOGUE = "Written by Joe Gage, 2024."

# shp_file = "/Volumes/gagelab/ProcessedUAVData/G2F2023_shapefile/G2F2023_plots.shp"
# las_file = "/Volumes/gagelab/ProcessedUAVData/DJI_202307271046_003_G2F2023/DJI_202307271046_003_G2F-2023_20231220T1220_points.las"
# plot_idx = 230

# get_shapes copied from Lucas Bauer's orthoslice
# https://github.com/gagelab/orthoslice/blob/main/orthoslice.py
def get_shapes(shp_file):
    """Shapefile -> points, (x,y) format, geographic coordinates
    """

    sf = shapefile.Reader(shp_file)
    shapes = sf.shapes()

    shape_dict = dict()
    for idx, sp in enumerate(shapes):
        shape_dict[idx] = {}
        shape_dict[idx]["points"] = sp.points
        shape_dict[idx]["bbox"] = sp.bbox

    return shape_dict

# Next two functions help identify whether a point is inside a given polygon
# Copied from https://stackoverflow.com/a/63529043
def is_on_right_side(x, y, xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = - a*x0 - b*y0
    return a*x + b*y + c >= 0

def test_point(x, y, vertices):
    num_vert = len(vertices)
    is_right = [is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert]) for i in range(num_vert)]
    all_left = not any(is_right)
    all_right = all(is_right)
    return all_left or all_right

# Function to find angle that the plot needs to be rotated:
def find_theta(p):
    # Assumes 'front' of plot is in lower left and plot is angled
    #  between 0 (12 oclock) and pi/2 (3 oclock)
    # Uses tan = opposite / adjacent to find the angle the plot is at.
    # Subtracts that value from pi/2 (90deg) to find the necessary rotation
    # In this diagram: angles are estimated from lower right corner (*).
    # Dashed lines show the triangle being used. The angle calculated with arctan
    #  is that from baseline (--) to hypotenuse (/)
    #     *    *
    #         /|
    #        / |
    # *     *--+
    # TODO: I think this angle might be dependent on the ordering of the
    #  corner points and maybe orientation of the plot? Could be made more
    #  robust.

    dx = p[1][0] - p[2][0]
    dy = p[1][1] - p[2][1]
    theta = np.pi / 2 - np.arctan(dy / dx)
    return theta

# Thanks ChatGPT :)
def rodrigues_rotation(coords, axis, angle):
    """
    Perform axis-angle rotation using Rodrigues' rotation formula.

    Parameters:
    - coords: NumPy array of shape (n, 3) representing existing coordinates.
    - axis: NumPy array of shape (3,) representing the axis of rotation.
    - angle: Rotation angle in degrees.

    Returns:
    - rotated_coords: NumPy array of shape (n, 3) representing the rotated coordinates.
    """

    # Convert angle to radians
    # angle_rad = np.radians(angle)
    angle_rad = angle

    # Normalize axis of rotation
    axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    dot_product = np.dot(coords, axis)
    cross_product = np.cross(axis, coords)
    rotated_coords = (coords * cos_theta +
                      cross_product * sin_theta +
                      np.outer(dot_product, (1 - cos_theta)) * axis)

    return rotated_coords

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=PROGRAM_DESCRIPTION,
        epilog=PROGRAM_EPILOGUE
    )

    # Positional args
    parser.add_argument("lasfile_path", type=str, help="String path to LAS format pointcloud.")
    parser.add_argument("shapefile_path", type=str, help="String path to shapefile")

    # Optional args
    parser.add_argument("--output_directory", "-o", type=str, default="./",
                        help="Output directory. Creates directory if none exist.")
    parser.add_argument("--rotate", "-r", action="store_true", help="Should plot level point cloud be centered and rotated so rows are parallel to y axis?")
    parser.add_argument("--plot_idx", "-p", type=int, help="Specify index of a single plot to extract. Otherwise all are extracted, iteratively.")
    parser.add_argument("--xyz", action="store_true", help="If set, plot point clouds will be written out as XYZ text files rather than LAS")

    args = parser.parse_args()

    # Read in all plot shapes and subset if desired
    shp = get_shapes(args.shapefile_path)
    if args.plot_idx:
        plot_idxs = [args.plot_idx]
    else:
        plot_idxs = range(len(shp))

    print(len(shp))
    for i in plot_idxs:
        # Read indexed LAS and subset to the desired plot index
        subset_las = IndexedLAS(args.lasfile_path)
        p = shp[i]['points']
        P = Polygon(shp[i]['points'])
        subset_las.map_polygon(P)

        # subset_las isn't fully downsampled to points inside the plot polygon;
        #  for some reason it seems to be points in the general vicinity. So now,
        #  identify points from subset_las that are actually inside the polygon
        inside = []
        for x, y in zip(subset_las.x, subset_las.y):
            inside.append(test_point(x, y, p))

        # Make np array of points to rotate
        pcd = np.zeros((np.sum(inside), 3))
        pcd[:,0] = subset_las.x[inside]
        pcd[:,1] = subset_las.y[inside]
        pcd[:,2] = subset_las.z[inside]

        if args.rotate:
            # Get rotation angle
            theta = find_theta(p)
            # Rotate plot points around Z axis (0,0,1)
            pcd_out = rodrigues_rotation(pcd, (0,0,1), theta)
        else:
            pcd_out = pcd

        # Define output file path
        name = os.path.basename(args.lasfile_path).removesuffix('.las')
        outfile = os.path.join(args.output_directory, name+"_plot"+str(i))

        # Write out new LAS
        if args.xyz:
            outfile=outfile+".txt"
            np.savetxt(outfile, pcd_out)
        else:
            outfile=outfile+".las"
            header = laspy.header.Header(file_version=subset_las.header.version, point_format=subset_las.header.data_format_id)
            single_plot = laspy.file.File(outfile, mode="w", header=header)
            single_plot.header.offset = np.min(pcd_out, axis=0)
            single_plot.header.scale = subset_las.header.scale
            single_plot.x = pcd_out[:,0]
            single_plot.y = pcd_out[:,1]
            single_plot.z = pcd_out[:,2]
            single_plot.red = subset_las.red[inside]
            single_plot.green = subset_las.green[inside]
            single_plot.blue = subset_las.blue[inside]
            single_plot.close()

