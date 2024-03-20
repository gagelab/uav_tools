## VERSION 0.1
import sys, os
import argparse

import shapefile
import matplotlib.pyplot as plt
import numpy as np

import rasterio
from rasterio.windows import Window
from rasterio.plot import show

import cv2

PROGRAM_DESCRIPTION = "Slice individual plots defined in a shapefile from a GeoTIFF image into individual images."
PROGRAM_EPILOGUE = "Written by Lucas Bauer in the Gage Lab, 2024."

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

### 2

def get_px_coords(geotiff_path, coord_list):
    """Input coord_list as a list of (x,y) tuples
    """
    with rasterio.open(geotiff_path) as src:
        meta = src.meta

        output_list = []
        for coords in coord_list:
            work_list = []
            for x, y in coords:
                rowcol = rasterio.transform.rowcol(meta["transform"], xs=x, ys=y)
                work_list.append(rowcol)
                
            output_list.append(work_list)
    return output_list


def get_px_coords_bbox(geotiff_path, bbox_list):
    """Input coord_list as a list of (x,y) tuples
    """
    with rasterio.open(geotiff_path) as src:
        meta = src.meta

        output_list = []
        for bbox in bbox_list:
            x1, y1, x2, y2 = bbox

            work_list = []
            rowcol1 = rasterio.transform.rowcol(meta["transform"], xs=x1, ys=y1)
            work_list.extend(rowcol1)
            rowcol2 = rasterio.transform.rowcol(meta["transform"], xs=x2, ys=y2)
            work_list.extend(rowcol2)

            output_list.append(work_list)
    
    return output_list

### 3

# 3: Pull subset files 


def rearrange_plot(rasterio_arr):
    r = rasterio_arr[0,:,:]
    g = rasterio_arr[1,:,:]
    b = rasterio_arr[2,:,:]
    
    stacked_img = np.stack([r, g, b], axis=2)

    return stacked_img

def keystone_correct(rgb_img, src_points, dest_points, zeros_arr=None):
    """Keystone correct image orientation from one list of points to another

    Args:
        rgb_img: NumPy array representing an image in RGB colorspace format
        src_points (list): list of tuples representing source points in original image, 
            format [(x1,y1),(x2,y2),...]
        dest_points (list): list of tuples representing destination points in relative to image, 
        format [(x1,y1),(x2,y2),...]

    Returns:
        dest_img: Keystone corrected image as numpy array in RGB colorspace

	"""

    # Check that lists are of matched length
    if len(src_points) != len(dest_points):
        raise Exception(f"Number of points in input lists not equal. src_points: {len(src_points)}, dest_points: {len(dest_points)}")
        
    if (zeros_arr is None):
        dstD = np.zeros(rgb_img.shape,dtype=np.uint8)
    else:
        dstD = zeros_arr
	
	# Keystone correction functionality
    H = cv2.findHomography(np.array(src_points,dtype=np.float32),np.array(dest_points,dtype=np.float32),cv2.LMEDS)
    dest_img=cv2.warpPerspective(rgb_img,H[0],(dstD.shape[1],dstD.shape[0]))
    
    return dest_img

def rotate_list(arr,d,n):
    """Rotates list arr of length n by number of positions d
    """
    arr=arr[:]
    arr=arr[d:n]+arr[0:d]
    return arr

def slice_bbox(open_raster, bbox):

    x1, y1, x2, y2 = bbox

    x_min = x1 if x1 < x2 else x2
    y_min = y1 if y1 < y2 else y2

    dx = abs(x2-x1)
    dy = abs(y2-y1)

    row, col = open_raster.index(y1, x1)
    window = Window(y_min, x_min, dy, dx)
    data = open_raster.read((1,2,3), window=window)

    return rearrange_plot(data)

def slice_bbox_correct(open_raster, bbox, points_list, final_resolution=(2048, 1024)):

    # Prepare bounding box values
    x1, y1, x2, y2 = bbox

    x_min = x1 if x1 < x2 else x2
    y_min = y1 if y1 < y2 else y2

    dx = abs(x2-x1)
    dy = abs(y2-y1)

    # Take slice from image
    row, col = open_raster.index(y1, x1)
    window = Window(y_min, x_min, dy, dx)
    data = open_raster.read((1,2,3), window=window)

    # Convert to OpenCV format
    data = rearrange_plot(data)

    # Prepare points source list
    points_converted = [(y-y_min, x-x_min) for x, y in points_list[:-1]]

    # Prepare destination points
    res_h, res_w = final_resolution
    dest_pts = [(0,0), (res_w, 0), (res_w, res_h), (0,res_h)]

    return keystone_correct(data, points_converted, dest_pts, np.zeros((*final_resolution, 3)))

def inset_correct_plot(l, dy=0, dx=0):

    out_list = []
    for sub_l in l:
        working_list = []
        for y, x in sub_l:
            working_list.append((y+dy, x+dx))
        out_list.append(working_list)

    return out_list

def inset_correct_bbox(l, dy=0, dx=0):

    out_list = []
    for y1, x1, y2, x2  in l:
        working_list = [y1+dy, x1+dx, y2+dy, x2+dx]
        out_list.append(working_list)

    return out_list

def scale_correct_bbox(l, factor=1.0):
    out_list = []

    factor_down_right = factor
    factor_up_left = (1-(factor-1))
    
    
    for y1, x1, y2, x2 in l:

        y1_corr = int(y1 * factor_down_right)
        x1_corr = int(x1 * factor_up_left)
        y2_corr = int(y2 * factor_up_left)
        x2_corr = int(x2 * factor_down_right)
        
        working_list = [y1_corr, x1_corr, y2_corr, x2_corr]
        out_list.append(working_list)

    return out_list

def scale_correct_plot(l, factor=0.0):
    out_list = []

    # find centroid
    for pts in l:
        y_sum = x_sum = 0
        for y, x in pts[:-1]:
            y_sum += y
            x_sum += x
    
        y_center = y_sum / 4
        x_center = x_sum / 4
    
        factor_down_right = factor
        factor_up_left = 1-(factor-1)
        
        work_list = []
        for y, x in pts:
            # up or down from centroid, scale accordingly
            if (y > y_center):
                y_work = int(y * factor_down_right)
            else:
                y_work = int(y * factor_up_left)
            
            # to right or left of centroid, scale accordingly
            if (x > x_center):
                x_work = int(x * factor_down_right)
            else:
                x_work = int(x * factor_up_left)
    
            work_list.append((y_work,x_work))

        out_list.append(work_list)
        
    
    return out_list

def scale_correct_plot_offset(plot_l, bbox_l, factor=1.0):

    plot_out_l = []
    
    bbox_scale_l = scale_correct_bbox(bbox_l, factor=factor)
    plot_scale_l = scale_correct_plot(plot_l, factor=factor)

    
    for idx, bbox_vals in enumerate(bbox_scale_l):
        y1, x1, y2, x2 = bbox_l[idx]
        y1p, x1p, y2p, x2p = bbox_vals

        yd = abs(min((y1p-y1, y2p-y2)))
        xd = abs(min((x1p-x1, x2p-x2)))
        
        plot_scale_l_work = [(y+yd, x+xd) for y, x in plot_scale_l[idx]]
        plot_out_l.append(plot_scale_l_work)

    return bbox_scale_l, plot_out_l

def scale_correct_plot_offset_reverse(plot_l, bbox_l, factor=1.0):

    plot_out_l = []
    
    bbox_scale_l = scale_correct_bbox(bbox_l, factor=factor)

    
    for idx, bbox_vals in enumerate(bbox_scale_l):
        y1, x1, y2, x2 = bbox_l[idx]
        y1p, x1p, y2p, x2p = bbox_vals

        yd = abs(min((y1p-y1, y2p-y2)))
        xd = abs(min((x1p-x1, x2p-x2)))
        
        plot_scale_l_work = [(y+yd, x+xd) for y, x in plot_l[idx]]
        plot_out_l.append(plot_scale_l_work)

    plot_scale_l = scale_correct_plot(plot_out_l, factor=factor)
    return bbox_scale_l, plot_scale_l
    
def slice_ortho(geotiff_path, shapefile_path, out_path=None, dx=0, dy=0, pad_factor=1.0):
    # 1: Get shape points from shapefile
    shapefile_pts_dict = get_shapes(shapefile_path)

    # 2: Convert shape points from geological coords to pixel (x,y)
    plot_points_dict = [shapefile_pts_dict[i]["points"] for i in shapefile_pts_dict.keys()]
    plot_bbox_dict = [shapefile_pts_dict[i]["bbox"] for i in shapefile_pts_dict.keys()]
    
    plot_convert_list = get_px_coords(geotiff_path, plot_points_dict)
    bbox_convert_list = get_px_coords_bbox(geotiff_path, plot_bbox_dict)

    # 3: Apply vertical and horizontal correction
    plot_convert_inset_list = inset_correct_plot(plot_convert_list, dy=dy, dx=dx)
    bbox_convert_inset_list = inset_correct_bbox(bbox_convert_list, dy=dy, dx=dx)

    # 4: Apply scale correction
    bbox_pad_list = scale_correct_bbox(bbox_convert_inset_list, factor=pad_factor)
    plot_pad_list = scale_correct_plot(plot_convert_inset_list, factor=pad_factor)

    # 5: Run pipline on images

    # Check path if needs to me made
    if (out_path is None):
        out_path = "."

    elif (not os.path.isdir(out_path)):
        os.makedirs(out_path)

    # Get pad number
    pad_val = len(str(len(plot_convert_list)))

    # Run correction
    with rasterio.open(geotiff_path) as src:
        for idx, bbox in enumerate(bbox_pad_list):
            file_name = f"plot{str(idx).zfill(pad_val)}.jpg"
            file_path = os.path.join(out_path, file_name)
            
            out_arr = slice_bbox_correct(src, bbox, plot_pad_list[idx])

            #if not os.path.isfile(file_path):
            # TODO make option to overwrite or not
            cv2.imwrite(file_path, cv2.cvtColor(out_arr, cv2.COLOR_RGB2BGR))

def main():

    parser = argparse.ArgumentParser(
        description=PROGRAM_DESCRIPTION,
        epilog=PROGRAM_EPILOGUE
    )

    # Core args
    parser.add_argument("geotiff_path", type=str, help="String path to geotiff.")
    parser.add_argument("shapefile_path", type=str, help="String path to shapefile")
    parser.add_argument("--output_directory", type=str, default=None, 
                        help="Output directory. Creates directory if none exist.")

    # Specific Args
    parser.add_argument("--dx", type=int, default=0, 
                        help="Horizontal shift in pixels on original geotiff image. Negative numbers shift left, positive shift right.")
    parser.add_argument("--dy", type=int, default=0,
                        help="Vertical shift in pixels on original geotiff image. Negative numbers shift up, positive shift down.")

    parser.add_argument("--pad_factor", type=float, default=1.0,
                        help="Adds padding factor as a percentage of both vertical and horizontal shape sizes. Factor > 1.0 increases size. Factor < 1.0 decreases size.")

    # Parse args
    args = parser.parse_args()

    # Run core function
    geotiff_path = args.geotiff_path
    shapefile_path = args.shapefile_path
    output_directory = args.output_directory
    dx = args.dx
    dy = args.dy
    pad_factor = args.pad_factor

    slice_ortho(geotiff_path=geotiff_path, shapefile_path=shapefile_path, out_path=output_directory,
                dx=dx, dy=dy, pad_factor=pad_factor)



if __name__ == "__main__":
    main()

