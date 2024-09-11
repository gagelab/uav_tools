#
# reffile = "/Volumes/rsstu/users/j/jlgage/gagelab/users/jlgage/ref2.txt"
# img_dir = "/Volumes/gagelab/RawUAVData/2023/DJI_202307271046_003_G2F-2023/"
# dtm_path = "/Volumes/gagelab/ProcessedUAVData/DJI_202307271046_003_G2F2023/DJI_202307271046_003_G2F2023_20231220T1220_dtm.tif"
# geotiff_out = '/Users/jlgage/Downloads/test.tif'

import argparse
from PIL import Image, ExifTags
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import os
from os import path
from rasterio.transform import Affine
import rasterio
from rasterio.plot import reshape_as_raster
from rasterio.mask import raster_geometry_mask

PROGRAM_DESCRIPTION = "Create georeferenced GeoTIFFs for raw UAV images"
PROGRAM_EPILOGUE = "Written Sept 2024; Joe Gage"

# Camera parameters
# From https://enterprise.dji.com/zenmuse-p1/specs
# Sections Camera > Sensor > Sensor size: "Sensor size (Still): 35.9×24 mm (Full frame)"
sensor_size_long = 35.9

# CRS params
camera_crs = "WGS84"
ortho_crs = "ESRI:103501"

# This is the FOV reported for the P1, but images do not line up properly with the ortho when using this value.
# Wikipedia had a formula for calculating FOV from focal length and sensor size that gives a different value and performs
#  better - see
# fov = 63.5


def estimate_fov(focal_length, sensor_size):
    """
    Calculate FOV from focal length and sensor size.
    FOV calculation from here: https://en.wikipedia.org/wiki/Field_of_view#Photography

    :param focal_length: camera focal length in mm
    :param sensor_size: camera sensor size (whichever dimension you desire) in mm
    :return: scalar value of FOV in degrees
    """
    return np.degrees(2 * np.arctan(sensor_size / (2 * focal_length)))

def get_img_dims(img):
    exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS}
    focal_length = exif["FocalLength"]
    w = exif["ExifImageWidth"]
    h = exif["ExifImageHeight"]
    # aspect_ratio = w / h
    return w, h, focal_length


def est_height_above_ground(camera_coords, camera_alt, dsm, buffer=1):
    """
    Estimates camera height above the ground.
    *** NOTE! May need modification to handle other CRSs

    :param camera_coords: geopandas GeoSeries with one Shapely Point specifying camera location
    :param camera_alt: estimated altitude of camera in meters. Comes from the camera reference file
    :param dsm: an open rasterio GeoTiff
    :param buffer: radius of circle around camera location over which to take mean elevation. Unit is probably the same as the CRS.
    :return: scalar value; difference between camera height and ground height
    """
    # Confession - I have no idea what unit the buffer is. Guessing same as the CRS, so feet? Rough estimate of pixel
    # count checks out
    camera_loc_mask = raster_geometry_mask(dsm, camera_coords.buffer(buffer), invert=True)[0]
    # np.sum(R[0])
    elev_ft = np.mean(dsm.read(1)[camera_loc_mask])  # Get the elevation of the DSM directly beneath the camera
    alt_ft = camera_alt / 0.3048  # convert altitude of UAV (above geoid?) in meters to feet
    return alt_ft - elev_ft  # Get height of UAV above ground level


def convert_camera_loc(lat, long, crs_in="WGS84", crs_out="ESRI:103501"):
    camera_loc = gpd.GeoSeries(Point(long, lat), crs=crs_in)
    camera_loc_ft = camera_loc.to_crs(crs_out)

    return camera_loc_ft


# TODO: Pretty sure I don't need to extrapolate the entire image. Try just using corners.
def extrapolate_pixel_coords_from_camera(camera_coords, alt_above_ground, field_of_view, img_width, img_height):
    """

    :param camera_coords:
    :param alt_above_ground:
    :param field_of_view:
    :param img_width:
    :param img_height:
    :return:
    """
    # Distance from center to edge of image along the wide axis
    long_width = 2 * alt_above_ground * np.tan(np.radians(field_of_view / 2))
    #aspect_ratio = w/h
    #short_width = long_width / aspect_ratio

    center = img_width / 2, img_height / 2

    # Make pixel-based coords for every pixel in the image.
    # These values are relative to the IMAGE (ie no geography yet) and suffixed with '_img'
    # Center is (0,0)
    coords = np.meshgrid(np.arange(-center[0], center[0]),
                         np.arange(-center[1], center[1]))
    # Put it into a DF
    img_info = pd.DataFrame({"x_img": np.ravel(coords[1]),
                             "y_img": np.ravel(coords[0])})
    # Calculate the distance of each pixel from the center and the angle
    img_info["d_img"] = np.sqrt(img_info["x_img"] ** 2 + img_info["y_img"] ** 2)
    img_info["theta_img"] = np.arctan2(img_info["y_img"], img_info["x_img"])

    # Now add "real world" positions

    # For angle - need to add pi/2 to move "0" from being horizontal left to the top of the image. Then add yaw to
    #  adjust for orientation of the UAV
    img_info["theta_geo"] = img_info["theta_img"] - np.pi / 2 - np.radians(yaw)
    #
    px_size = long_width / img_width
    img_info["d_geo"] = img_info["d_img"] * px_size
    img_info["x_geo"] = np.array(camera_coords.x) + img_info["d_geo"] * np.cos(img_info["theta_geo"])
    img_info["y_geo"] = np.array(camera_coords.y) + img_info["d_geo"] * np.sin(img_info["theta_geo"])
    # coord_out = gpd.GeoSeries(gpd.points_from_xy(img_info["x_geo"], img_info["y_geo"]), crs="ESRI:103501").to_crs("EPSG:2264")

    return img_info


def estimate_transform(img_info):
    # Choose three random points to use for estimating the rotation
    idx = np.random.choice(np.arange(len(img_info)), 3, replace=False)
    C = img_info.loc[idx, ['x_img', 'y_img', 'x_geo', 'y_geo']]
    C["x"] = C.x_img - np.min(img_info.x_img)
    C["y"] = C.y_img - np.min(img_info.y_img)

    orig = np.array([C.y, C.x])
    # Add a row of ones to X for the translation component
    orig_aug = np.vstack([orig, np.ones((1, orig.shape[1]))])
    # Compute the pseudoinverse of the augmented original points matrix
    orig_pseudo_inverse = np.linalg.pinv(orig_aug)

    # Get transformed corners and add row of ones
    trans = np.array([C.x_geo, C.y_geo])
    # trans = np.array([C.y_geo, C.x_geo])
    trans_aug = np.vstack([trans, np.ones((1, trans.shape[1]))])

    # Compute the affine transformation matrix P
    P = trans_aug @ orig_pseudo_inverse
    # Make the rasterio Affine object
    transform = Affine(P[0, 0], P[0, 1], P[0, 2], P[1, 0], P[1, 1], P[1, 2])

    return transform

def parse_command_args():
    parser = argparse.ArgumentParser(
        description=PROGRAM_DESCRIPTION,
        epilog=PROGRAM_EPILOGUE
    )

    # parser.Namespace("img_path", "/Volumes/gagelab/RawUAVData/2023/DJI_202307271046_003_G2F-2023/DJI_202307271046_003_G2F-2023/DJI_20230727105702_0001.JPG")

    # Core args
    parser.add_argument("img_path", type=str,
                        help="String path to raw image or directory of raw images.")
    parser.add_argument("dtm_path", type=str,
                        help="String path to Digital Terrain Model.")
    parser.add_argument("ref_path", type=str,
                        help="String path to references exported by Metashape (ie camera positions). Must include "
                             "estimated X, Y, altitude, and Yaw.")
    parser.add_argument("-o", "--output_directory", type=str, default="./",
                        help="Output directory. Creates directory if none exist.")

    # Specific Args
    parser.add_argument("--camera_crs", type=str, default="WGS84",
                        help="Coordinate reference system in camera exif metadata.")
    parser.add_argument("--ortho_crs", type=str, default="ESRI:103501",
                        help="Coordinate reference system of the ortho/dsm/shapefile/downstream use.")

    # Parse args
    args = parser.parse_args()
    # args = parser.parse_args(args=[
    #     "--img_path", "/Volumes/gagelab/RawUAVData/2023/DJI_202307271046_003_G2F-2023/DJI_20230727105702_0001.JPG",
    #     # "--img_path", "/Volumes/gagelab/RawUAVData/2023/DJI_202307271046_003_G2F-2023/",
    #     "--ref_path", "/Volumes/rsstu/users/j/jlgage/gagelab/users/jlgage/ref2.txt"]
    # )

    return args

if __name__ == "__main__":

    # Parse command line args
    args = parse_command_args()

    # Open DSM for reading
    dtm = rasterio.open(args.dtm_path, 'r')

    # Read ref data and subset to the image provided by img_path or images in directory provided by img_path
    ref = pd.read_csv(args.ref_path, skiprows=1, sep="\t", index_col=0)
    if path.isfile(args.img_path):
        img_files = [args.img_path]
        img_basenames = [path.basename(args.img_path)]
    elif path.isdir(args.img_path):
        img_basenames = os.listdir(args.img_path)
        img_files = [path.join(path.dirname(args.img_path), b) for b in img_basenames]
    else:
        raise Exception("img_path does not exist as a file or directory.")

    # Only keep images (or files in image dir) that are in the index of the reference dataframe
    images_in_ref = [b in ref.index for b in img_basenames]
    img_basenames = np.array(img_basenames)[images_in_ref]
    img_files = np.array(img_files)[images_in_ref]
    ref = ref.loc[img_basenames]

    # *** For testing - only run on a subset of 5 images ***
    # random_images = np.random.choice(ref.index, 5)
    # for imgname in random_images: # Testing only

    # Iterate through raw images and create GeoTiff of each
    for img_file, img_basename in zip(img_files, img_basenames):
        # Read image file
        img = Image.open(img_file)

        # Get needed info from image
        w, h, focal_length = get_img_dims(img)
        fov = estimate_fov(focal_length, sensor_size_long)

        # Get lat/lon/altitude/yaw from Metashape ref data
        lat = ref.loc[img_basename, 'Y_est']
        lon = ref.loc[img_basename, 'X_est']
        alt = ref.loc[img_basename, 'Z_est']
        yaw = ref.loc[img_basename, 'Yaw_est']

        # Get camera location in ortho CRS and estimate camera height above ground
        camera_loc = convert_camera_loc(lat, lon, crs_in=args.camera_crs, crs_out=args.ortho_crs)
        ft_above_ground = est_height_above_ground(camera_loc, alt, dsm=dtm, buffer=1)

        # Create DataFrame with per-pixel coords in image coordinates as well as ortho CRS coords
        # Also includes some info used to convert (angles, distances from center, etc)
        img_info = extrapolate_pixel_coords_from_camera(camera_coords=camera_loc, alt_above_ground=ft_above_ground,
                                                        field_of_view=fov, img_width=w, img_height=h)

        # Estimate transform from image coords to ortho CRS and use this to write the raw image out as a GeoTiff
        transform = estimate_transform(img_info)
        rgb = np.array(img, dtype=np.uint8)
        geotiff_out = path.join(args.output_directory, path.splitext(img_basename)[0] + ".tiff")
        with rasterio.open(geotiff_out,
                           'w',
                           driver='GTiff',
                           width=w,
                           height=h,
                           count=3,
                           dtype=np.uint8,
                           crs=args.ortho_crs,
                           transform=transform
                           ) as dst:
            dst.write(reshape_as_raster(rgb))

        # ***** Stuff for troubleshooting *****

        # Print corner coords in a format that can be pasted into a QGIS layer
        # corners = [np.argmax(img_info["y_geo"]),
        #            np.argmin(img_info["y_geo"]),
        #            np.argmax(img_info["x_geo"]),
        #            np.argmin(img_info["x_geo"])]
        # for i in range(len(corners)):
        #     print("POINT (" + str(img_info.loc[corners[i], "x_geo"]) + " " + str(img_info.loc[corners[i], "y_geo"]) + ")")
