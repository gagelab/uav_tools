# uav_tools
A collection of tools for processing our UAV data.

## Conda environment
`environment.yaml` contains packages needed to run all tools. Get started with:

```
# Create the environment:
conda env create --name uav_tools --file environment.yaml
# Activate the environment:
conda activate uav_tools
```

## slice_orthomosaic.py
Slices up an orthomosaic and returns individual images of each plot. For more details run:

```
python slice_orthomosaic.py -h
```

## slice_point_cloud.py
Slices up a field-scale point cloud and returns individual point clouds of each plot. For more details run:

```
python slice_point_cloud.py -h
```

## Contact
* Joe Gage jlgage@ncsu.edu
