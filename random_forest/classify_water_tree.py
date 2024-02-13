import os
import timeit
import math
import warnings 

import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
import rasterio
import geopandas as gpd
from rasterio import plot
from rasterio.mask import mask
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle


def generateTreeQ(raster, water, land, bandnames):
    # Load in the raster
    ds = rasterio.open(raster)
    image = ds.read()

    # Generate water points
    water = gpd.read_file(water)
    out, _ = mask(ds, water.geometry, invert=False, filled=False)
    mask_ar = np.invert(out.mask)
    data = out.data

    band_data = {name: [] for name in bandnames}
    for i, name in enumerate(band_data.keys()):
        band_data[name] = data[i][mask_ar[i]]

    water_df = pandas.DataFrame(band_data)
    water_df['class'] = [1 for i in range(len(water_df))]

    # Generate land points
    land = gpd.read_file(land)
    out, _ = mask(ds, land.geometry, invert=False, filled=False)
    mask_ar = np.invert(out.mask)
    data = out.data

    band_data = {name: [] for name in bandnames}
    for i, name in enumerate(band_data.keys()):
        band_data[name] = data[i][mask_ar[i]]

    not_water_df = pandas.DataFrame(band_data)
    not_water_df['class'] = [0 for i in range(len(not_water_df))]

    # Set up whole df
    df = pandas.concat([water_df, not_water_df])

    # Remove Nan
    df = df.dropna(how='any')

    # Initialize tree
    clf = DecisionTreeClassifier(
        random_state=0, 
        max_depth=5
    )

    feature_cols = [b for b in bandnames]
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_cols], 
        df['class'], 
        test_size=0.1, 
        random_state=1
    )

    clf = clf.fit(
        x_train,
        y_train
    )

    y_pred = clf.predict(x_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    return clf


def predictPixelsQ(ds, clf, bandnames):
    image = ds.read()

    # Reshape to correct shape
    nans = image[image == 0]
    new_shape = (image.shape[1] * image.shape[2], image.shape[0])
    image_predict = np.moveaxis(image, 0, -1)
    img_as_array = image_predict[:, :, :].reshape(new_shape)
    print('Reshaped from {o} to {n}'.format(
        o=image.shape,
        n=img_as_array.shape)
    )

    # Crazy method to predict for each pixel
    predictions = np.empty([img_as_array.shape[0],])
    predictions[:] = None
    for i, row in enumerate(img_as_array):
        if len(row[~np.isnan(row)]) > 0:
            predictions[i] = clf.predict(row.reshape(1, len(bandnames)))[0]

    # Reshape our classification map
    class_prediction = predictions.reshape(image_predict[:, :, 0].shape)
    class_prediction[nans] = 0

    return class_prediction.astype(rasterio.int8)



raster = 'Planet_AmuDarya_Kerki/water_masks/Color/2013_RE05.tif'
water = 'Planet_AmuDarya_Kerki/water_masks/Shapes_2010/water_2013RE05.gpkg'
land = 'Planet_AmuDarya_Kerki/water_masks/Shapes_2010/land_2013RE05.gpkg'

bandnames = ['blue', 'green', 'red', 'red_edge', 'NIR']
#bandnames = ['blue', 'green', 'red', 'NIR']

clf = generateTreeQ(raster, water, land, bandnames)
warnings.filterwarnings("ignore")
ds = rasterio.open(raster)
pred = predictPixelsQ(ds, clf, bandnames)

dsmeta = ds.meta
dsmeta.update(
    width=pred.shape[1],
    height=pred.shape[0],
    count=1,
)
outfile = 'WaterMask_2013RE.tif'
with rasterio.open(outfile, 'w', **dsmeta) as dst:
    dst.write(pred.astype(rasterio.uint16), 1)

# raster = 'Tuotuo/2016/Tuotuo_2016_RapidEye_SR.tif'
# ds = rasterio.open(raster)
# pred = predictPixelsQ(ds, clf, bandnames)

# dsmeta = ds.meta
# dsmeta.update(
#     width=pred.shape[1],
#     height=pred.shape[0],
#     count=1,
# )
# outfile = 'WaterMask_2016.tif'
# with rasterio.open(outfile, 'w', **dsmeta) as dst:
#     dst.write(pred.astype(rasterio.uint16), 1)

# raster = 'Tuotuo/2012/Tuotuo_2012_RapidEye_SR.tif'
# ds = rasterio.open(raster)
# pred = predictPixelsQ(ds, clf, bandnames)

# dsmeta = ds.meta
# dsmeta.update(
#     width=pred.shape[1],
#     height=pred.shape[0],
#     count=1,
# )
# outfile = 'WaterMask_2012.tif'
# with rasterio.open(outfile, 'w', **dsmeta) as dst:
#     dst.write(pred.astype(rasterio.uint16), 1)
