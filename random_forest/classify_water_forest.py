from os import path as op
import pickle
import warnings 
import geopandas as gpd
import pandas
import shapely as shp
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterstats.io import bounds_window
import rasterstats
import folium

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from treeinterpreter import treeinterpreter as ti
from rasterio.mask import mask

def generateforestQ(raster, water, land, bandnames):
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


    feature_cols = [b for b in bandnames]
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_cols], 
        df['class'], 
        test_size=0.6, 
        random_state=42
    )

    # calculate class weights to allow for training on inbalanced training samples
    labels, counts = np.unique(y_train, return_counts=True)
    class_weight_dict = dict(zip(labels, 1 / counts))
    class_weight_dict

    # Initialize tree
    clf = RandomForestClassifier(
    n_estimators=500,
    class_weight=class_weight_dict,
    max_depth=6,
    n_jobs=-1,
    verbose=1,
    random_state=0)

    clf = clf.fit(
        x_train,
        y_train
    )

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # plot the confusion matrix
    %matplotlib inline
    classes=['Water', 'Land']
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.rcParams.update({'font.size': 20})
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]))
        # ... and label them with the respective list entries
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_title('Normalized Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)


    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    sample = 100
    prediction, bias, contributions = ti.predict(clf, x_test[:sample])
    c = np.sum(contributions, axis=0)

    # plot the contributions
    gdf = gpd.GeoDataFrame(c, columns=classes, index=bandnames)
    display(gdf.style.background_gradient(cmap='viridis'))
    return clf


raster = 'Tuotuo/2015/Color/2015_10RE.tif'
water = 'Tuotuo/2015/Shapes/2015_10RE_water.gpkg'
land = 'Tuotuo/2015/Shapes/2015_10RE_land.gpkg'

bandnames = ['blue', 'green', 'red', 'red_edge', 'NIR', 'NDWI']
#bandnames = ['blue', 'green', 'red', 'NIR','NDWI']

clf = generateforestQ(raster, water, land, bandnames)
warnings.filterwarnings("ignore")


outfile = 'Tuotuo/2015/Binary/2015_10RE_500trees.tif'

with rasterio.open(raster, 'r') as src:
    profile = src.profile
    profile.update(
        dtype=rasterio.uint8,
        count=1,
    )
    with rasterio.open(outfile, 'w', **profile) as dst:

        # perform prediction on each small image patch to minimize required memory
        patch_size = 500

        for i in range((src.shape[0] // patch_size) + 1):
            for j in range((src.shape[1] // patch_size) + 1):
                # define the pixels to read (and write) with rasterio windows reading
                window = rasterio.windows.Window(
                    j * patch_size,
                    i * patch_size,
                    # don't read past the image bounds
                    min(patch_size, src.shape[1] - j * patch_size),
                    min(patch_size, src.shape[0] - i * patch_size))
                
                # read the image into the proper format
                data = src.read(window=window)
                
                # adding indices if necessary
                img_swp = np.moveaxis(data, 0, 2)
                img_flat = img_swp.reshape(-1, img_swp.shape[-1])

                m = np.ma.masked_invalid(img_flat)
                to_predict = img_flat[~m.mask].reshape(-1, img_flat.shape[-1])

                # skip empty inputs
                if not len(to_predict):
                    continue
                # predict
                img_preds = clf.predict(to_predict)

                # add the prediction back to the valid pixels (using only the first band of the mask to decide on validity)
                # makes the assumption that all bands have identical no-data value arrangements
                output = np.zeros(img_flat.shape[0])
                output[~m.mask[:, 0]] = img_preds.flatten()
                # resize to the original image dimensions
                output = output.reshape(*img_swp.shape[:-1])

                # create our final mask
                mask = (~m.mask[:, 0]).reshape(*img_swp.shape[:-1])

                # write to the final files
                dst.write(output.astype(rasterio.uint8), 1, window=window)
                dst.write_mask(mask, window=window)