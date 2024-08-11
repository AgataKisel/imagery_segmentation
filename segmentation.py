import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.filters
from osgeo import gdal

def segmentation_method_Otsu(input_path: str, output_path: str, num_classes: int=2, smoothing: int=6) -> int:
    """
    Performs segmentation of any gdal-compatible raster
    
    Parametrs
    ---------
    input_path: str
        path to source gdal-compatible raster

    num_clusters: int
        number of clusters to be created

    output_path: str
        path to create new raster in
        
     smoothing: int
         determines the smoothing strength

    Returns
    --------
    int 
        0 if finished correct
        1 if invalid data source
        2 if error in file creation
    Description 
    -----------
    based on https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_multiotsu
    """
    gdal.UseExceptions()
    gdal.AllRegister()
    img_ds = gdal.Open(input_path, gdal.GA_ReadOnly)

    if (img_ds==None):
        return 1 
    
    num_rasters = img_ds.RasterCount
    img = img_ds.ReadAsArray()
    if num_rasters > 1:
        img = np.moveaxis(img, 0, -1)
    else:
        img = img[..., np.newaxis]

    if len(img.shape) > 2:  
        img = img.mean(axis=2)
    median_filtered = scipy.ndimage.median_filter(img, size=smoothing)
    plt.imshow(median_filtered, cmap='gray')
    plt.axis('off')
    plt.title('median filtered image')


    threshold = skimage.filters.threshold_multiotsu(median_filtered, classes=num_classes)

    res = np.zeros(median_filtered.shape)
    res = np.where(median_filtered <= threshold[0], 1, res)
    for i in range(1, len(threshold) - 1):
        res = np.where(
            (median_filtered > threshold[i]) & (median_filtered <= threshold[i+1]), 
            i + 1, 
            res
        )
    res = np.where(median_filtered > threshold[-1], len(threshold) + 1, res)
    # predicted = np.uint8(median_filtered > threshold) * 255
    plt.imshow(res)
    plt.axis('off')
    
    try:
        format = "GTiff"
        driver = gdal.GetDriverByName(format)
        out_data_raster = driver.Create(output_path, img_ds.RasterXSize, img_ds.RasterYSize, 1, gdal.GDT_Byte)
        out_data_raster.SetGeoTransform(img_ds.GetGeoTransform())
        out_data_raster.SetProjection(img_ds.GetProjection())
        
        out_data_raster.GetRasterBand(1).WriteArray(res)
        out_data_raster.FlushCache() 
        del out_data_raster
    except:
        return 2
    return 0