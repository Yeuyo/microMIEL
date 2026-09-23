import os, sys, math
import numpy as np
import pandas as pd
import tifffile
import plotly_express as px
from skimage.filters import threshold_multiotsu

from epilands.image_read_write import save_segmentation_data
from scipy import ndimage

folSDir = r"\\shared.sydney.edu.au\research-data\PRJ-DataT\Bitong\Bitong HiLo 405\HILO405 without SOX2 Type III"
folSDir = r"\\shared.sydney.edu.au\research-data\PRJ-DataT\Bitong\Bitong HiLo 405\HILO405 without SOX2 Type III\New"
outputDir = r"\\shared.sydney.edu.au\research-data\PRJ-DataT\Bitong's data\Bitong HiLo 405\Data for HILO405 analysis\Result"

cellFol = ["Cell 1_1001", "Cell 2_1015", "Cell 3_0919", "Cell 3_1015", "Cell 4_0924", "Cell 4_1015", "Cell 5_1001", "Cell 5_1015",
           "Cell 6_1015", "Cell 7_0924", "Cell 7_1015", "Cell 8_1015", "Cell 9_1015", "Cell 10_1015"]
saveFold = ["C101", "C215", "C319", "C315", "C424", "C415", "C501", "C515",
            "C615", "C724", "C715", "C815", "C915", "C1015"]
smtCorrections = [[35, -38], [35, -36], [32, -36], [35, -36], [35, -36], [35, -36], [34, -38], [35, -36],
                  [35, -36], [34, -37], [35, -36], [35, -36], [35, -38], [35, -36]]
curExposure = 20 # Don't edit
tarExposure = 500 # Don't edit
smtCorrection = [35, -38]
pxSize = 0.11 # Don't edit
# preBurstFrame = [0, 27]
# postBurstFrame = [28, 60]
preBurstFrame = [0, 61]
postBurstFrame = [62, 120]

## DO NOT EDIT BELOW THIS LINE
def createfov(dim_y, dim_x, center, rad):
    masks = np.zeros((dim_y, dim_x))
    masks[int(center[0]), int(center[1])] = 1
    n = 0
    masks[int(center[0]), int(center[1]) - rad : int(center[1]) + rad + 1] = [0] * n + [1] * (((2 * rad) + 1) - (2 * n)) + [0] * n
    for n in range(1, rad + 1):
        masks[int(center[0]) - n, int(center[1]) - rad : int(center[1]) + rad + 1] = [0] * n + [1] * (((2 * rad) + 1) - (2 * n)) + [0] * n
        masks[int(center[0]) + n, int(center[1]) - rad : int(center[1]) + rad + 1] = [0] * n + [1] * (((2 * rad) + 1) - (2 * n)) + [0] * n
    # fig = px.imshow(masks)
    # fig.show(renderer = 'browser')
    return masks


for fol in range(len(cellFol)):
    folDir = folSDir + "/" + cellFol[fol]
    smtCorrection = smtCorrections[fol]
    files = [file for file in os.listdir(folDir) if file.endswith(".tif")]
    files.sort()
    xlfiles = [file for file in os.listdir(folDir) if file.endswith(".xls")]
    xlfiles.sort()

    outputDir1 = outputDir + "/" + saveFold[fol] + "_Pre/segmentation"
    outputDir2 = outputDir + "/" + saveFold[fol] + "_Post/segmentation"
    outputDir3 = outputDir + "/" + saveFold[fol] + "_Others/segmentation"
    os.makedirs(outputDir1, exist_ok=True)
    os.makedirs(outputDir2, exist_ok=True)
    os.makedirs(outputDir3, exist_ok=True)
    for n in range(len(files)):
        image = tifffile.imread(folDir + "/" + files[n]).astype(np.float32)
        data = pd.read_excel(folDir + "/" + xlfiles[n])
        for m in range(0, np.shape(image)[0], 3):
            coord = np.round(data.loc[m, ["GFP x Coordinate", "GFP y Coordinate"]].tolist())
            coord = coord - smtCorrection
            masks = createfov(np.shape(image)[1], np.shape(image)[2], coord, 4)
            points_data = np.floor(np.array([ndimage.center_of_mass(masks == label) for label in np.unique(masks)[masks.max() > 0:]]))
            row_col_fov = [m - (math.floor(m / 100) * 100), math.floor(m / 100), 0]
            final_file_name = f"row{row_col_fov[0]}col{row_col_fov[1]}fov{row_col_fov[2]}_segmented.hdf5"

            # fig = px.imshow(masks * 3500)
            # fig.add_trace(px.imshow(image[0]).data[0])
            # fig.data[1].opacity = 0.2
            # fig.show(renderer = "browser")
            if m >= preBurstFrame[0] and (m <= preBurstFrame[1]):
                save_segmentation_data(path = outputDir1, filename = final_file_name, image_data = {("HUV", ): image[m]} , masks = masks.astype(int), details = {'coord': [], 'points': points_data, 'prob': [1]}, objects =[1])
                tifffile.imwrite(outputDir1 + "/" + files[n] + "_" + str(m) + ".tif", image[m], extratags = [(305, 's', i, "") for i in range(np.shape(image)[0] + 1)])
            elif m >= postBurstFrame[0] and (m <= postBurstFrame[1]):
                save_segmentation_data(path = outputDir2, filename = final_file_name, image_data = {("HUV", ): image[m]} , masks = masks.astype(int), details = {'coord': [], 'points': points_data, 'prob': [1]}, objects =[1])
                tifffile.imwrite(outputDir2 + "/" + files[n] + "_" + str(m) + ".tif", image[m], extratags = [(305, 's', i, "") for i in range(np.shape(image)[0] + 1)])
            else:
                save_segmentation_data(path = outputDir3, filename = final_file_name, image_data = {("HUV", ): image[m]} , masks = masks.astype(int), details = {'coord': [], 'points': points_data, 'prob': [1]}, objects =[1])
                tifffile.imwrite(outputDir3 + "/" + files[n] + "_" + str(m) + ".tif", image[m], extratags = [(305, 's', i, "") for i in range(np.shape(image)[0] + 1)])
            # if m >= (np.shape(image)[0] / 2):
            #     tifffile.imwrite(outputDir2 + "/" + files[n] + "_" + str(m) + ".tif", image[m], extratags = [(305, 's', i, "") for i in range(np.shape(image)[0] + 1)])
            # else:
            #     tifffile.imwrite(outputDir1 + "/" + files[n] + "_" + str(m) + ".tif", image[m], extratags = [(305, 's', i, "") for i in range(np.shape(image)[0] + 1)])
