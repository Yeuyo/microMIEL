# USE THIS 250701
import re, sys, time, os, math, copy

from pathlib import Path
import multiprocessing as mp

# Libaries
def object_processing(arg):
    import pandas as pd
    import numpy as np
    import tifffile
    masks, n, number_of_channels, filename, imgs, loadPath_cellpose, y_dim, x_dim = arg
    coords = pd.DataFrame (columns = ["Img.no","PV","z","y","x", "y.dim", "x.dim", "ch"])
    output_file_dimensions = pd.DataFrame (columns = ["Filename", "y.dim", "x.dim", "ch"])
    if n >= 1:
        z, y, x = ((np.where(masks == n))) # Get pixel coords
        print ("Getting object coordinates. Image: " + filename + ", PV: " + str(n) + ", Pixels: " + str(len(z)))
        PV = pd.Series ([n] * len(z))
        z = pd.Series(z)
        y = pd.Series(y)
        x = pd.Series(x)

        if (np.count_nonzero(y == 0) > 5) or (np.count_nonzero(x == 0) > 5) or (np.count_nonzero(y == y_dim) > 5) or (np.count_nonzero(x == x_dim) > 5):
            return output_file_dimensions
        # Calculate 3D dimensions of object
        number_of_z_in_mask = max(z) - min(z)
        number_of_y_in_mask = max(y) - min(y)
        number_of_x_in_mask = max(x) - min(x)
        new_image_for_mask_object = np.zeros((number_of_channels, number_of_z_in_mask + 1, number_of_y_in_mask, number_of_x_in_mask))

        # Record 3D coordinates, dimensions for each object
        coords_cols = {"Filename": filename, "PV": PV, "z": z, "y": y, "x": x, "y.dim": number_of_y_in_mask, "x.dim": number_of_x_in_mask}
        coords_temp = pd.DataFrame (coords_cols)
        coords = pd.concat ([coords, coords_temp], ignore_index = True, axis = 0)

        # TODO: Crop each mask from source image, perform MIP and save image.

        # Crop each object from source image, perform MIP and save image.
        # Iterate through each channel.
        for m in range(number_of_channels): # m = channel

            # Iterate through each z
            zRun = 0
            for o in range(min(z), max(z)): # o = z coordinates
                y_index_in_this_z = np.where(z == o)[0].tolist()
                # Check this if indexing error #new_image_for_mask_object[m, zRun, y[y_index_in_this_z] - 1 - min(y), x[y_index_in_this_z] - 1 - min(x)] = imgs[zRun, m, y[y_index_in_this_z], x[y_index_in_this_z]]
                new_image_for_mask_object[m, zRun, (y[y_index_in_this_z] - 1 - min(y)).clip(lower = 0), (x[y_index_in_this_z] - 1 - min(x)).clip(lower = 0)] = imgs[zRun, m, y[y_index_in_this_z], x[y_index_in_this_z]]
                zRun = zRun + 1

            # Perform MIPs and save images.
            MIP_image= np.max(new_image_for_mask_object[m], axis=0) # Perform MIP
            MIP_output_filename = filename + "_c" + str(n) + "_MIP" + "_Ch" + str(m) + ".tif"
            tifffile.imwrite(loadPath_cellpose + "cell_MIPs/" + MIP_output_filename, MIP_image)

            # Record 2D MIP file dimensions
            output_file_dimensions_cols = {"Filename": MIP_output_filename, "y.dim": number_of_y_in_mask, "x.dim": number_of_x_in_mask, "ch": m}
            output_file_dimensions_temp = pd.DataFrame (output_file_dimensions_cols, index = [0])
            output_file_dimensions = pd.concat ([output_file_dimensions, output_file_dimensions_temp], ignore_index = True, axis = 0)
    return output_file_dimensions

if __name__ == "__main__":
    do3DMIP = False
    if do3DMIP == True:
        cell_dia = 90
        channels = [[0,0]]
    useCellpose = True
    if useCellpose == True:
        from cellpose import models
        from cellpose.io import imread
        from cellpose import denoise
        from scipy import ndimage
        seg_model = "cyto3"
        cellDiam = 25
        use_denoise = True
        cellpose3D = False
    else:
        from epilands.image_segmentation import (
            segment_image_stardist2d,
            segment_image_stardist3d,
        )
    saveSegment = True

    ## Library Initialise
    from imagepubautomation.gputools import gpuinit
    from imagepubautomation.segmentation.modules import ImageObjectTransformer
    from imagepubautomation.feature_extraction.modules import ObjectFeatureTransformer
    from imagepubautomation.segmentation.processes import (
        normalize_segmentation_image,
        segment_images,
        resize_masks,
        resize_segmentation_image,
    )
    from imagepubautomation.feature_extraction.processes import (
        extract_features,
        devmultiprocessing_feature_extraction,
    )
    from epilands.image_preprocessing import glaylconvert
    import numpy as np
    import tifffile as tiff
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    from epilands.generic_read_write import (
        read_dataframe_from_h5_file,
        save_dataframe_to_h5_file
    )
    from epilands.image_read_write import (
        extract_imagelike_file_information,
        save_segmentation_data,
        read_segmentation_data,
        read_images
    )
    if saveSegment == True:
        import tifffile

    ## Start
    start_time = time.time()

    import logging
    logger = logging.getLogger("imagepubautomation")
    ############################################################################
    loadPathRoot = Path("//shared.sydney.edu.au/research-data/PRJ-DataT/Bitong's data/Bitong HiLo 405/Data for HILO405 analysis/Result") # edit
    savePathRoot = Path("//shared.sydney.edu.au/research-data/PRJ-DataT/Bitong's data/Bitong HiLo 405/Data for HILO405 analysis/Result") # edit
    search_pattern = ".tif"
    metadata_pattern = re.compile("r([0-9]{2})c([0-9]{2})f([0-9]{2})p([0-9]{2})-([a-zA-Z0-9]{3})")
    channelIndex = 4
    rowIndex = 0
    colIndex = 1
    zIndex = 0
    FOVIndex = 2
    tIndex = None
    if not os.path.isdir(savePathRoot):
        os.mkdir(savePathRoot)
    # list(os.walk(loadPathRoot)) #TODO:
    # folders = ["Pre", "Post", "Others"] # edit
    folders = ["C317T4_Pre", "C317T4_Post", "C317T4_Others", "C317T5_Pre", "C317T5_Post", "C317T5_Others",
               "C617T4_Pre", "C617T4_Post", "C617T4_Others", "C617T5_Pre", "C617T5_Post", "C617T5_Others"]
    folders = ["C612T6_Pre", "C612T6_Post", "C612T6_Others", "C317T6_Pre", "C317T6_Post", "C317T6_Others", "C617T6_Pre", "C617T6_Post", "C617T6_Others"]
    folders = ["C112_Pre", "C112_Post", "C112_Others",
               "C117_Pre", "C117_Post", "C117_Others",
               "C212_Pre", "C212_Post", "C212_Others",
               "C217_Pre", "C217_Post", "C217_Others",
               "C312_Pre", "C312_Post", "C312_Others",
               "C412_Pre", "C412_Post", "C412_Others",
               "C417_Pre", "C417_Post", "C417_Others",
               "C512_Pre", "C512_Post", "C512_Others",
               "C517_Pre", "C517_Post", "C517_Others",
               "C610_Pre", "C610_Post", "C610_Others",
               "C910_Pre", "C910_Post", "C910_Others"]
    # folders = ["C224T1_Pre", "C224T1_Post", "C224T1_Others",
    #            "C224T2_Pre", "C224T2_Post", "C224T2_Others",
    #            "C719T1_Pre", "C719T1_Post", "C719T1_Others",
    #            "C719T2_Pre", "C719T2_Post", "C719T2_Others"]
    # folders = os.listdir(loadPathRoot)
    # folders = folders[:-1]
    folders = ["C401T1_Pre", "C401T1_Post", "C401T1_Others",
               "C401T2_Pre", "C401T2_Post", "C401T2_Others",
               "C101_Pre", "C101_Post", "C101_Others",
               "C215_Pre", "C215_Post", "C215_Others",
               "C319_Pre", "C319_Post", "C319_Others",
               "C315_Pre", "C315_Post", "C315_Others",
               "C424_Pre", "C424_Post", "C424_Others",
               "C415_Pre", "C415_Post", "C415_Others",
               "C501_Pre", "C501_Post", "C501_Others",
               "C515_Pre", "C515_Post", "C515_Others",
               "C615_Pre", "C615_Post", "C615_Others",
               "C724_Pre", "C724_Post", "C724_Others",
               "C715_Pre", "C715_Post", "C715_Others",
               "C815_Pre", "C815_Post", "C815_Others",
               "C915_Pre", "C915_Post", "C915_Others",
               "C1015_Pre", "C1015_Post", "C1015_Others"]
    
    for n in range(len(folders)):
        folder = folders[n]
        assert os.path.exists(loadPathRoot)
        assert os.path.exists(savePathRoot)
        loadPath = (loadPathRoot / folder).resolve()
        # rename
        # files = os.listdir(loadPath)
        # for m in range(len(files)):
        #     cValue = math.floor(m / 100)
        #     rValue = m - (cValue * 100)
        #     os.rename(loadPath.__str__().replace("\\", "/") + "/" + files[m], loadPath.__str__().replace("\\", "/") + "/r" + f"{rValue:02d}" + "c" + f"{cValue:02d}" + "f00p00-HUVEC-" + files[m])
        # segmentation_files = []
        # name = (
        #     re.search(re.compile("[[]{1}[A-Za-z0-9_]+[]]{1}"), folder)[0]
        #     .replace("[", "")
        #     .replace("]", "")
        # )
        name = copy.deepcopy(folder)
        experiment_output_folder = os.path.join(savePathRoot, name)
        segmentation_path = experiment_output_folder + "/segmentation"
        mask_path = experiment_output_folder + "/segmentationMask"
        os.makedirs(experiment_output_folder, exist_ok=True)
        os.makedirs(segmentation_path, exist_ok=True)
        # image_file_information = extract_imagelike_file_information(
        #             file_path=loadPath,
        #             search_pattern=search_pattern,
        #             metadata_pattern=metadata_pattern,
        #             channelIndex=channelIndex,
        #             rowIndex=rowIndex,
        #             colIndex=colIndex,
        #             zIndex=zIndex,
        #             FOVIndex=FOVIndex,
        #             tIndex=tIndex,
        #         )

        # MIEL starts
        experiment_output_folder = (savePathRoot / name).resolve()
        assert experiment_output_folder.exists()

        try:
            feature_extraction_mod = ObjectFeatureTransformer(
                name="feature_extraction",
                object_image_directory="segmentation",
                output_directory=experiment_output_folder,
                search_pattern=".hdf5",
                metadata_pattern=re.compile("row([0-9]+)col([0-9]+)fov([0-9]+)"),
                channelIndex=None,
                rowIndex=0,
                colIndex=1,
                zIndex=None,
                FOVIndex=2,
                tIndex=None,
            )

            feature_extraction_mod.add_process(extract_features)
            feature_extraction_mod.run()

            # feature_extraction_mod.run_multiprocessing(
            #     process=devmultiprocessing_feature_extraction
            # )

        except Exception as e:
            logger.error(f"There was an error processing {name}, see below")
            logger.error(e)

    print("--- %s seconds ---" % (time.time() - start_time))
    1