from pathlib import Path
import nd2
import numpy as np
import cv2

"""
Can hold either an ND2 file or a series of images
"""

class ImageData:
    def __init__(self, data, is_nd2=False):
        self.data = data
        self.processed_images = []
        self.is_nd2 = is_nd2
        self.segmentation_cache = {}
        self.has_channels = False
        self.dimensions = None

    def load_from_folder(folder_path):
        p = Path(folder_path)

        images = p.iterdir()
        # images = filter(lambda x : x.name.lower().endswith(('.tif')), images)
        img_filelist = sorted(images, key=lambda x: int(x.stem))

        preproc_img = lambda img: img  # Placeholder for now
        loaded = np.array([preproc_img(cv2.imread(str(_img))) for _img in img_filelist])

        _image_data = ImageData(loaded, is_nd2=False)

        print(f"Loaded dataset: {_image_data.data.shape}")
        
        # self.info_label.setText(f"Dataset size: {_image_data.data.shape}")
        # QMessageBox.about(
        #     self, "Import", f"Loaded {_image_data.data.shape[0]} pictures"
        # )

        _image_data.phc_path = folder_path

        _image_data.segmentation_cache.clear()  # Clear segmentation cache
        print("Segmentation cache cleared.")
        
        return _image_data
    
    def load_nd2_file(file_path):

        with nd2.ND2File(file_path) as nd2_file:
            _nd2_dims = nd2_file.sizes
            info_text = f"Number of dimensions: {nd2_file.sizes}\n"

            # for dim, size in _nd2_dims.items():
            #     info_text += f"{dim}: {size}\n"

            _image_data = ImageData(nd2.imread(file_path, dask=True), is_nd2=True)
            
            if "C" in _nd2_dims.keys():
                _image_data.has_channels = True
                _image_data.dimensions = _nd2_dims["C"]
            else:
                _image_data.has_channels = False

            return _image_data, _nd2_dims

        return None

