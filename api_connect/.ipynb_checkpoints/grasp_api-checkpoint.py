# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%

import json
import numpy as np
import matplotlib.pyplot as plt
try:
    import pyrealsense2 as rs
except ImportError:
    print("Unable to import pyrealsense2.")


class RealsenseSensor():
    """Class for interacting with a RealSense D400-series sensor.

    pyrealsense2 should be installed from source with the following
    commands:
    
    FIRST TRY PIP INSTALL!!!! If this does not work for you, then go ahead:

    >>> git clone https://github.com/IntelRealSense/librealsense
    >>> cd librealsense
    >>> mkdir build
    >>> cd build
    >>> cmake .. \
        -DBUILD_EXAMPLES=true \
        -DBUILD_WITH_OPENMP=false \
        -DHWM_OVER_XU=false \
        -DBUILD_PYTHON_BINDINGS=true \
        -DPYTHON_EXECUTABLE:FILEPATH=/path/to/your/python/library/ \
        -G Unix\ Makefiles
    >>> make -j4
    >>> sudo make install
    >>> export PYTHONPATH=$PYTHONPATH:/usr/local/lib
    """

    def __init__(self, sn):
        self._running = None,
        self.id = sn
        self._pipe = rs.pipeline()
        self._rs_cfg = rs.config()
        self._align = rs.align(rs.stream.color)
        self._intrinsics = {}
        ctx = rs.context()
        # prints out cams and S/N, which should be defined as id in config
        for d in ctx.devices:
            print("Cam {} connected.".format(d))
        
  

    def __del__(self):
        if self._running:
            self.stop()

    def _setup_pipe(self):
        self._rs_cfg.enable_device(self.id)
        self._rs_cfg.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.bgr8,
            30
        )
        self._rs_cfg.enable_stream(
            rs.stream.depth,
            640,
            480,
            rs.format.z16,
            30
        )

    def _get_intrinsics(self):
        stream = self._profile.get_stream(rs.stream.color)
        obj = stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics = obj

    def start(self):
        self._setup_pipe()
        self._profile = self._pipe.start(self._rs_cfg)
        self._get_intrinsics()
        self._set_depth_scale()
        for _ in range(5):
            self._pipe.wait_for_frames()
        self._running = True

    def stop(self):
        if not self._running:
            return False
        self._pipe.stop()
        self._running = False
        return True

    def _to_numpy(self, frame, dtype):
        arr = np.asanyarray(frame.get_data(), dtype=dtype)
        return arr

    def _read_image(self):
        
        frames = self._pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        depth_image = self._to_numpy(depth_frame, np.float32)

        depth_image *= self._depth_scale
        color_im = self._to_numpy(color_frame, np.uint8)
        return color_im, depth_image
    
    
    def _set_depth_scale(self):
        sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = sensor.get_depth_scale()
    
    def frames(self):
        color_im, depth_im = self._read_image()
        return color_im, depth_im


#%%
import sampleClient
import cv2
sensor = RealsenseSensor("821312061822")
sensor.start()


#%%
img_bgr, d = sensor.frames()
d.dtype
#


#%%
plt.imshow(img_bgr)
#%%
plt.imshow(d)


#%%
intrinsics = {
    "cx": sensor._intrinsics.ppx,
    "cy": sensor._intrinsics.ppy,
    "fx": sensor._intrinsics.fx,
    "fy": sensor._intrinsics.fy
}




#%%
d_ = cv2.medianBlur(d, 5)
#%%
np.mean(d_)
#%%
mask = sampleClient.predictMask(d_, **intrinsics, host='http://141.72.237.12:5000')

#%%
#sampleClient.predictGQCNN_pj(img_bgr, d, **intrinsics, host='http://141.72.237.12:5000')

#%%
sampleClient.predictFCGQCNN_pj(img_bgr, d, mask['masks'][0],  **intrinsics, host='http://141.72.237.12:5000')
#%%
mask['masks'][0][0]




#%%
import cv2
from matplotlib import colors
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


#%%
#dm, m = vision.maskDepth(d, 0.7,1.4 )
#m = cv2.medianBlur(m.astype(np.uint8), 5)
m = d
segmask = sampleClient.predictMask(d, **intrinsics, host='http://141.72.237.12:5000')
print("sampled mask")
image = img_bgr.copy()
rcolors = np.random.randint(0, len(STANDARD_COLORS), size=len(segmask["masks"]))
for i, mask in enumerate(segmask["masks"]):
    c = colors.to_rgb(STANDARD_COLORS[rcolors[i]])
    mask = np.bitwise_and(mask.astype(np.bool), m.astype(np.bool))
    colored = np.ones((*mask.shape, 3)) * c
    colored[~mask.astype(np.bool)] = 0
    image = cv2.addWeighted(image.astype(np.uint8),1, (colored * 255).astype(np.uint8), 0.9, 0)
image=image
#%%
plt.imshow(image)

#%%
