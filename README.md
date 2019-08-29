## Computer Vision Pipeline for robotic percetion

Computer Vision Pipeline for robotic perception tasks. This repoitory integrates am [Mask R-CNN](https://github.com/BerkeleyAutomation/sd-maskrcnn) to perform scene segmentation based on rdbd data. Furthermore a [Fully Convolutional Grasp Quality](https://github.com/BerkeleyAutomation/gqcnn) approach by Uni Berkeley is used to propose and evaluate grasp points for parallel jaw and suction cup gripper.


## Installation
1. Clone this repository https://github.com/ralfgulde/cv_pipeline/ 
2. Download pre-trained models
   ```bash
   chmod +x download_models.sh
   ./download_models.sh
   ``` 
3. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
