# Trash Throwing Action Detection

https://github.com/bhaskarshukla002/Trash-Throwing-Action-Detection/assets/86822762/ac6b9fe3-7055-4865-95a6-0af1f8aed410

## Overview
This project is designed to detect trash-throwing actions using computer vision techniques. It utilizes YOLOv8 for pose estimation to identify specific human actions related to throwing trash.

## Features
- *Pose Estimation*: Uses YOLOv8 for detecting human poses.
- *Action Detection*: Identifies the action of throwing trash.
- *Result Visualization*: Provides visual outputs of the detection process.

## Installation
1. Clone the repository:

       git clone https://github.com/bhaskarshukla002/Trash-Throwing-Action-Detection.git
    
3. Navigate to the project directory:

       cd Trash-Throwing-Action-Detection
    
4. Install the required dependencies:

       pip install opencv-python numpy torch ultralytics
    
## Usage
1. *Jupyter Notebook*:
    - Open main.ipynb to run the detection process step by step.
    
2. *Python Script*:
    - Run the detection script:
          python python_implementation.py
    

## Project Structure
- main.ipynb: Jupyter Notebook for interactive development and testing.
- python_implementation.py: Python script for action detection.
- result images/: Directory containing result images.
- videos/: Directory containing input video files.
- resultdemo.mp4: Demo video showcasing the detection results.
- yolov8n-pose.pt: Pre-trained model weights for YOLOv8 pose estimation.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License.

## Contact
For any inquiries, please contact the repository owner at bhaskarshukla002@gmail.com.

---

For more details, visit the [repository](https://github.com/bhaskarshukla002/Trash-Throwing-Action-Detection).
