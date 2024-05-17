import subprocess

def run_yolov5_detection(source_path):
    command = f"python yolov5/detect.py --weights yolov5s.pt --source {source_path} --conf 0.5 --classes 2 5 7 3 6"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    source_path = "img6.jpg"
    run_yolov5_detection(source_path)

"""
Make sure that the image you want to run this on and the .py file are in the same directory.
Create a virtual environment in the directory, then install yolov5 model as per: https://github.com/ultralytics/yolov5/

Run these commands in the terminal one by one (after activating the virtual env)
git clone https://github.com/ultralytics/yolov5  # clone repository
cd yolov5
pip install -r requirements.txt  # install yolov5

Results of the runs are stored in yolov5\runs\detect\exp (number indicates run #)
"""
