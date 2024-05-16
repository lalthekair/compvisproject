import subprocess

def run_yolov5_detection(source_path):
    command = f"python yolov5/detect.py --weights yolov5s.pt --source {source_path} --conf 0.5 --classes 0 2 5 7 3 4"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    source_path = "vid1.mp4"
    run_yolov5_detection(source_path)
