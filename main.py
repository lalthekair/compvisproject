import subprocess

def run_yolov5_detection(source_path):
    command = f"python yolov5/detect.py --weights yolov5s.pt --source {source_path} --conf 0.5 --classes 2 5 7 3 6"
    subprocess.run(command, shell=True)

if __name__ == "__main__":
    source_path = "img6.jpg"
    run_yolov5_detection(source_path)
