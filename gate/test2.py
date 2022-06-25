import subprocess

subprocess.run('conda activate environment-name && "conda activate yolov5" && conda deactivate', shell=True)