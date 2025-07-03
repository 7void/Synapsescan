import os
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), ".kaggle")
import subprocess
subprocess.run([
	"kaggle", "datasets", "download",
	"-d", "sunilthite/ovarian-cancer-classification-dataset",
	"--unzip"
], check=True)
print('Data downloaded successfully')
