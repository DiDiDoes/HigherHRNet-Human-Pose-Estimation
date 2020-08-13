# Prepare requirements
pip install -r requirements.txt

# Install CrowdPose API
cd ~
git clone https://github.com/Jeff-sjtu/CrowdPose.git
cd CrowdPose/crowdpose-api/PythonAPI
bash install.sh
cd ~/HigherHRNet-Human-Pose-Estimation

# Download model
cd models/pytorch/pose_coco
bash download.sh
bash download.sh
cd ../../../


