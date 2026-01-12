#!/bin/bash

#sudo apt -y install nsight-compute nsight-systems libboost-program-options-dev
sudo apt -y install nsight-compute libboost-program-options-dev

# newer version
# instead of lambda.ai provided NVIDIA Nsight Systems version 2024.6.2.225-246235244400v0
mkdir ~/Downloads
wget -P ~/Downloads https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_6/NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb
sudo dpkg -i ~/Downloads/NsightSystems-linux-cli-public-2025.6.1.190-3689520.deb

# maybe?
#export PATH=$PATH:/usr/lib/nsight-systems/host-linux-x64
