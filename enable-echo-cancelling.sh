#!/bin/bash

# Colored message variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}This script enables the PulseAudio echo cancelling module.${NC}"
echo -e "${YELLOW}Note: This script should only be run on the host machine, NOT inside Docker!${NC}"
echo -e "${RED}Are you sure you want to proceed? (y/n)${NC}"
read -r answer

if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo -e "${RED}Aborting.${NC}"
    exit 1
fi

# Check if pactl is installed
if ! command -v pactl &> /dev/null; then
    echo -e "${RED}pactl could not be found. Please install PulseAudio.${NC}"
    exit 1
fi

# Load the echo cancellation module
sudo bash -c "echo 'load-module module-echo-cancel source_name=aec_source source_properties=device.description=aec_source sink_name=aec_sink sink_properties=device.description=aec_sink aec_method=webrtc channels=1' >> /etc/pulse/default.pa"
sudo bash -c "echo 'set-default-source aec_source' >> /etc/pulse/default.pa"
sudo bash -c "echo 'set-default-sink aec_sink' >> /etc/pulse/default.pa"
echo -e "${GREEN}Echo cancellation module configuration appended to /etc/pulse/default.pa.${NC}"
echo -e "${YELLOW}You may need to restart PulseAudio for the changes to take effect.${NC}"