#!/bin/bash

command_exists() {
    command -v "$1" &> /dev/null
}

install_python() {
    echo "Python not found. Installing Python..."
    
    if command_exists apt-get; then

        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv
    elif command_exists dnf; then
        
        sudo dnf update -y
        sudo dnf install -y python3 python3-pip python3-virtualenv
    elif command_exists pacman; then
        
        sudo pacman -Sy python python-pip
    elif command_exists zypper; then
        
        sudo zypper refresh
        sudo zypper install -y python3 python3-pip python3-virtualenv
    elif command_exists brew; then

        brew install python3
    else
        echo "Unsupported operating system or package manager not found."
        echo "Please install Python 3 manually and run this script again."
        exit 1
    fi
}

setup_venv() {
    if ! command_exists python3; then
        install_python
    fi
    
    if ! command_exists python3; then
        echo "Python installation failed. Please install Python 3 manually."
        exit 1
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Python version: $python_version"
    
    if ! command_exists pip3; then
        echo "pip3 not found. Installing pip..."
        if command_exists apt-get; then
            sudo apt-get install -y python3-pip
        elif command_exists dnf; then
            sudo dnf install -y python3-pip
        fi
    fi
}

echo "Checking Python installation..."
setup_venv

echo "Creating model directory..."
mkdir -p model
cd model || exit 1

echo "Downloading main.py..."
if command_exists curl; then
    curl -o main.py https://raw.githubusercontent.com/byui-cloudsociety/Trip_Planner/refs/heads/main/notebook/main.py?token=GHSAT0AAAAAAC2MBCSJXOLLBQZGVXK2MMFQZZ6KFLQ
elif command_exists wget; then
    wget -O main.py https://raw.githubusercontent.com/byui-cloudsociety/Trip_Planner/refs/heads/main/notebook/main.py?token=GHSAT0AAAAAAC2MBCSJXOLLBQZGVXK2MMFQZZ6KFLQ
else
    echo "Neither curl nor wget found. Please install either one."
    exit 1
fi

echo "Downloading poiTrainingData.csv..."
if command_exists curl; then
    curl -L -o poiTrainingData.csv "https://file.fergendergen.dev/f/b2efe0bfd3624b5a9762/?dl=1"
elif command_exists wget; then
    wget -O poiTrainingData.csv "https://file.fergendergen.dev/f/b2efe0bfd3624b5a9762/?dl=1"
fi

echo "Setting up virtual environment..."
python3 -m venv venv || {
    echo "Failed to create virtual environment. Installing python3-venv..."
    if command_exists apt-get; then
        sudo apt-get install -y python3-venv
    elif command_exists dnf; then
        sudo dnf install -y python3-virtualenv
    fi
    python3 -m venv venv
}

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "Virtual environment activation failed. Please check your Python installation."
    exit 1
fi

echo "Installing required packages..."
pip install --upgrade pip
pip install pandas numpy scikit-learn

echo "Running main.py..."
python main.py


echo "Run `deactivate` when done."