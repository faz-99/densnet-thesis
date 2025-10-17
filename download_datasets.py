import os
import json

# Load kaggle credentials from local file
with open('kaggle.json', 'r') as f:
    creds = json.load(f)

os.environ['KAGGLE_USERNAME'] = creds['username']
os.environ['KAGGLE_KEY'] = creds['key']

import kaggle

def download_datasets():
    """Download BreakHis and ICIAR2018 breast cancer datasets"""
    
    # Create datasets directory
    os.makedirs('datasets', exist_ok=True)
    os.chdir('datasets')
    
    # Download BreakHis dataset
    print("Downloading BreakHis dataset...")
    kaggle.api.dataset_download_files('forderation/breakhis-400x', unzip=True)
    
    # Download ICIAR2018 dataset  
    print("Downloading ICIAR2018 dataset...")
    kaggle.api.dataset_download_files('paultimothymooney/breast-histopathology-images', unzip=True)
    
    print("Datasets downloaded successfully!")

if __name__ == "__main__":
    download_datasets()