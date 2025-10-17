import os
import subprocess

# Set environment variables from kaggle.json
import json
with open('kaggle.json', 'r') as f:
    creds = json.load(f)

os.environ['KAGGLE_USERNAME'] = creds['username']
os.environ['KAGGLE_KEY'] = creds['key']

# Create datasets directory
os.makedirs('datasets', exist_ok=True)

# Download datasets using kaggle CLI
print("Downloading BreakHis dataset...")
subprocess.run(['kaggle', 'datasets', 'download', '-d', 'forderation/breakhis-400x', '-p', 'datasets', '--unzip'])

print("Downloading ICIAR2018 dataset...")
subprocess.run(['kaggle', 'datasets', 'download', '-d', 'paultimothymooney/breast-histopathology-images', '-p', 'datasets', '--unzip'])

print("Done!")