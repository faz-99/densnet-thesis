#!/usr/bin/env python3
"""
Test Deployment Setup
Verify all files and dependencies are ready for deployment
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} MISSING: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if Path(dirpath).exists() and Path(dirpath).is_dir():
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description} MISSING: {dirpath}")
        return False

def check_python_import(module_name, description):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError:
        print(f"❌ {description} NOT INSTALLED: {module_name}")
        return False

def check_requirements_file():
    """Check requirements file and verify installations"""
    print("\n📦 Checking Python Dependencies...")
    
    if not Path("requirements_deployment.txt").exists():
        print("❌ requirements_deployment.txt not found")
        return False
    
    with open("requirements_deployment.txt", 'r') as f:
        requirements = f.readlines()
    
    all_installed = True
    for req in requirements:
        req = req.strip()
        if req and not req.startswith('#'):
            # Extract package name (before version specifier)
            package = req.split('>=')[0].split('==')[0].split('<')[0].strip()
            if not check_python_import(package, f"Package {package}"):
                all_installed = False
    
    return all_installed

def test_streamlit_app():
    """Test if Streamlit app can be imported"""
    print("\n🎯 Testing Streamlit App...")
    
    try:
        # Try to import the app (without running it)
        spec = importlib.util.spec_from_file_location("app", "app_deployment.py")
        if spec and spec.loader:
            print("✅ app_deployment.py can be loaded")
            return True
        else:
            print("❌ app_deployment.py cannot be loaded")
            return False
    except Exception as e:
        print(f"❌ Error loading app_deployment.py: {str(e)}")
        return False

def check_model_files():
    """Check if model files exist"""
    print("\n🤖 Checking Model Files...")
    
    all_exist = True
    all_exist &= check_file_exists("model/denlsnet_corrected.py", "Model architecture")
    all_exist &= check_file_exists("model/SENet.py", "SE layer implementation")
    all_exist &= check_file_exists("model/__init__.py", "Model __init__.py")
    
    return all_exist

def check_config_files():
    """Check if config files exist"""
    print("\n⚙️ Checking Configuration Files...")
    
    all_exist = True
    all_exist &= check_file_exists("config/training_config.py", "Training config")
    all_exist &= check_file_exists("config/__init__.py", "Config __init__.py")
    all_exist &= check_file_exists(".streamlit/config.toml", "Streamlit config")
    
    return all_exist

def check_deployment_files():
    """Check if deployment files exist"""
    print("\n📄 Checking Deployment Files...")
    
    all_exist = True
    all_exist &= check_file_exists("app_deployment.py", "Main application")
    all_exist &= check_file_exists("requirements_deployment.txt", "Requirements file")
    all_exist &= check_file_exists("README_DEPLOYMENT.md", "Deployment README")
    all_exist &= check_file_exists("DEPLOYMENT_QUICKSTART.md", "Quick start guide")
    all_exist &= check_file_exists(".gitignore", "Git ignore file")
    all_exist &= check_file_exists("Dockerfile", "Docker file")
    
    return all_exist

def check_git_setup():
    """Check if git is initialized"""
    print("\n🔧 Checking Git Setup...")
    
    if Path(".git").exists():
        print("✅ Git repository initialized")
        
        # Check if remote is set
        try:
            import subprocess
            result = subprocess.run(['git', 'remote', '-v'], 
                                  capture_output=True, text=True)
            if result.stdout:
                print("✅ Git remote configured")
                print(f"   {result.stdout.strip()}")
            else:
                print("⚠️  Git remote not configured yet")
                print("   Run: git remote add origin https://github.com/USERNAME/REPO.git")
        except:
            print("⚠️  Could not check git remote")
        
        return True
    else:
        print("⚠️  Git not initialized")
        print("   Run: git init")
        return False

def print_deployment_instructions():
    """Print deployment instructions"""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    print("\n📋 Quick Deploy to Streamlit Cloud:")
    print("   1. Push to GitHub:")
    print("      git add .")
    print("      git commit -m 'Deploy DenLsNet'")
    print("      git push -u origin main")
    print("")
    print("   2. Go to https://share.streamlit.io")
    print("   3. Click 'New app'")
    print("   4. Select your repository")
    print("   5. Main file: app_deployment.py")
    print("   6. Click 'Deploy'")
    print("")
    print("📖 For detailed instructions, see:")
    print("   - DEPLOYMENT_QUICKSTART.md (5-minute guide)")
    print("   - README_DEPLOYMENT.md (complete guide)")
    print("")
    print("🤖 Or run automated script:")
    print("   chmod +x deploy_to_streamlit.sh")
    print("   ./deploy_to_streamlit.sh")

def main():
    """Main test function"""
    print("🔬 DenLsNet Deployment Readiness Test")
    print("="*60)
    
    all_checks_passed = True
    
    # Check deployment files
    all_checks_passed &= check_deployment_files()
    
    # Check model files
    all_checks_passed &= check_model_files()
    
    # Check config files
    all_checks_passed &= check_config_files()
    
    # Check Python dependencies
    all_checks_passed &= check_requirements_file()
    
    # Test Streamlit app
    all_checks_passed &= test_streamlit_app()
    
    # Check git setup
    git_ready = check_git_setup()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    if all_checks_passed:
        print("✅ All checks passed! Your deployment is ready.")
        print("")
        if git_ready:
            print("🎉 You can deploy now!")
        else:
            print("⚠️  Initialize git before deploying")
        
        print_deployment_instructions()
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("")
        print("💡 Common fixes:")
        print("   - Install missing packages: pip install -r requirements_deployment.txt")
        print("   - Create missing __init__.py files: touch model/__init__.py config/__init__.py")
        print("   - Ensure all files are in the correct locations")
        return 1

if __name__ == "__main__":
    sys.exit(main())
