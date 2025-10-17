"""
Complete pipeline runner for DenseNet medical image classification
Runs training, evaluation, explainability analysis, and launches UI
"""
import os
import sys
import subprocess
import argparse
from datetime import datetime
import json


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False


def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_files = [
        'requirements_explainability.txt',
        'requirements_ui.txt'
    ]
    
    for req_file in required_files:
        if not os.path.exists(req_file):
            print(f"❌ Missing requirements file: {req_file}")
            return False
    
    # Check if key modules can be imported
    try:
        import torch
        import streamlit
        import plotly
        import sklearn
        print("✅ Key packages available")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install requirements with:")
        print("pip install -r requirements_explainability.txt")
        print("pip install -r requirements_ui.txt")
        return False


def main():
    parser = argparse.ArgumentParser(description='Complete DenseNet pipeline')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training step (use existing model)')
    parser.add_argument('--skip_evaluation', action='store_true',
                       help='Skip evaluation step')
    parser.add_argument('--skip_explainability', action='store_true',
                       help='Skip explainability analysis')
    parser.add_argument('--skip_ui', action='store_true',
                       help='Skip UI launch')
    parser.add_argument('--model_path', type=str, default="weight/save/40/iaff40_5.pth",
                       help='Path to model (for skipping training)')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples for explainability analysis')
    
    args = parser.parse_args()
    
    print("🔬 DenseNet Medical Image Classification - Complete Pipeline")
    print("="*70)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed. Please install missing packages.")
        return
    
    # Create directories
    os.makedirs("weight/save/40", exist_ok=True)
    os.makedirs("csv/40", exist_ok=True)
    
    pipeline_results = {
        'timestamp': datetime.now().isoformat(),
        'steps_completed': [],
        'steps_failed': [],
        'model_path': None,
        'evaluation_dir': None,
        'explainability_dir': None
    }
    
    # Step 1: Training
    if not args.skip_training:
        print("\n🎯 Step 1: Model Training")
        success = run_command("python train.py", "Training DenseNet model")
        
        if success:
            pipeline_results['steps_completed'].append('training')
            pipeline_results['model_path'] = args.model_path
        else:
            pipeline_results['steps_failed'].append('training')
            print("❌ Training failed. Check the error messages above.")
            return
    else:
        print("\n⏭️ Skipping training step")
        if os.path.exists(args.model_path):
            pipeline_results['model_path'] = args.model_path
            print(f"✅ Using existing model: {args.model_path}")
        else:
            print(f"❌ Model not found: {args.model_path}")
            return
    
    # Step 2: Evaluation
    if not args.skip_evaluation:
        print("\n📊 Step 2: Model Evaluation")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = f"evaluation_results_{timestamp}"
        
        eval_command = f"python run_evaluation.py --model_path {pipeline_results['model_path']} --save_dir {eval_dir}"
        success = run_command(eval_command, "Running comprehensive evaluation")
        
        if success:
            pipeline_results['steps_completed'].append('evaluation')
            pipeline_results['evaluation_dir'] = eval_dir
        else:
            pipeline_results['steps_failed'].append('evaluation')
            print("⚠️ Evaluation failed, but continuing with pipeline...")
    else:
        print("\n⏭️ Skipping evaluation step")
    
    # Step 3: Explainability Analysis
    if not args.skip_explainability:
        print("\n🧠 Step 3: Explainability Analysis")
        
        explain_command = f"python run_explainability.py --model_path {pipeline_results['model_path']} --num_samples {args.num_samples} --techniques gradcam"
        success = run_command(explain_command, "Running explainability analysis")
        
        if success:
            pipeline_results['steps_completed'].append('explainability')
            pipeline_results['explainability_dir'] = 'explainability'
        else:
            pipeline_results['steps_failed'].append('explainability')
            print("⚠️ Explainability analysis failed, but continuing with pipeline...")
    else:
        print("\n⏭️ Skipping explainability analysis")
    
    # Step 4: Launch UI
    if not args.skip_ui:
        print("\n🌐 Step 4: Launching Interactive UI")
        print("The Streamlit UI will open in your default web browser.")
        print("If it doesn't open automatically, go to: http://localhost:8501")
        print("\nPress Ctrl+C to stop the UI server when done.")
        
        ui_command = "streamlit run app.py"
        print(f"Command: {ui_command}")
        
        try:
            subprocess.run(ui_command, shell=True, check=True)
            pipeline_results['steps_completed'].append('ui_launch')
        except KeyboardInterrupt:
            print("\n🛑 UI server stopped by user")
            pipeline_results['steps_completed'].append('ui_launch')
        except subprocess.CalledProcessError as e:
            print(f"❌ UI launch failed: {e}")
            pipeline_results['steps_failed'].append('ui_launch')
    else:
        print("\n⏭️ Skipping UI launch")
        print("To launch UI manually, run: streamlit run app.py")
    
    # Save pipeline results
    results_file = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 PIPELINE SUMMARY")
    print("="*70)
    
    print(f"✅ Completed steps: {', '.join(pipeline_results['steps_completed'])}")
    if pipeline_results['steps_failed']:
        print(f"❌ Failed steps: {', '.join(pipeline_results['steps_failed'])}")
    
    if pipeline_results['model_path']:
        print(f"📁 Model: {pipeline_results['model_path']}")
    
    if pipeline_results['evaluation_dir']:
        print(f"📊 Evaluation results: {pipeline_results['evaluation_dir']}/")
    
    if pipeline_results['explainability_dir']:
        print(f"🧠 Explainability results: {pipeline_results['explainability_dir']}/")
    
    print(f"📋 Pipeline log: {results_file}")
    
    print("\n🚀 Next steps:")
    if 'ui_launch' not in pipeline_results['steps_completed']:
        print("- Launch UI: streamlit run app.py")
    print("- Check evaluation results for model performance")
    print("- Review explainability visualizations")
    print("- Use the interactive UI for testing new images")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()