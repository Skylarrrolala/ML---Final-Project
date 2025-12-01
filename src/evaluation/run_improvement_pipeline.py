"""
Model Improvement Orchestrator
Runs the complete pipeline to boost model performance
"""

import subprocess
import sys
from pathlib import Path
import json
from datetime import datetime


class ModelImprovementOrchestrator:
    """Orchestrate the complete model improvement pipeline"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.summary = {
            'timestamp': datetime.now().isoformat(),
            'steps_completed': [],
            'improvements': {}
        }
    
    def run_step(self, step_name, script_path):
        """Run a pipeline step and track results"""
        print("\n" + "="*80)
        print(f"STEP: {step_name}")
        print("="*80)
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            print(result.stdout)
            
            self.summary['steps_completed'].append({
                'name': step_name,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error in {step_name}:")
            print(e.stderr)
            
            self.summary['steps_completed'].append({
                'name': step_name,
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            })
            
            return False
    
    def compare_results(self, baseline_path, improved_path):
        """Compare baseline vs improved metrics"""
        
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        # Load improved results from various sources
        improvements = {}
        
        # Check stacking results
        stacking_path = Path('results/ensemble/stacking_results.json')
        if stacking_path.exists():
            with open(stacking_path) as f:
                stacking = json.load(f)
                improvements['stacking'] = stacking
        
        # Compare
        baseline_mape = baseline['metrics']['ensemble']['mape']
        
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        print(f"\nBaseline Ensemble MAPE: {baseline_mape:.2f}%")
        
        if 'ensemble_stacking' in improvements.get('stacking', {}):
            stacking_mape = improvements['stacking']['ensemble_stacking']['mape']
            improvement_pct = ((baseline_mape - stacking_mape) / baseline_mape) * 100
            
            print(f"Stacking Ensemble MAPE: {stacking_mape:.2f}%")
            print(f"Improvement: {improvement_pct:.1f}%")
            
            self.summary['improvements']['mape'] = {
                'baseline': baseline_mape,
                'improved': stacking_mape,
                'improvement_pct': improvement_pct
            }
    
    def run_complete_pipeline(self):
        """Run the complete improvement pipeline"""
        
        print("="*80)
        print("MODEL IMPROVEMENT PIPELINE")
        print("Starting at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*80)
        
        # Step 1: Baseline Evaluation
        print("\n📊 Step 1/5: Baseline Evaluation")
        success = self.run_step(
            "Baseline Evaluation",
            "src/evaluation/baseline_evaluation.py"
        )
        
        if not success:
            print("\n⚠️  Baseline evaluation failed. Please check the error above.")
            return
        
        # Step 2: Feature Engineering
        print("\n🔧 Step 2/5: Feature Engineering")
        success = self.run_step(
            "Feature Engineering",
            "src/evaluation/feature_engineering.py"
        )
        
        if not success:
            print("\n⚠️  Feature engineering failed. Continuing with remaining steps...")
        
        # Step 3: Hyperparameter Tuning (optional - takes time)
        print("\n⚙️  Step 3/5: Hyperparameter Tuning")
        print("(This step can take 30-60 minutes. Skip for faster results.)")
        
        user_input = input("Run hyperparameter tuning? (y/n): ").lower()
        
        if user_input == 'y':
            success = self.run_step(
                "Hyperparameter Tuning",
                "src/evaluation/hyperparameter_tuning.py"
            )
        else:
            print("Skipping hyperparameter tuning.")
            self.summary['steps_completed'].append({
                'name': 'Hyperparameter Tuning',
                'status': 'skipped',
                'timestamp': datetime.now().isoformat()
            })
        
        # Step 4: Advanced Ensemble
        print("\n🎯 Step 4/5: Advanced Ensemble (Stacking)")
        success = self.run_step(
            "Advanced Ensemble",
            "src/evaluation/advanced_ensemble.py"
        )
        
        if not success:
            print("\n⚠️  Advanced ensemble failed. Continuing...")
        
        # Step 5: Uncertainty Quantification
        print("\n📈 Step 5/5: Uncertainty Quantification")
        success = self.run_step(
            "Uncertainty Quantification",
            "src/evaluation/uncertainty_quantification.py"
        )
        
        # Compare results
        baseline_results = Path('results/metrics/baseline_results.json')
        if baseline_results.exists():
            self.compare_results(baseline_results, None)
        
        # Save summary
        summary_path = self.results_dir / 'improvement_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"\nSummary saved to: {summary_path}")
        
        # Print summary
        print("\n📋 EXECUTION SUMMARY:")
        for step in self.summary['steps_completed']:
            status_icon = "✅" if step['status'] == 'success' else "⏭️ " if step['status'] == 'skipped' else "❌"
            print(f"  {status_icon} {step['name']}: {step['status']}")
        
        if 'mape' in self.summary['improvements']:
            imp = self.summary['improvements']['mape']
            print(f"\n🎉 PERFORMANCE IMPROVEMENT:")
            print(f"  Baseline MAPE: {imp['baseline']:.2f}%")
            print(f"  Improved MAPE: {imp['improved']:.2f}%")
            print(f"  Improvement: {imp['improvement_pct']:.1f}%")


if __name__ == "__main__":
    orchestrator = ModelImprovementOrchestrator()
    orchestrator.run_complete_pipeline()
