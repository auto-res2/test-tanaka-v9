import os
import sys
import argparse
import torch
from preprocess import get_data_loader
from train import BackdoorModule, CamouflageModule, CamouflageMLP, CamouflageResNet, CamouflageTransformer, SurrogateDefense
from evaluate import run_dual_objective_experiment, run_architecture_ablation, run_defense_experiment, DEFAULT_LAMBDAS

def test_run_experiments(device):
    print("\n----- Running Test Experiments -----")
    
    test_loader = get_data_loader(num_samples=2, batch_size=2, test_mode=True)
    
    backdoor_model = BackdoorModule().to(device)
    camouflage_model = CamouflageModule().to(device)
    
    print("\n[Experiment 1]: Dual-Objective Loss Sensitivity Analysis (Test Mode)")
    run_dual_objective_experiment(backdoor_model, camouflage_model, test_loader, device, 
                                  lambdas=[DEFAULT_LAMBDAS[0]], epochs=1)
    
    print("\n[Experiment 2]: Camouflage Network Architecture Ablation (Test Mode)")
    camouflage_variants = {
        "MLP": CamouflageMLP().to(device),
        "ResNet": CamouflageResNet().to(device),
        "Transformer": CamouflageTransformer().to(device)
    }
    run_architecture_ablation(backdoor_model, camouflage_variants, test_loader, device, epochs=1)
    
    print("\n[Experiment 3]: Simulated Defense-In-The-Loop Effectiveness (Test Mode)")
    defense_module = SurrogateDefense().to(device)
    run_defense_experiment(backdoor_model, camouflage_model, defense_module, test_loader, device, epochs=1)
    
    print("----- Test Experiments Completed -----\n")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help="Run in test mode for quick execution")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('.research/iteration1/images', exist_ok=True)
    
    loader = get_data_loader(num_samples=10, batch_size=2, test_mode=False)
    
    backdoor_model = BackdoorModule().to(device)
    camouflage_model = CamouflageModule().to(device)
    
    if args.test:
        test_run_experiments(device)
    
    print("\n----- Running Full Experiments -----\n")
    
    print("\n[Experiment 1]: Dual-Objective Loss Sensitivity Analysis")
    dual_results = run_dual_objective_experiment(backdoor_model, camouflage_model, loader, device)
    
    print("\n[Experiment 2]: Camouflage Network Architecture Ablation")
    camouflage_variants = {
        "MLP": CamouflageMLP(),
        "ResNet": CamouflageResNet(),
        "Transformer": CamouflageTransformer()
    }
    ablation_results = run_architecture_ablation(backdoor_model, camouflage_variants, loader, device)
    
    print("\n[Experiment 3]: Simulated Defense-In-The-Loop Effectiveness")
    defense_module = SurrogateDefense().to(device)
    defense_results = run_defense_experiment(backdoor_model, camouflage_model, defense_module, loader, device)
    
    print("\n----- Full Experiments Completed -----\n")
    print("status_enum: stopped")

if __name__ == '__main__':
    main()
