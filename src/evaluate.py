import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from train import backdoor_activation_loss, defense_evasion_loss

DEFAULT_LAMBDAS = [(0.1, 1.0), (0.5, 0.5), (1.0, 0.1)]

def run_dual_objective_experiment(backdoor_model, camouflage_model, data_loader, device, 
                                  lambdas=DEFAULT_LAMBDAS, epochs=5):
    results = {}
    for lam1, lam2 in lambdas:
        activation_rates = []
        latent_deviations = []
        optimizer = optim.Adam(list(backdoor_model.parameters()) + list(camouflage_model.parameters()), lr=1e-3)
        print("Testing weights: λ1 = {}, λ2 = {}".format(lam1, lam2))
        for epoch in range(epochs):
            for images, captions in data_loader:
                images = images.to(device)
                benign_latent = images.view(images.size(0), -1)[:, :512]
                guidance = torch.randn(images.size(0), 512).to(device)
                
                backdoor_out = backdoor_model(benign_latent, guidance)
                poisoned_latent = camouflage_model(benign_latent)
                
                loss_activation = backdoor_activation_loss(backdoor_out, guidance)
                loss_evasion = defense_evasion_loss(poisoned_latent, benign_latent)
                loss = lam1 * loss_activation + lam2 * loss_evasion
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                activation_rates.append(1 if loss_activation.item() < 0.5 else 0)
                latent_deviations.append(loss_evasion.item())
        avg_activation_rate = np.mean(activation_rates)
        avg_deviation = np.mean(latent_deviations)
        results[(lam1, lam2)] = {"Activation Rate": avg_activation_rate, "Latent Deviation": avg_deviation}
        print("Weights {}: Activation Rate = {:.3f}, Latent Deviation = {:.3f}".format((lam1, lam2), avg_activation_rate, avg_deviation))
        
    lamb_str = [f"({l1},{l2})" for l1, l2 in results.keys()]
    activation_vals = [results[k]["Activation Rate"] for k in results.keys()]
    deviation_vals = [results[k]["Latent Deviation"] for k in results.keys()]

    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    axes[0].bar(lamb_str, activation_vals, color='skyblue')
    axes[0].set_xlabel('Lambda pair (λ1,λ2)')
    axes[0].set_ylabel('Activation Rate')
    axes[0].set_title('Trigger Activation Rate')
    
    axes[1].bar(lamb_str, deviation_vals, color='salmon')
    axes[1].set_xlabel('Lambda pair (λ1,λ2)')
    axes[1].set_ylabel('Latent Deviation')
    axes[1].set_title('Latent Deviation (Cosine Loss)')
    
    plt.tight_layout()
    plt.savefig(".research/iteration1/images/dual_objective_loss_pair1.pdf", bbox_inches="tight")
    plt.close(fig)
    
    return results

def run_architecture_ablation(backdoor_model, camouflage_variants, data_loader, device, epochs=5):
    ablation_results = {}
    for variant_name, model_camouflage in camouflage_variants.items():
        model_camouflage.to(device)
        print("Evaluating camouflage architecture: {}".format(variant_name))
        activation_rates = []
        latent_deviations = []
        optimizer = optim.Adam(list(backdoor_model.parameters()) + list(model_camouflage.parameters()), lr=1e-3)
        for epoch in range(epochs):
            for images, captions in data_loader:
                images = images.to(device)
                benign_latent = images.view(images.size(0), -1)[:, :512]
                guidance = torch.randn(images.size(0), 512).to(device)
                
                backdoor_out = backdoor_model(benign_latent, guidance)
                poisoned_latent = model_camouflage(benign_latent)
    
                loss_activation = backdoor_activation_loss(backdoor_out, guidance)
                loss_evasion = defense_evasion_loss(poisoned_latent, benign_latent)
                loss = loss_activation + loss_evasion
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                activation_rates.append(1 if loss_activation.item() < 0.5 else 0)
                latent_deviations.append(loss_evasion.item())
        avg_activation_rate = np.mean(activation_rates)
        avg_deviation = np.mean(latent_deviations)
        ablation_results[variant_name] = {"Activation Rate": avg_activation_rate, "Latent Deviation": avg_deviation}
        print("Architecture {}: Activation Rate = {:.3f}, Latent Deviation = {:.3f}".format(variant_name, avg_activation_rate, avg_deviation))
    
    variants = list(ablation_results.keys())
    activations = [ablation_results[v]["Activation Rate"] for v in variants]
    deviations = [ablation_results[v]["Latent Deviation"] for v in variants]
    
    fig_act, ax_act = plt.subplots(figsize=(6,5))
    ax_act.bar(variants, activations, color='lightgreen')
    ax_act.set_xlabel("Camouflage Variant")
    ax_act.set_ylabel("Activation Rate")
    ax_act.set_title("Camouflage Architecture: Trigger Activation Rate")
    plt.tight_layout()
    plt.savefig(".research/iteration1/images/architecture_ablation_activation_pair1.pdf", bbox_inches="tight")
    plt.close(fig_act)
    
    fig_dev, ax_dev = plt.subplots(figsize=(6,5))
    ax_dev.bar(variants, deviations, color='orange')
    ax_dev.set_xlabel("Camouflage Variant")
    ax_dev.set_ylabel("Latent Deviation")
    ax_dev.set_title("Camouflage Architecture: Latent Deviation")
    plt.tight_layout()
    plt.savefig(".research/iteration1/images/architecture_ablation_deviation_pair1.pdf", bbox_inches="tight")
    plt.close(fig_dev)
    
    return ablation_results

def run_defense_in_the_loop_experiment(backdoor_model, camouflage_model, defense_module, data_loader, device, with_defense=True, epochs=5):
    activation_rates = []
    latent_deviations = []
    optimizer = optim.Adam(list(backdoor_model.parameters()) + list(camouflage_model.parameters()), lr=1e-3)
    
    for epoch in range(epochs):
        for images, captions in data_loader:
            images = images.to(device)
            benign_latent = images.view(images.size(0), -1)[:, :512]
            guidance = torch.randn(images.size(0), 512).to(device)
            backdoor_out = backdoor_model(benign_latent, guidance)
            poisoned_latent = camouflage_model(benign_latent)
            
            loss_activation = backdoor_activation_loss(backdoor_out, guidance)
            if with_defense:
                recovered_latent = defense_module(poisoned_latent)
                loss_defense = defense_evasion_loss(recovered_latent, benign_latent)
            else:
                loss_defense = torch.tensor(0.0).to(device)
                
            total_loss = loss_activation + loss_defense
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            activation_rates.append(1 if loss_activation.item() < 0.5 else 0)
            latent_deviations.append(loss_defense.item())
    
    avg_activation_rate = np.mean(activation_rates)
    avg_deviation = np.mean(latent_deviations)
    condition = "with defense" if with_defense else "without defense"
    print("Condition {}: Activation Rate = {:.3f}, Latent Deviation = {:.3f}".format(condition, avg_activation_rate, avg_deviation))
    return {"Activation Rate": avg_activation_rate, "Latent Deviation": avg_deviation}

def run_defense_experiment(backdoor_model, camouflage_model, defense_module, data_loader, device, epochs=5):
    metrics_with_defense = run_defense_in_the_loop_experiment(backdoor_model, camouflage_model, defense_module, data_loader, device, with_defense=True, epochs=epochs)
    metrics_without_defense = run_defense_in_the_loop_experiment(backdoor_model, camouflage_model, defense_module, data_loader, device, with_defense=False, epochs=epochs)
    
    conditions = ["with defense", "without defense"]
    activations = [metrics_with_defense["Activation Rate"], metrics_without_defense["Activation Rate"]]
    deviations = [metrics_with_defense["Latent Deviation"], metrics_without_defense["Latent Deviation"]]
    
    fig_act, ax_act = plt.subplots(figsize=(6,5))
    ax_act.bar(conditions, activations, color=['blue', 'gray'])
    ax_act.set_xlabel("Condition")
    ax_act.set_ylabel("Activation Rate")
    ax_act.set_title("Defense-In-The-Loop: Trigger Activation Rate")
    plt.tight_layout()
    plt.savefig(".research/iteration1/images/defense_activation_pair1.pdf", bbox_inches="tight")
    plt.close(fig_act)
    
    fig_dev, ax_dev = plt.subplots(figsize=(6,5))
    ax_dev.bar(conditions, deviations, color=['purple', 'brown'])
    ax_dev.set_xlabel("Condition")
    ax_dev.set_ylabel("Latent Deviation")
    ax_dev.set_title("Defense-In-The-Loop: Latent Deviation")
    plt.tight_layout()
    plt.savefig(".research/iteration1/images/defense_deviation_pair1.pdf", bbox_inches="tight")
    plt.close(fig_dev)
    
    return metrics_with_defense, metrics_without_defense
