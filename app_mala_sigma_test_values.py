import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------
# Weight clipping function
# -------------------------------------------------
def clip_weights(model, max_weight_value):
    return 0
    # with torch.no_grad():
    #     for module in model.modules():
    #         if isinstance(module, nn.Linear):
    #             module.weight.data.clamp_(-max_weight_value, max_weight_value)

# -------------------------------------------------
# Model definition
# -------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(NeuralNetwork, self).__init__()
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.hidden_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.ln_layers.append(nn.LayerNorm(hidden_size))

        self.fc_out = nn.Linear(hidden_size, 1)

        # Initialization
        for layer in [self.fc1] + list(self.hidden_layers) + [self.fc_out]:
            nn.init.normal_(layer.weight, mean=0.0, std=STD_INIT)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
            x = self.ln_layers[i](x)
            x = torch.relu(x)
        x = torch.sigmoid(self.fc_out(x))
        return x

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def normalize_data(X_train, X_test, scale_factor=1.0):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = ((X_train - mean) / std) * scale_factor
    X_test_norm = ((X_test - mean) / std) * scale_factor
    return X_train_norm, X_test_norm

def create_dataloaders(X_train, Y_train, X_test, Y_test, batch_size=32):
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_test).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# -------------------------------------------------
# Testing
# -------------------------------------------------
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total, None

# -------------------------------------------------
# Posterior Prediction
# -------------------------------------------------
def predict_with_samples(model, samples, X_test, Y_test):
    model.eval()
    predictions = []
    device = next(model.parameters()).device
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    Y_test_tensor = torch.FloatTensor(Y_test).unsqueeze(1).to(device)

    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_weights[name] = module.weight.detach().clone()

    num_samples = len(next(iter(samples.values())))
    for i in range(num_samples):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.copy_(samples[f"{name}.weight"][i].to(device))
        with torch.no_grad():
            outputs = model(X_test_tensor)
            predictions.append(outputs.detach())

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data.copy_(original_weights[name])

    predictions = torch.stack(predictions).mean(dim=0)
    predicted = (predictions > 0.5).float()
    accuracy = (predicted == Y_test_tensor).float().mean().item()
    return accuracy, predictions.cpu()

# -------------------------------------------------
# Binary entropy
# -------------------------------------------------
def binary_entropy(p):
    eps = 1e-8
    return -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)

# -------------------------------------------------
# Predict with uncertainty (Mutual Information)
# -------------------------------------------------
def predict_with_uncertainty(model, samples, data_loader):
    model.eval()
    device = next(model.parameters()).device

    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_weights[name] = module.weight.detach().clone()

    num_samples = len(next(iter(samples.values())))
    all_probs = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            batch_predictions = []

            for i in range(num_samples):
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        module.weight.data.copy_(samples[f"{name}.weight"][i].to(device))

                outputs = model(inputs)
                batch_predictions.append(outputs.squeeze(1))

            batch_predictions = torch.stack(batch_predictions)
            all_probs.append(batch_predictions)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data.copy_(original_weights[name])

    all_probs = torch.cat(all_probs, dim=1)

    mean_prob = all_probs.mean(dim=0)
    total_entropy = binary_entropy(mean_prob)
    aleatoric = binary_entropy(all_probs).mean(dim=0)
    epistemic = total_entropy - aleatoric

    return all_probs.cpu(), mean_prob.cpu(), aleatoric.cpu(), epistemic.cpu()

# -------------------------------------------------
# Helper: log-prob of Langevin proposal
# -------------------------------------------------
def log_q(x_new, x_old, grad, lr):
    diff = x_new - x_old - lr * grad
    return -0.5 * torch.sum(diff**2) / (2 * lr)


# -------------------------------------------------
# Training with deterministic + MALA with ramped temperature
# -------------------------------------------------
def train_model(model, train_loader, test_loader,
                deterministic_epochs=20,
                mala_epochs=30,
                burnin_epochs=10,
                thinning=5,
                lr=0.0005,
                prior_std=1.0,
                ramp_batches=50,
                max_weight_value=1.0):

    model = model.to(device)
    criterion = nn.BCELoss(reduction="mean")
    prior_precision = 1.0 / (prior_std ** 2)
    N = len(train_loader.dataset)

    linear_weight_params = []
    other_params = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_weight_params.append(module.weight)
            if module.bias is not None:
                other_params.append(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                other_params.append(module.weight)
                other_params.append(module.bias)

    adam_optimizer = optim.Adam(linear_weight_params + other_params, lr=lr)

    samples = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            samples[f"{name}.weight"] = []

    print("=== Deterministic training phase ===")
    # gradient descent of the loss 
    for epoch in range(deterministic_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            adam_optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) * (N / inputs.size(0))
            log_prior = sum(-0.5 * prior_precision * torch.sum(w**2) for w in linear_weight_params)
            total_loss = loss - log_prior
            total_loss.backward()
            adam_optimizer.step()
            clip_weights(model, max_weight_value)
        acc, _ = test_model(model, test_loader)
        print(f"Epoch {epoch+1}/{deterministic_epochs} | Test Acc: {acc:.4f}")



    print("=== MALA sampling phase ===")
    for p in other_params:
        p.requires_grad = False

    step_counter = 0
    accepted_steps = 0
    total_steps = 0
    burnin_steps = burnin_epochs * len(train_loader)
    batch_idx = 0

    for epoch in range(mala_epochs):
        model.train()
        epoch_accepted_steps = 0
        epoch_total_steps = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_idx += 1
            T = min(batch_idx / ramp_batches, 1.0)

            current_weights = [w.clone() for w in linear_weight_params]

            model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) * (N / inputs.size(0))
            log_prior = sum(-0.5 * prior_precision * torch.sum(w**2) for w in linear_weight_params)
            log_post = -loss + log_prior
            (-loss + log_prior).backward()
            current_grads = [w.grad.clone() for w in linear_weight_params]

            proposed_weights = []
            for w, g in zip(linear_weight_params, current_grads):
                noise = torch.randn_like(w) * torch.sqrt(torch.tensor(2.0 * lr, device=device))
                proposed_weights.append(w + lr * g + noise)

            for w, w_prop in zip(linear_weight_params, proposed_weights):
                w.data.copy_(w_prop)

            model.zero_grad()
            outputs = model(inputs)
            loss_prop = criterion(outputs, labels) * (N / inputs.size(0))
            log_prior_prop = sum(-0.5 * prior_precision * torch.sum(w**2) for w in linear_weight_params)
            log_post_prop = -loss_prop + log_prior_prop
            (-loss_prop + log_prior_prop).backward()
            proposed_grads = [w.grad.clone() for w in linear_weight_params]

            log_q_forward = sum(log_q(w_prop, w_curr, g, lr)
                                for w_prop, w_curr, g in zip(proposed_weights, current_weights, current_grads))
            log_q_backward = sum(log_q(w_curr, w_prop, g_prop, lr)
                                 for w_curr, w_prop, g_prop in zip(current_weights, proposed_weights, proposed_grads))

            log_alpha = T * (log_post_prop - log_post) + (log_q_backward - log_q_forward)

            if torch.rand(1, device=device) < torch.exp(log_alpha):
                accepted_steps += 1
                epoch_accepted_steps += 1
                clip_weights(model, max_weight_value)
            else:
                # if rejected, revert to current weights
                for w, w_curr in zip(linear_weight_params, current_weights):
                    w.data.copy_(w_curr)
                clip_weights(model, max_weight_value)

            total_steps += 1
            epoch_total_steps += 1
            step_counter += 1

            if step_counter > burnin_steps and (step_counter - burnin_steps) % thinning == 0:
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        samples[f"{name}.weight"].append(module.weight.detach().cpu().clone())

        acc, _ = test_model(model, test_loader)
        epoch_acceptance_rate = epoch_accepted_steps / epoch_total_steps
        print(f"MALA Epoch {epoch+1}/{mala_epochs} | Test Acc: {acc:.4f} | Epoch Acceptance Rate: {epoch_acceptance_rate:.4f}")

    acceptance_rate  = accepted_steps / total_steps
    print("Acceptance Rate:", acceptance_rate)
    print("Total posterior samples collected:", len(next(iter(samples.values()))))
    return samples, acceptance_rate

# -------------------------------------------------
# Training script
# -------------------------------------------------
# X_test =  np.loadtxt("datasetmixed/X_test10")
# X_train = np.loadtxt("datasetmixed/X_train10")
# Y_test =  np.loadtxt("datasetmixed/Y_test10")
# Y_train = np.loadtxt("datasetmixed/Y_train10")

X_test =  np.loadtxt("dataset/X_test")
X_train = np.loadtxt("dataset/X_train")
Y_test =  np.loadtxt("dataset/Y_test").astype(float)
Y_train = np.loadtxt("dataset/Y_train").astype(float)

# -------------------------------------------------
# Load OOD data
# -------------------------------------------------
X_ood = np.loadtxt("dataset/X_ood")

# Create OOD DataLoader (with dummy labels if needed, but not used in predict_with_uncertainty)
ood_dataset = TensorDataset(torch.FloatTensor(X_ood), torch.zeros(len(X_ood)))  # Dummy labels
ood_loader = DataLoader(ood_dataset, batch_size=256, shuffle=False)

# Y_test = (Y_test[:, 0] == 1).astype(float)
# Y_train = (Y_train[:, 0] == 1).astype(float)

prior_std = 0.25
STD_INIT = 0.25
ramp_batches = 1000
max_weight_value = 1.1  # Set the  max weight value here

X_train_norm, X_test_norm = normalize_data(X_train, X_test, scale_factor=2.0)
train_loader, test_loader = create_dataloaders(X_train_norm, Y_train, X_test_norm, Y_test, batch_size=64)

# sigma_val = np.logspace(0.01, 0.001).tolist()
sigma_val = [0.04]
print(f"Testing sigma values: {sigma_val}")

accuracy_val = []
epistemic_val = []
acceptance_rate_val = []

for sigma in sigma_val:
    lr = sigma**2/2
    print(f"\n=== Training with sigma={sigma:.4f} (lr={lr:.6f}) ===")

    model = NeuralNetwork(input_size=32, hidden_size=128, num_layers=3)

    samples, ar = train_model(model, train_loader, test_loader,
                        deterministic_epochs=0,
                        mala_epochs=80,
                        burnin_epochs=30,
                        thinning=5,
                        lr=lr,
                        prior_std=prior_std,
                        ramp_batches=ramp_batches,
                        max_weight_value=max_weight_value)
    
    acceptance_rate_val.append(ar)

    posterior_accuracy, posterior_predictions = predict_with_samples(model, samples, X_test_norm, Y_test)
    print(f"Posterior Accuracy: {posterior_accuracy:.4f}")
    accuracy_val.append(posterior_accuracy)

    all_probs_id, mean_prob_id, aleatoric_id, epistemic_id = predict_with_uncertainty(model, samples, test_loader)
    all_probs_ood, mean_prob_ood, aleatoric_ood, epistemic_ood = predict_with_uncertainty(model, samples, ood_loader)
    
    roc_auc_score = metrics.roc_auc_score([0]*len(epistemic_id.flatten()) + [1]*len(epistemic_ood.flatten()), np.concatenate((epistemic_id.flatten(), epistemic_ood.flatten())))



np.savez("sigma_test_values/sigma_test.npz",
         sigma_val=sigma_val,
         accuracy_val=accuracy_val,
         acceptance_rate_val=acceptance_rate_val,
         roc_auc_score=roc_auc_score,
)




