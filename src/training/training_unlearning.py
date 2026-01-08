import torch
import torch.nn as nn
from itertools import cycle
from src.unlearning.compute_parameters import calculate_parameters
from src.utils.utils import choose_device
from src.utils.calculate_accuracy import evaluate
from src.utils.utils import save_accuracy_history, save_model


def clip_gradients_l2(gradients, c1):
    total_norm = torch.sqrt(sum(torch.sum(g**2) for g in gradients if g is not None))
    clip_coef = min(1.0, c1 / (total_norm + 1e-6))
    return [g * clip_coef if g is not None else None for g in gradients]


def clip_parameters_l2(model, c0):
    # Flatten all params into one norm
    total_norm = torch.sqrt(
        sum(torch.sum(p.data**2) for p in model.parameters() if p is not None)
    )

    clip_coef = min(1.0, c0 / (total_norm + 1e-6))
    for p in model.parameters():
        if p is not None:
            p.data.mul_(clip_coef)


@torch.no_grad()
def apply_noise_step(params, sigma):
    """Apply Gaussian noise to parameters."""
    for param in params:
        param.add_(torch.randn_like(param) * sigma)


@torch.no_grad()
def apply_finetune_step(params, grads, lr, weight_decay, clip_norm):
    """Apply a single clipped gradient descent step with weight decay."""
    clipped_grads = clip_gradients_l2(grads, clip_norm)

    for param, grad in zip(params, clipped_grads):
        param.mul_(1 - lr * weight_decay)
        param.add_(grad, alpha=-lr)


def finetune_model(
    model,
    infinite_loader,
    criterion,
    device,
    max_steps,
    step,
    test_acc_per_iteration,
    log_every,
    test_loader,
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
):
    """
    Perform standard SGD fine-tuning on the given model.

    Args:
        model: torch.nn.Module
        infinite_loader: data loader that yields (images, labels)
        criterion: loss function
        device: "cuda", "mps" or "cpu"
        max_steps: number of SGD steps
        lr: learning rate
    """

    print(
        f"Finetuning model with lr={lr}, momentum={momentum}, weight_decay={weight_decay}"
    )
    # Unfreeze all params
    for p in model.parameters():
        p.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    while step < max_steps:
        images, labels = next(infinite_loader)
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

        if step % log_every == 0:
            acc = evaluate(model, test_loader, device=device)
            test_acc_per_iteration.append((step, acc))

            print(f"[Step {step}] Loss: {loss.item():.4f} - Test Acc: {acc:.2f}%")

    return model


def train_base_unlearning_model(
    trained_full_state_path,
    retain_loader,
    test_loader,
    config_path,
    modelname,
    log_every=500,
    max_T=5000,
    save_model_weights=True,
    model_class=None,
):

    config = calculate_parameters(config_path)

    # Define the model again
    device = choose_device()
    model = model_class(num_classes=10).to(device)

    # load base model
    trained_full_state = torch.load(trained_full_state_path, map_location=device)
    model.load_state_dict(trained_full_state)

    # DataLoader over retain dataset
    infinite_loader = cycle(retain_loader)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Clip model
    # clip_parameters_l2(model, c0=config["c_0"])

    test_acc_per_iteration = []

    # Training loop: T iterations
    model.train()
    for step in range(config["T"]):
        images, labels = next(infinite_loader)
        images, labels = images.to(device), labels.to(device)

        model.zero_grad()

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Calculte params and grads
        params = [p for p in model.parameters() if p.requires_grad]
        grads = [p.grad for p in params]

        with torch.no_grad():
            # Every M-th step is noise step
            if (step + 1) % config["M"] == 0:
                apply_noise_step(params, config["sigma_new"])

            # All other steps are finetuning steps
            else:
                apply_finetune_step(
                    params=params,
                    grads=grads,
                    lr=config["gamma_t"],
                    weight_decay=config["lambda_t"],
                    clip_norm=config["c_1"],
                )

        # Logging
        if step < config["T"] or step % log_every == 0:
            acc = evaluate(model, test_loader, device=device)
            test_acc_per_iteration.append((step, acc))

            T = config["T"]
            print(f"[Step {step}/{T}] Loss: {loss.item():.4f} - Test Acc: {acc:.2f}%")

    finetune_model(
        model,
        infinite_loader,
        criterion,
        device,
        max_steps=max_T,
        step=step,
        test_acc_per_iteration=test_acc_per_iteration,
        log_every=log_every,
        test_loader=test_loader,
        lr=float(config["finetuning_lr"]),
        weight_decay=float(config["finetuning_weight_decay"]),
        momentum=float(config["finetuning_momentum"]),
    )

    accuracy_filepath = f"experiments_results/accuracy/{modelname}.pt"
    save_accuracy_history(test_acc_per_iteration, filepath=accuracy_filepath)

    if save_model_weights:
        save_model(
            model,
            filepath=f"experiments_results/models/{modelname}.pt",
        )

    return model, test_acc_per_iteration
