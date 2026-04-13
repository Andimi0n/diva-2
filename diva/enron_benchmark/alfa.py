import numpy as np
from sklearn.svm import LinearSVC
import torch
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

def calculate_hinge_loss(model, X, y):
    """
    Calculates the individual hinge loss for each sample: max(0, 1 - y * decision_function(x))
    """
    decision_values = model.decision_function(X)
    # Ensure y and decision_values are 1D arrays to prevent broadcasting bugs
    losses = np.maximum(0, 1 - np.ravel(y) * np.ravel(decision_values))
    return losses

def alfa_poison(X, y, logger, epsilon=0.03, max_iter=10):
    """
    Implementation of the ALFA attack variant.
    Selects poisoned points from the dataset to maximize the loss difference 
    between the clean model and the poisoned model, flipping labels in-place.
    
    Expects y to be in {-1, 1}.
    """
    logger.info(f"   -> Starting in-place ALFA (epsilon={epsilon*100}%)...")
    
    # FIX WARNING: Force y to be a strict 1D array 
    y = np.ravel(y)
    
    n_samples = X.shape[0]
    n_poison = int(epsilon * n_samples)
    
    if n_poison == 0:
        return X, np.copy(y)
        
    # 1. Define the pool of potential poisoned points: The entire dataset flipped
    X_pool = X
    y_pool = -y 
    
    # 2. Train the original clean model 
    # FIX CRASH: Use LinearSVC instead of SVC(kernel='linear')
    # dual=False is highly recommended when n_samples > n_features (50000 > 100)
    logger.info("      Training original clean model...")
    clean_model = LinearSVC(dual=False, random_state=42) 
    clean_model.fit(X, y)

    clean_acc = clean_model.score(X, y)
    logger.info(f"      [METRIC] Clean model accuracy (on true labels): {clean_acc:.4f}")
    
    # Calculate loss of the flipped pool under the clean model
    loss_clean = calculate_hinge_loss(clean_model, X_pool, y_pool)
    
    # 3. Initialize by greedily selecting points with the highest loss under the clean model
    top_indices = np.argsort(loss_clean)[-n_poison:]
    current_poison_indices = set(top_indices)
    
    # 4. Iterative Optimization to find the best subset
    final_poisoned_model = None
    for i in range(max_iter):
        
        # Create the current poisoned label array by flipping the selected indices
        y_poisoned = np.copy(y)
        idx_list = list(current_poison_indices)
        y_poisoned[idx_list] = -y_poisoned[idx_list]
        
        # Train the poisoned model 
        # FIX CRASH: LinearSVC here too
        poisoned_model = LinearSVC(dual=False, random_state=42)
        poisoned_model.fit(X, y_poisoned)
        
        # Calculate loss of the entire flipped pool under the new poisoned model
        loss_poisoned = calculate_hinge_loss(poisoned_model, X_pool, y_pool)
        
        # Objective: Maximize Loss_clean - Loss_poisoned
        alfa_scores = loss_clean - loss_poisoned
        
        # Select the new best indices based on the updated scores
        new_top_indices = np.argsort(alfa_scores)[-n_poison:]
        new_poison_indices = set(new_top_indices)
        
        # Check for convergence
        if new_poison_indices == current_poison_indices:
            logger.info(f"   -> ALFA successfully converged at iteration {i+1}!")
            final_poisoned_model = poisoned_model
            break
            
        # Update for next iteration
        current_poison_indices = new_poison_indices
        
    logger.info("   -> ALFA optimization finished.")

    # Create the final poisoned array
    y_poisoned_final = np.copy(y)
    final_idx_list = list(current_poison_indices)
    y_poisoned_final[final_idx_list] = -y_poisoned_final[final_idx_list]

    if final_poisoned_model is not None:
        # Evaluate how much the poisoned model degrades on the TRUE underlying labels
        poisoned_acc = final_poisoned_model.score(X, y)
        logger.info(f"      [METRIC] Poisoned model accuracy (on true labels): {poisoned_acc:.4f}")
        logger.info(f"      [METRIC] Total accuracy drop caused by ALFA: {(clean_acc - poisoned_acc)*100:.2f}%")
    
    return X, y_poisoned_final

def alfa_pytorch(X_train, y_train, rate, C=1.0, max_iter=10, inner_iter=50):
    """
    GPU-Accelerated PyTorch implementation of the ALFA attack.
    Replaces CPU-bound LP/QP solvers with Alternating Gradient Descent.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   -> Running PyTorch ALFA on: {device}")

    if sp.issparse(X_train):
        X_train = X_train.toarray()
        
    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)

    N, D = X.shape
    budget = int(rate * N)

    if budget == 0:
        return y_train.copy()

    # Initialize Model Parameters (w, b) and Attack Mask (q)
    w = torch.zeros((D, 1), requires_grad=True, device=device)
    b = torch.zeros(1, requires_grad=True, device=device)
    
    # q: 0 = original label, 1 = flipped label
    q = torch.zeros((N, 1), device=device)

    optimizer = optim.Adam([w, b], lr=0.05)

    pbar = tqdm(range(max_iter), ncols=100, desc="ALFA Outer Loop")

    for _ in pbar:
        # Calculate the effectively poisoned labels: 
        # If q=0, y_pois = y. If q=1, y_pois = -y.
        y_pois = y * (1 - 2 * q)
        
        for _ in range(inner_iter):
            optimizer.zero_grad()
            
            # Linear SVM Forward Pass & Hinge Loss
            margin = y_pois * (X.mm(w) + b)
            hinge_loss = torch.clamp(1 - margin, min=0).mean()
            
            # L2 Regularization
            reg_loss = 0.5 * torch.sum(w ** 2) / C
            
            # Total Loss
            loss = hinge_loss + reg_loss
            loss.backward()
            optimizer.step()

        # ---------------------------------------------------------
        # STEP 2: OUTER OPTIMIZATION (Update Attack) - Replaces solveLP
        # ---------------------------------------------------------
        with torch.no_grad():
            # Calculate the loss if the label was untouched
            margin_orig = y * (X.mm(w) + b)
            loss_orig = torch.clamp(1 - margin_orig, min=0)
            
            # Calculate the loss if the label was flipped
            margin_flipped = -y * (X.mm(w) + b)
            loss_flipped = torch.clamp(1 - margin_flipped, min=0)
            
            # The "benefit" of flipping is exactly the 'psi' from the original paper
            flip_benefit = loss_flipped - loss_orig
            
            # Reset q and enforce budget by taking the top K most beneficial flips
            q.zero_()
            _, top_indices = torch.topk(flip_benefit.squeeze(), budget)
            q[top_indices] = 1.0
            
        pbar.set_postfix({'Max Flip Benefit': f"{flip_benefit[top_indices[0]].item():.4f}"})

    with torch.no_grad():
        flip_indices = top_indices.cpu().numpy()
        y_flip = y_train.copy()
        y_flip[flip_indices] = -y_flip[flip_indices]

    return y_flip