import numpy as np
from sklearn.svm import SVC

def calculate_hinge_loss(model, X, y):
    """
    Calculates the individual hinge loss for each sample: max(0, 1 - y * decision_function(x))
    """
    decision_values = model.decision_function(X)
    # Hinge loss per sample
    losses = np.maximum(0, 1 - y * decision_values)
    return losses

def alfa_poison(X, y, epsilon=0.03, max_iter=10):
    """
    Implementation of the ALFA attack variant.
    Selects poisoned points from the dataset to maximize the loss difference 
    between the clean model and the poisoned model, flipping labels in-place.
    
    Expects y to be in {-1, 1}.
    """
    print(f"   -> Starting in-place ALFA (epsilon={epsilon*100}%)...")
    
    n_samples = X.shape[0]
    n_poison = int(epsilon * n_samples)
    
    if n_poison == 0:
        return X, np.copy(y)
        
    # 1. Define the pool of potential poisoned points: The entire dataset flipped
    X_pool = X
    y_pool = -y # Flip all labels to evaluate the potential cost/benefit
    
    # 2. Train the original clean model (theta*)
    print("      Training original clean model...")
    clean_model = SVC(kernel='linear') 
    clean_model.fit(X, y)

    clean_acc = clean_model.score(X, y)
    print(f"      [METRIC] Clean model accuracy (on true labels): {clean_acc:.4f}")
    
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
        y_poisoned[list(current_poison_indices)] = -y_poisoned[list(current_poison_indices)]
        
        # Train the poisoned model (theta_hat)
        poisoned_model = SVC(kernel='linear')
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
            print(f"   -> ALFA successfully converged at iteration {i+1}!")
            final_poisoned_model = poisoned_model
            break
            
        # Update for next iteration
        current_poison_indices = new_poison_indices
        
    print("   -> ALFA optimization finished.")

    # Create the final poisoned array
    y_poisoned_final = np.copy(y)
    y_poisoned_final[list(current_poison_indices)] = -y_poisoned_final[list(current_poison_indices)]

    if final_poisoned_model is not None:
        # Evaluate how much the poisoned model degrades on the TRUE underlying labels
        poisoned_acc = final_poisoned_model.score(X, y)
        print(f"      [METRIC] Poisoned model accuracy (on true labels): {poisoned_acc:.4f}")
        print(f"      [METRIC] Total accuracy drop caused by ALFA: {(clean_acc - poisoned_acc)*100:.2f}%")
    
    # Return exact same size X and y, with only labels flipped
    return X, y_poisoned_final