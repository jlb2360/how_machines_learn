import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def grad_desc(func, guess: list, X: torch.Tensor, y: torch.Tensor, l_rate: float, n_epochs: int=100, loss_fn=None):
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    params = [p.clone().detach().requires_grad_(True) for p in guess]

    loss_history = []
    device = X.device

    for epoch in range(n_epochs):

        predictions = func(X, params)

        loss = loss_fn(predictions, y)
        loss_history.append(loss.item())

        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        loss.backward()

        with torch.no_grad():
            for i, p in enumerate(params):
                if p.grad is not None:
                    p -= l_rate * p.grad
                

        if (epoch + 1) % (n_epochs // 10 if n_epochs >= 10 else 1) == 0 or epoch == n_epochs - 1:
            param_values_str = ", ".join([f"{p.item():.4f}" if p.numel() == 1 else f"Tensor(shape={p.shape})" for p in params])
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}, Params: [{param_values_str}]")

    return params, loss_history
    

def linear_model_predictor(X, params):
    return params[0]*X + params[0]

def quadratic_model_predictor(X, params):
    return params[0] * (X**2) + params[1]*X + params[2]


print("\n--- Training Linear Model: y = w*x + b ---")
# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate sample data for linear model
X_linear_data = torch.arange(-5, 5, 0.1, device=device, dtype=torch.float32).view(-1, 1)
true_w_linear, true_b_linear = 2.0, 1.0
y_linear_measured = true_w_linear * X_linear_data + true_b_linear + torch.randn(X_linear_data.size(), device=device) * 0.5

# Initial parameters for linear model (MUST have requires_grad=True)
initial_w_lin = torch.randn(1, requires_grad=True, device=device, dtype=torch.float32)
initial_b_lin = torch.randn(1, requires_grad=True, device=device, dtype=torch.float32)
initial_params_for_linear_model = [initial_w_lin, initial_b_lin]

print(f"Initial params (linear): w={initial_params_for_linear_model[0].item():.4f}, b={initial_params_for_linear_model[1].item():.4f}")

# Hyperparameters for linear model training
lr_lin = 0.01
epochs_lin = 500

# Perform gradient descent
optimized_linear_params, linear_loss_history = grad_desc(
    func=linear_model_predictor,
    guess=initial_params_for_linear_model,
    X=X_linear_data,
    y=y_linear_measured,
    l_rate=lr_lin,
    n_epochs=epochs_lin
)
print(f"Optimized params (linear): w={optimized_linear_params[0].item():.4f}, b={optimized_linear_params[1].item():.4f}")


# --- Test with Quadratic Model ---
print("\n--- Training Quadratic Model: y = a*x^2 + b*x + c ---")
# Generate sample data for quadratic model
X_quad_data = torch.arange(-3, 3, 0.1, device=device, dtype=torch.float32).view(-1, 1)
true_a_quad, true_b_quad, true_c_quad = 1.5, -2.0, 0.5
y_quad_measured = true_a_quad * (X_quad_data**2) + true_b_quad * X_quad_data + true_c_quad + torch.randn(X_quad_data.size(), device=device) * 0.3

# Initial parameters for quadratic model (MUST have requires_grad=True)
initial_a_quad = torch.randn(1, requires_grad=True, device=device, dtype=torch.float32)
initial_b_quad = torch.randn(1, requires_grad=True, device=device, dtype=torch.float32)
initial_c_quad = torch.randn(1, requires_grad=True, device=device, dtype=torch.float32)
initial_params_for_quadratic_model = [initial_a_quad, initial_b_quad, initial_c_quad]

print(f"Initial params (quadratic): a={initial_params_for_quadratic_model[0].item():.4f}, b={initial_params_for_quadratic_model[1].item():.4f}, c={initial_params_for_quadratic_model[2].item():.4f}")

# Hyperparameters for quadratic model training
lr_quad = 0.005 # Quadratic models can sometimes be more sensitive to learning rate
epochs_quad = 500

# Perform gradient descent
optimized_quadratic_params, quadratic_loss_history = grad_desc(
    func=quadratic_model_predictor,
    guess=initial_params_for_quadratic_model,
    X=X_quad_data,
    y=y_quad_measured,
    l_rate=lr_quad,
    n_epochs=epochs_quad
)
print(f"Optimized params (quadratic): a={optimized_quadratic_params[0].item():.4f}, b={optimized_quadratic_params[1].item():.4f}, c={optimized_quadratic_params[2].item():.4f}")


# --- Plotting the results ---
plt.figure(figsize=(16, 10)) # Adjusted figure size for better layout

# Linear Model Plot
plt.subplot(2, 2, 1)
plt.title('Linear Model Fit (y = wx + b)')
plt.scatter(X_linear_data.cpu().numpy(), y_linear_measured.cpu().numpy(), label='Measured Data', s=15, alpha=0.6)
final_w_plot, final_b_plot = [p.detach().cpu().numpy() for p in optimized_linear_params]
y_pred_linear_final_plot = final_w_plot * X_linear_data.cpu().numpy() + final_b_plot
plt.plot(X_linear_data.cpu().numpy(), y_pred_linear_final_plot, color='red', linewidth=2, label=f'Fitted Line: w={final_w_plot[0]:.2f}, b={final_b_plot[0]:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 2, 2)
plt.title('Linear Model Loss Curve')
plt.plot(linear_loss_history, color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)

# Quadratic Model Plot
plt.subplot(2, 2, 3)
plt.title('Quadratic Model Fit (y = ax² + bx + c)')
plt.scatter(X_quad_data.cpu().numpy(), y_quad_measured.cpu().numpy(), label='Measured Data', s=15, alpha=0.6)
final_a_plot, final_b_plot, final_c_plot = [p.detach().cpu().numpy() for p in optimized_quadratic_params]
y_pred_quad_final_plot = final_a_plot * (X_quad_data.cpu().numpy()**2) + final_b_plot * X_quad_data.cpu().numpy() + final_c_plot
plt.plot(X_quad_data.cpu().numpy(), y_pred_quad_final_plot, color='green', linewidth=2, label=f'Fitted Curve: a={final_a_plot[0]:.2f}, b={final_b_plot[0]:.2f}, c={final_c_plot[0]:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 2, 4)
plt.title('Quadratic Model Loss Curve')
plt.plot(quadratic_loss_history, color='purple')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout() # Adjusts subplot params for a tight layout.
plt.show()

print(f"\nTrue Linear Params: w={true_w_linear:.4f}, b={true_b_linear:.4f}")
print(f"True Quadratic Params: a={true_a_quad:.4f}, b={true_b_quad:.4f}, c={true_c_quad:.4f}")

