import math
import random
from smolgrad.nn import Sequential,Linear,Tanh
from smolgrad.loss import MSELoss
from smolgrad.optim import Adam
# =====================================================================
# 1. SETUP THE MODEL ARCHITECTURE USING YOUR SEQUENTIAL CLASS
#              A 1-4-4-1 Multi-Layer Perceptron (MLP).
# =====================================================================

model = Sequential(
    Linear(nin=1, nout=4),
    Tanh(),
    Linear(nin=4, nout=4),
    Tanh(),
    Linear(nin=4, nout=1)
)

print(f"Model initialized with {model.num_parameters()} parameters.")

# =====================================================================
# 2. GENERATE THE TRAINING DATA
# =====================================================================
num_samples = 50
x_train = []
y_train = []

for i in range(num_samples):
    # Scale linear space between -pi and pi
    x_val = -math.pi + (2 * math.pi * i / (num_samples - 1))
    x_train.append([x_val]) # Wrapped in a list to match nin=1
    y_train.append(math.sin(x_val))

# =====================================================================
# 3. INITIALIZE YOUR OPTIMIZER AND LOSS FUNCTION
# =====================================================================

optimizer = Adam(model.parameters(), lr=0.01)
criterion = MSELoss()

# =====================================================================
# 4. TRAINING LOOP
# =====================================================================
epochs = 500
print("\nStarting training...")
for epoch in range(epochs):
    #forward pass
    predictions = []
    for x_sample in x_train:
        pred = model(x_sample)
        predictions.append(pred)
    loss = criterion(predictions, y_train)
    loss_val = loss / num_samples    
    #backward pass
    optimizer.zero_grad()    
    loss_val.backward()      
    optimizer.step()         
    
   
    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {loss_val.data:.6f}")

# =====================================================================
# 5. TESTING INDIVIDUAL PREDICTIONS
# =====================================================================
print("\n--- Testing Model Predictions ---")
test_points = [-math.pi/2, 0.0, math.pi/2]

for tp in test_points:
    pred_val = model([tp]).data
    actual_val = math.sin(tp)
    print(f"sin({tp:6.3f}) -> Predicted: {pred_val:6.3f} | Actual: {actual_val:6.3f}")
