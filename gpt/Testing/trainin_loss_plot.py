import matplotlib.pyplot as plt

# Data provided
steps = [i for i in range(1, 89)]  # Step numbers
training_loss = [
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0001,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0195, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0045,
    0.0000, 0.0001, 0.0034, 0.4200, 0.0006, 0.0009, 0.0026, 0.0038, 0.3653, 0.0170,
    0.1009, 0.0861, 0.6789, 0.0844, 0.1956, 0.2051, 0.4150, 1.3170, 0.4912, 1.1711,
    1.7412, 2.9793, 3.7454, 2.8467, 3.2202, 2.7145, 4.1948, 3.4385
]  # Training loss values

# Reverse the order of data
steps
training_loss.reverse()

# Plotting the training loss
plt.figure(figsize=(10, 6))
plt.plot(steps, training_loss, marker='o', color='b', linestyle='-')
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.grid(True)
plt.xticks(steps[::5])  # Show every 5th step on x-axis
plt.tight_layout()
plt.show()
