import numpy as np

# FUNCIONES DE ACTIVACION

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	return x * (1 - x) # derivada de la sigmoide


# DATOS DE ENTRENAMIENTO (XOR)

X = np.array([[0,0],[0,1],[1,0],[1,1]])

y = np.array([[0],[1],[1],[0]])


# Inicializacion de pesos

np.random.seed(42)  # reproducible
W1 = np.random.randn(2, 3)   # 2 entradas -> 3 neuronas ocultas
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1)   # 3 neuronas ocultas -> 1 salida
b2 = np.zeros((1, 1))

# Entrenamiento

lr = 0.1   # tasa de aprendizaje
epochs = 10000

for epoch in range(epochs):

    # FORWARD
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Pérdida (error cuadratico medio)
    loss = np.mean((y - a2) ** 2)

    # BACKPROP
    error = y - a2
    d_a2 = error * sigmoid_deriv(a2)

    error_hidden = d_a2.dot(W2.T)
    d_a1 = error_hidden * sigmoid_deriv(a1)

    # Actualizar pesos y sesgos
    W2 += a1.T.dot(d_a2) * lr
    b2 += np.sum(d_a2, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_a1) * lr
    b1 += np.sum(d_a1, axis=0, keepdims=True) * lr

    # Mostrar cada 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Probar la red
print("\nPredicciones finales:")
print(a2.round(2))
