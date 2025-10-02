# Mi Primera Red Neuronal en Python

Este proyecto implementa una red neuronal desde cero usando únicamente `numpy`.  
El objetivo es mostrar, paso a paso, cómo funciona una red neuronal y cómo se relaciona con el cálculo multivariado.

---

## Introducción

Una red neuronal artificial es un modelo matemático inspirado en el cerebro humano.  
Se compone de capas de neuronas artificiales, que transforman datos de entrada en una salida.

Estructura típica:

1. **Capa de entrada** → recibe los datos (x₁, x₂, ...).  
2. **Capas ocultas** → procesan y aprenden representaciones intermedias.  
3. **Capa de salida** → da el resultado final (ejemplo: 0 o 1).  

---

## Relación con Cálculo Multivariado

Las redes neuronales están directamente relacionadas con conceptos de cálculo multivariado:

- La red es una **función multivariable**:  
  f(x; W, b)

- El entrenamiento consiste en **optimizar una función multivariable**:  
  minimizar la función de pérdida L(W, b).

- Conceptos aplicados:
  - **Derivadas parciales**: cómo cambia el error respecto a cada peso.  
  - **Gradiente**: vector que agrupa todas las derivadas parciales.  
  - **Regla de la cadena**: utilizada en backpropagation para derivar funciones compuestas.  
  - **Optimización**: gradiente descendente para encontrar mínimos locales.  

---

## Flujo de funcionamiento

### Diagrama conceptual

<<<<<<< HEAD
           ┌────────────┐
   X ----> │  W1, b1    │
           └─────┬──────┘
                 │
                 ▼
           ┌────────────┐
           │  Sigmoide  │
           └─────┬──────┘
                 │
                 ▼
           ┌────────────┐
           │   a1       │
           └─────┬──────┘
                 │
                 ▼
           ┌────────────┐
           │  W2, b2    │
           └─────┬──────┘
                 │
                 ▼
           ┌────────────┐
           │  Sigmoide  │
           └─────┬──────┘
                 │
                 ▼
           ┌────────────┐
           │   a2       │
           └─────┬──────┘
                 │
                 ▼
   y ---> Comparación ---> Pérdida L = (y - a2)²
                 │
                 ▼
     Gradientes (regla de la cadena)
                 │
                 ▼
  Actualización de parámetros (W, b)
                 │
                 ▼
             Repetir ciclo
=======
Entrada (X)
│
▼
Multiplicación por W1 y suma de b1
│
▼
Función de activación (sigmoide) → Capa oculta (a1)
│
▼
Multiplicación por W2 y suma de b2
│
▼
Función de activación (sigmoide) → Salida (a2)
│
▼
Comparación con y (valor esperado)
│
▼
Función de pérdida L = (y - a2)²
│
▼
Cálculo de gradiente (derivadas parciales, regla de la cadena)
│
▼
Actualización de parámetros (W, b) con gradiente descendente
│
▼
Repetir hasta minimizar el error
>>>>>>> parent of ce4a35f (fix(README.md): update README)


---

## Pasos matemáticos

1. **Propagación hacia adelante (Forward Pass)**  
   - Operación lineal:  
     \[
     z = W \cdot x + b
     \]
   - Activación no lineal (sigmoide):  
     \[
     a = \sigma(z) = \frac{1}{1 + e^{-z}}
     \]

2. **Función de pérdida**  
   Error cuadrático medio:  
   \[
   L = \frac{1}{n}\sum (y - \hat{y})^2
   \]

3. **Backpropagation**  
   Aplicamos la regla de la cadena:  
   \[
   \frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w_{ij}}
   \]

4. **Actualización de parámetros (Gradiente descendente)**  
   \[
   w := w - \eta \frac{\partial L}{\partial w}
   \]

---

## Código explicado (`src/red.py`)

### Funciones de activación
```python
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x): 
    return x * (1 - x)
