# Tensor++

Implementación en C++ de una librería básica de tensores de hasta 3 dimensiones.

## Contenido

- `Tensor.h`: definición de la clase `Tensor`, interfaz abstracta `TensorTransform` y clases derivadas `ReLU` y `Sigmoid`.
- `Tensor.cpp`: implementación completa.
- `main.cpp`: pruebas básicas y demostración de la red neuronal solicitada en la tarea.

## Características implementadas

- Tensores 1D, 2D y 3D.
- Memoria dinámica con `double*`.
- Regla de 5:
  - constructor de copia
  - constructor de movimiento
  - asignación por copia
  - asignación por movimiento
  - destructor
- Métodos estáticos:
  - `zeros`
  - `ones`
  - `random`
  - `arange`
- Sobrecarga de operadores:
  - `+`
  - `-`
  - `*` tensor a tensor
  - `*` tensor por escalar
- `view`
- `unsqueeze`
- `concat`
- `dot`
- `matmul`
- Polimorfismo con:
  - `ReLU`
  - `Sigmoid`
- Ejemplo final de red neuronal.

## Compilación

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic main.cpp Tensor.cpp -o tensorpp
```

## Ejecución

```bash
./tensorpp
```

## Nota de diseño

Para que `view` y `unsqueeze` no copien datos y el tensor original siga siendo válido, se comparte el mismo bloque dinámico `double*` mediante un conteo manual de referencias. Sin embargo, el constructor de copia y la asignación por copia realizan **deep copy**, tal como pide el enunciado.
