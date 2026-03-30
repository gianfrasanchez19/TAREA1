#include "Tensor.h"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace std;
void imprimir_forma(const Tensor& t, const string& nombre) {
    cout << nombre << " forma = {";
    for (size_t i = 0; i < t.forma().size(); ++i) {
        cout << t.forma()[i];
        if (i + 1 < t.forma().size()) {
            cout << ", ";
        }
    }
    cout << "}" << endl;
}
void imprimir_primeros_valores(const Tensor& t, const string& nombre, size_t cantidad = 8) {
    cout << nombre << " primeros valores: ";
    size_t limite = min(cantidad, t.tamano());
    for (size_t i = 0; i < limite; ++i) {
        cout << fixed << setprecision(4) << t[i] << " ";
    }
    if (t.tamano() > limite) {
        cout << "...";
    }
    cout << endl;
}
int main() {
    try {
        cout << "==== Pruebas basicas de Tensor++ ====\n";

        Tensor ceros_tensor = Tensor::ceros({2, 3});
        Tensor unos_tensor = Tensor::unos({2, 3});
        Tensor rango_tensor = Tensor::rango(0, 6).vista({2, 3});
        Tensor aleatorio_tensor = Tensor::aleatorio({2, 3}, -1.0, 1.0);

        imprimir_forma(ceros_tensor, "ceros");
        imprimir_forma(unos_tensor, "unos");
        imprimir_forma(rango_tensor, "rango");
        imprimir_forma(aleatorio_tensor, "aleatorio");

        Tensor suma = unos_tensor + rango_tensor;
        Tensor resta = rango_tensor - unos_tensor;
        Tensor producto = rango_tensor * unos_tensor;
        Tensor escalado = rango_tensor * 2.0;

        imprimir_primeros_valores(suma, "unos + rango");
        imprimir_primeros_valores(resta, "rango - unos");
        imprimir_primeros_valores(producto, "rango * unos");
        imprimir_primeros_valores(escalado, "rango * 2.0");

        Tensor vector = Tensor::rango(0, 3);
        Tensor vector_expandido_0 = vector.expandir(0);
        Tensor vector_expandido_1 = vector.expandir(1);
        imprimir_forma(vector, "vector");
        imprimir_forma(vector_expandido_0, "vector.expandir(0)");
        imprimir_forma(vector_expandido_1, "vector.expandir(1)");

        Tensor concatenado = Tensor::concatenar({Tensor::unos({2, 3}), Tensor::ceros({2, 3})}, 0);
        imprimir_forma(concatenado, "concatenar dim 0");

        Tensor x = Tensor::rango(-5, 5).vista({2, 5});
        ReLU relu;
        Sigmoid sigmoide;
        Tensor x_relu = x.aplicar(relu);
        Tensor x_sigmoide = x.aplicar(sigmoide);
        imprimir_primeros_valores(x, "x");
        imprimir_primeros_valores(x_relu, "ReLU(x)");
        imprimir_primeros_valores(x_sigmoide, "Sigmoide(x)");

        Tensor punto_a = Tensor::rango(1, 4);
        Tensor punto_b = Tensor::rango(4, 7);
        Tensor punto_resultado = dot(punto_a, punto_b);
        imprimir_primeros_valores(punto_resultado, "punto([1,2,3],[4,5,6])", 1);

        Tensor matriz1 = Tensor::rango(0, 6).vista({2, 3});
        Tensor matriz2 = Tensor::rango(0, 12).vista({3, 4});
        Tensor multiplicacion_matriz = matmul(matriz1, matriz2);
        imprimir_forma(multiplicacion_matriz, "matmul(matriz1,matriz2)");
        imprimir_primeros_valores(multiplicacion_matriz, "matmul(matriz1,matriz2)");

        cout << "\n==== Red neuronal solicitada en el enunciado ====\n";

        Tensor entrada = Tensor::aleatorio({1000, 20, 20}, 0.0, 1.0);
        imprimir_forma(entrada, "Entrada");

        Tensor aplanado = entrada.vista({1000, 400});
        imprimir_forma(aplanado, "Despues de vista");

        Tensor pesos1 = Tensor::aleatorio({400, 100}, -0.5, 0.5);
        Tensor sesgos1 = Tensor::aleatorio({1, 100}, -0.1, 0.1);
        Tensor oculto_lineal = matmul(aplanado, pesos1);
        imprimir_forma(oculto_lineal, "matmul(aplanado, pesos1)");

        Tensor oculto_con_sesgos = oculto_lineal + sesgos1;
        imprimir_forma(oculto_con_sesgos, "+ sesgos1");

        Tensor oculto = oculto_con_sesgos.aplicar(relu);
        imprimir_forma(oculto, "ReLU");

        Tensor pesos2 = Tensor::aleatorio({100, 10}, -0.5, 0.5);
        Tensor sesgos2 = Tensor::aleatorio({1, 10}, -0.1, 0.1);
        Tensor salida_lineal = matmul(oculto, pesos2);
        imprimir_forma(salida_lineal, "matmul(oculto, pesos2)");

        Tensor salida_con_sesgos = salida_lineal + sesgos2;
        imprimir_forma(salida_con_sesgos, "+ sesgos2");

        Tensor salida = salida_con_sesgos.aplicar(sigmoide);
        imprimir_forma(salida, "Salida final con Sigmoide");
        imprimir_primeros_valores(salida, "Salida final", 10);

        cout << "\nTodo ejecuto correctamente.\n";
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}