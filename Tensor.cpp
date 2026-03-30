#include "Tensor.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <utility>
using namespace std;
namespace {
string forma_a_texto(const vector<size_t>& forma) {
    ostringstream oss;
    oss << "{";
    for (size_t i = 0; i < forma.size(); ++i) {
        oss << forma[i];
        if (i + 1 < forma.size()) {
            oss << ", ";
        }
    }
    oss << "}";
    return oss.str();
}
vector<size_t> calcular_pasos_local(const vector<size_t>& forma) {
    vector<size_t> pasos(forma.size(), 1);
    for (int i = static_cast<int>(forma.size()) - 2; i >= 0; --i) {
        pasos[i] = pasos[i + 1] * forma[i + 1];
    }
    return pasos;
}
Tensor operacion_elemento_a_elemento(const Tensor& a, const Tensor& b, char op) {
    vector<size_t> forma_a = a.forma();
    vector<size_t> forma_b = b.forma();

    auto calcular_forma_broadcast_local = [](const vector<size_t>& x,
                                             const vector<size_t>& y) {
        size_t max_dims = max(x.size(), y.size());
        vector<size_t> resultado(max_dims, 1);

        for (size_t i = 0; i < max_dims; ++i) {
            size_t dim_x = 1;
            size_t dim_y = 1;

            if (i >= max_dims - x.size()) {
                dim_x = x[i - (max_dims - x.size())];
            }
            if (i >= max_dims - y.size()) {
                dim_y = y[i - (max_dims - y.size())];
            }

            if (dim_x != dim_y && dim_x != 1 && dim_y != 1) {
                throw invalid_argument("Formas incompatibles para broadcasting: " +
                                       forma_a_texto(x) + " y " + forma_a_texto(y));
            }

            resultado[i] = max(dim_x, dim_y);
        }

        if (resultado.size() > 3) {
            throw invalid_argument("El resultado excede el maximo de 3 dimensiones");
        }

        return resultado;
    };

    auto offset_broadcast_local = [](const vector<size_t>& forma_original,
                                     const vector<size_t>& pasos_originales,
                                     const vector<size_t>& forma_resultado,
                                     size_t indice_lineal) {
        vector<size_t> coords_resultado(forma_resultado.size(), 0);

        for (int i = static_cast<int>(forma_resultado.size()) - 1; i >= 0; --i) {
            coords_resultado[i] = indice_lineal % forma_resultado[i];
            indice_lineal /= forma_resultado[i];
        }

        size_t offset = 0;
        size_t delta = forma_resultado.size() - forma_original.size();

        for (size_t i = 0; i < forma_original.size(); ++i) {
            size_t coord = coords_resultado[i + delta];
            if (forma_original[i] == 1) {
                coord = 0;
            }
            offset += coord * pasos_originales[i];
        }

        return offset;
    };

    vector<size_t> forma_resultado = calcular_forma_broadcast_local(forma_a, forma_b);
    vector<size_t> pasos_a = calcular_pasos_local(forma_a);
    vector<size_t> pasos_b = calcular_pasos_local(forma_b);

    size_t tamano_resultado = 1;
    for (size_t dim : forma_resultado) {
        tamano_resultado *= dim;
    }

    vector<double> valores_resultado(tamano_resultado, 0.0);

    for (size_t i = 0; i < tamano_resultado; ++i) {
        size_t offset_a = offset_broadcast_local(forma_a, pasos_a, forma_resultado, i);
        size_t offset_b = offset_broadcast_local(forma_b, pasos_b, forma_resultado, i);

        switch (op) {
            case '+':
                valores_resultado[i] = a.datos()[offset_a] + b.datos()[offset_b];
                break;
            case '-':
                valores_resultado[i] = a.datos()[offset_a] - b.datos()[offset_b];
                break;
            case '*':
                valores_resultado[i] = a.datos()[offset_a] * b.datos()[offset_b];
                break;
            default:
                throw invalid_argument("Operacion elementwise no soportada");
        }
    }

    return Tensor(forma_resultado, valores_resultado);
}

}

void Tensor::validar_forma(const vector<size_t>& forma) const {
    if (forma.empty() || forma.size() > 3) {
        throw invalid_argument("El tensor debe tener entre 1 y 3 dimensiones");
    }
    for (size_t dim : forma) {
        if (dim == 0) {
            throw invalid_argument("Las dimensiones del tensor no pueden ser cero");
        }
    }
}

size_t Tensor::calcular_tamano_total(const vector<size_t>& forma) {
    if (forma.empty() || forma.size() > 3) {
        throw invalid_argument("El tensor debe tener entre 1 y 3 dimensiones");
    }

    size_t total = 1;
    for (size_t dim : forma) {
        if (dim == 0) {
            throw invalid_argument("Las dimensiones del tensor no pueden ser cero");
        }
        total *= dim;
    }
    return total;
}

void Tensor::calcular_pasos() {
    pasos_.assign(forma_.size(), 1);
    for (int i = static_cast<int>(forma_.size()) - 2; i >= 0; --i) {
        pasos_[i] = pasos_[i + 1] * forma_[i + 1];
    }
}

void Tensor::asignar_y_llenar(const vector<double>& valores) {
    datos_ = new double[tamano_total_];
    contador_ref_ = new size_t(1);
    for (size_t i = 0; i < tamano_total_; ++i) {
        datos_[i] = valores[i];
    }
}

void Tensor::liberar() {
    if (contador_ref_ != nullptr) {
        --(*contador_ref_);
        if (*contador_ref_ == 0) {
            delete[] datos_;
            delete contador_ref_;
        }
    }

    datos_ = nullptr;
    contador_ref_ = nullptr;
    tamano_total_ = 0;
    forma_.clear();
    pasos_.clear();
}

void Tensor::retener() {
    if (contador_ref_ != nullptr) {
        ++(*contador_ref_);
    }
}

Tensor::Tensor()
    : tamano_total_(0), datos_(nullptr), contador_ref_(nullptr) {}

Tensor::Tensor(const vector<size_t>& forma, const vector<double>& valores)
    : forma_(forma), tamano_total_(0), datos_(nullptr), contador_ref_(nullptr) {
    validar_forma(forma_);
    tamano_total_ = calcular_tamano_total(forma_);

    if (valores.size() != tamano_total_) {
        throw invalid_argument("La cantidad de valores no coincide con el producto de las dimensiones");
    }

    calcular_pasos();
    asignar_y_llenar(valores);
}

Tensor::Tensor(initializer_list<size_t> forma, initializer_list<double> valores)
    : Tensor(vector<size_t>(forma), vector<double>(valores)) {}

Tensor::Tensor(const vector<size_t>& forma,
               double* datos_compartidos,
               size_t* contador_ref_compartido,
               bool compartir_almacenamiento_existente)
    : forma_(forma), tamano_total_(0), datos_(nullptr), contador_ref_(nullptr) {
    (void)compartir_almacenamiento_existente;
    validar_forma(forma_);
    tamano_total_ = calcular_tamano_total(forma_);
    calcular_pasos();
    datos_ = datos_compartidos;
    contador_ref_ = contador_ref_compartido;
    retener();
}

Tensor::Tensor(const Tensor& other)
    : forma_(other.forma_),
      pasos_(other.pasos_),
      tamano_total_(other.tamano_total_),
      datos_(nullptr),
      contador_ref_(nullptr) {
    if (other.datos_ == nullptr) {
        return;
    }

    datos_ = new double[tamano_total_];
    contador_ref_ = new size_t(1);
    for (size_t i = 0; i < tamano_total_; ++i) {
        datos_[i] = other.datos_[i];
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : forma_(move(other.forma_)),
      pasos_(move(other.pasos_)),
      tamano_total_(other.tamano_total_),
      datos_(other.datos_),
      contador_ref_(other.contador_ref_) {
    other.tamano_total_ = 0;
    other.datos_ = nullptr;
    other.contador_ref_ = nullptr;
    other.forma_.clear();
    other.pasos_.clear();
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }

    liberar();

    forma_ = other.forma_;
    pasos_ = other.pasos_;
    tamano_total_ = other.tamano_total_;

    if (other.datos_ != nullptr) {
        datos_ = new double[tamano_total_];
        contador_ref_ = new size_t(1);
        for (size_t i = 0; i < tamano_total_; ++i) {
            datos_[i] = other.datos_[i];
        }
    }

    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    liberar();

    forma_ = move(other.forma_);
    pasos_ = move(other.pasos_);
    tamano_total_ = other.tamano_total_;
    datos_ = other.datos_;
    contador_ref_ = other.contador_ref_;

    other.tamano_total_ = 0;
    other.datos_ = nullptr;
    other.contador_ref_ = nullptr;
    other.forma_.clear();
    other.pasos_.clear();

    return *this;
}

Tensor::~Tensor() {
    liberar();
}
Tensor Tensor::ceros(const vector<size_t>& forma) {
    size_t total = calcular_tamano_total(forma);
    return Tensor(forma, vector<double>(total, 0.0));
}

Tensor Tensor::unos(const vector<size_t>& forma) {
    size_t total = calcular_tamano_total(forma);
    return Tensor(forma, vector<double>(total, 1.0));
}

Tensor Tensor::aleatorio(const vector<size_t>& forma, double min, double max) {
    if (min >= max) {
        throw invalid_argument("En aleatorio, min debe ser menor que max");
    }

    size_t total = calcular_tamano_total(forma);
    vector<double> valores(total);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(min, max);

    for (size_t i = 0; i < total; ++i) {
        valores[i] = dist(gen);
    }

    return Tensor(forma, valores);
}
Tensor Tensor::rango(int inicio, int fin) {
    if (fin <= inicio) {
        throw invalid_argument("En rango, fin debe ser mayor que inicio");
    }

    vector<double> valores;
    valores.reserve(static_cast<size_t>(fin - inicio));
    for (int i = inicio; i < fin; ++i) {
        valores.push_back(static_cast<double>(i));
    }

    return Tensor({valores.size()}, valores);
}
Tensor Tensor::aplicar(const TransformacionTensor& transformacion) const {
    return transformacion.aplicar(*this);
}
Tensor Tensor::vista(const vector<size_t>& nueva_forma) const {
    validar_forma(nueva_forma);
    size_t nuevo_tamano = calcular_tamano_total(nueva_forma);
    if (nuevo_tamano != tamano_total_) {
        throw invalid_argument("vista requiere que el numero total de elementos se mantenga constante");
    }
    return Tensor(nueva_forma, datos_, contador_ref_, true);
}

Tensor Tensor::expandir(size_t dim) const {
    if (forma_.size() >= 3) {
        throw invalid_argument("expandir excede el maximo de 3 dimensiones");
    }
    if (dim > forma_.size()) {
        throw invalid_argument("Posicion invalida para expandir");
    }

    vector<size_t> nueva_forma = forma_;
    nueva_forma.insert(nueva_forma.begin() + static_cast<long>(dim), 1);
    return Tensor(nueva_forma, datos_, contador_ref_, true);
}

Tensor Tensor::concatenar(const vector<Tensor>& tensores, size_t dim) {
    if (tensores.empty()) {
        throw invalid_argument("concatenar requiere al menos un tensor");
    }

    const vector<size_t>& forma_base = tensores[0].forma_;
    if (dim >= forma_base.size()) {
        throw invalid_argument("Dimension invalida en concatenar");
    }

    vector<size_t> forma_resultado = forma_base;
    forma_resultado[dim] = 0;

    for (const Tensor& t : tensores) {
        if (t.forma_.size() != forma_base.size()) {
            throw invalid_argument("Todos los tensores deben tener el mismo numero de dimensiones en concatenar");
        }
        for (size_t i = 0; i < forma_base.size(); ++i) {
            if (i != dim && t.forma_[i] != forma_base[i]) {
                throw invalid_argument("Formas incompatibles para concatenar");
            }
        }
        forma_resultado[dim] += t.forma_[dim];
    }

    size_t tamano_resultado = calcular_tamano_total(forma_resultado);
    vector<double> valores_resultado;
    valores_resultado.reserve(tamano_resultado);

    if (forma_base.size() == 1) {
        for (const Tensor& t : tensores) {
            for (size_t i = 0; i < t.tamano_total_; ++i) {
                valores_resultado.push_back(t.datos_[i]);
            }
        }
    } else if (forma_base.size() == 2) {
        if (dim == 0) {
            for (const Tensor& t : tensores) {
                for (size_t i = 0; i < t.tamano_total_; ++i) {
                    valores_resultado.push_back(t.datos_[i]);
                }
            }
        } else {
            size_t filas = forma_base[0];
            for (size_t f = 0; f < filas; ++f) {
                for (const Tensor& t : tensores) {
                    for (size_t c = 0; c < t.forma_[1]; ++c) {
                        valores_resultado.push_back(t.datos_[f * t.forma_[1] + c]);
                    }
                }
            }
        }
    } else {
        size_t A = forma_base[0];
        size_t B = forma_base[1];
        size_t C = forma_base[2];

        if (dim == 0) {
            for (const Tensor& t : tensores) {
                for (size_t i = 0; i < t.tamano_total_; ++i) {
                    valores_resultado.push_back(t.datos_[i]);
                }
            }
        } else if (dim == 1) {
            for (size_t i = 0; i < A; ++i) {
                for (const Tensor& t : tensores) {
                    for (size_t j = 0; j < t.forma_[1]; ++j) {
                        for (size_t k = 0; k < C; ++k) {
                            size_t indice = i * t.forma_[1] * C + j * C + k;
                            valores_resultado.push_back(t.datos_[indice]);
                        }
                    }
                }
            }
        } else {
            for (size_t i = 0; i < A; ++i) {
                for (size_t j = 0; j < B; ++j) {
                    for (const Tensor& t : tensores) {
                        for (size_t k = 0; k < t.forma_[2]; ++k) {
                            size_t indice = i * B * t.forma_[2] + j * t.forma_[2] + k;
                            valores_resultado.push_back(t.datos_[indice]);
                        }
                    }
                }
            }
        }
    }

    return Tensor(forma_resultado, valores_resultado);
}

const vector<size_t>& Tensor::forma() const {
    return forma_;
}

size_t Tensor::ndim() const {
    return forma_.size();
}

size_t Tensor::tamano() const {
    return tamano_total_;
}

double* Tensor::datos() {
    return datos_;
}

const double* Tensor::datos() const {
    return datos_;
}

Tensor Tensor::zeros(const vector<size_t>& shape) {
    return ceros(shape);
}

Tensor Tensor::ones(const vector<size_t>& shape) {
    return unos(shape);
}

Tensor Tensor::random(const vector<size_t>& shape, double min, double max) {
    return aleatorio(shape, min, max);
}

Tensor Tensor::arange(int start, int end) {
    return rango(start, end);
}

Tensor Tensor::apply(const TransformacionTensor& transformacion) const {
    return aplicar(transformacion);
}

Tensor Tensor::view(const vector<size_t>& nueva_forma) const {
    return vista(nueva_forma);
}

Tensor Tensor::unsqueeze(size_t dim) const {
    return expandir(dim);
}

Tensor Tensor::concat(const vector<Tensor>& tensores, size_t dim) {
    return concatenar(tensores, dim);
}

const vector<size_t>& Tensor::shape() const {
    return forma();
}

size_t Tensor::size() const {
    return tamano();
}

double* Tensor::data() {
    return datos();
}

const double* Tensor::data() const {
    return datos();
}

double& Tensor::operator[](size_t index) {
    if (index >= tamano_total_) {
        throw out_of_range("Indice fuera de rango");
    }
    return datos_[index];
}

const double& Tensor::operator[](size_t index) const {
    if (index >= tamano_total_) {
        throw out_of_range("Indice fuera de rango");
    }
    return datos_[index];
}

vector<size_t> Tensor::calcular_forma_broadcast(const vector<size_t>& a,
                                                const vector<size_t>& b) {
    size_t max_dims = max(a.size(), b.size());
    vector<size_t> resultado(max_dims, 1);

    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim_a = 1;
        size_t dim_b = 1;

        if (i >= max_dims - a.size()) {
            dim_a = a[i - (max_dims - a.size())];
        }
        if (i >= max_dims - b.size()) {
            dim_b = b[i - (max_dims - b.size())];
        }

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            throw invalid_argument("Formas incompatibles para broadcasting: " +
                                   forma_a_texto(a) + " y " + forma_a_texto(b));
        }

        resultado[i] = max(dim_a, dim_b);
    }

    if (resultado.size() > 3) {
        throw invalid_argument("El resultado excede el maximo de 3 dimensiones");
    }

    return resultado;
}

size_t Tensor::offset_broadcast(const vector<size_t>& forma_original,
                                const vector<size_t>& pasos_originales,
                                const vector<size_t>& forma_resultado,
                                size_t indice_lineal) {
    vector<size_t> coords_resultado(forma_resultado.size(), 0);

    for (int i = static_cast<int>(forma_resultado.size()) - 1; i >= 0; --i) {
        coords_resultado[i] = indice_lineal % forma_resultado[i];
        indice_lineal /= forma_resultado[i];
    }

    size_t offset = 0;
    size_t delta = forma_resultado.size() - forma_original.size();

    for (size_t i = 0; i < forma_original.size(); ++i) {
        size_t coord = coords_resultado[i + delta];
        if (forma_original[i] == 1) {
            coord = 0;
        }
        offset += coord * pasos_originales[i];
    }

    return offset;
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    return operacion_elemento_a_elemento(a, b, '+');
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    return operacion_elemento_a_elemento(a, b, '-');
}

Tensor operator*(const Tensor& a, const Tensor& b) {
    return operacion_elemento_a_elemento(a, b, '*');
}

Tensor operator*(const Tensor& a, double scalar) {
    vector<double> valores_resultado(a.tamano_total_);
    for (size_t i = 0; i < a.tamano_total_; ++i) {
        valores_resultado[i] = a.datos_[i] * scalar;
    }
    return Tensor(a.forma_, valores_resultado);
}

Tensor operator*(double scalar, const Tensor& a) {
    return a * scalar;
}

Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.forma_.size() != 1 || b.forma_.size() != 1) {
        throw invalid_argument("dot requiere dos tensores unidimensionales");
    }
    if (a.forma_[0] != b.forma_[0]) {
        throw invalid_argument("dot requiere vectores del mismo tamano");
    }

    double suma = 0.0;
    for (size_t i = 0; i < a.forma_[0]; ++i) {
        suma += a.datos_[i] * b.datos_[i];
    }

    return Tensor({1}, {suma});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.forma_.size() != 2 || b.forma_.size() != 2) {
        throw invalid_argument("matmul requiere dos tensores bidimensionales");
    }
    if (a.forma_[1] != b.forma_[0]) {
        throw invalid_argument("Formas incompatibles para matmul: " +
                               forma_a_texto(a.forma_) + " y " + forma_a_texto(b.forma_));
    }

    size_t filas = a.forma_[0];
    size_t interna = a.forma_[1];
    size_t columnas = b.forma_[1];

    vector<double> valores_resultado(filas * columnas, 0.0);

    for (size_t i = 0; i < filas; ++i) {
        for (size_t j = 0; j < columnas; ++j) {
            double suma = 0.0;
            for (size_t k = 0; k < interna; ++k) {
                suma += a.datos_[i * interna + k] * b.datos_[k * columnas + j];
            }
            valores_resultado[i * columnas + j] = suma;
        }
    }

    return Tensor({filas, columnas}, valores_resultado);
}

Tensor ReLU::aplicar(const Tensor& t) const {
    vector<double> valores_resultado(t.tamano_total_);
    for (size_t i = 0; i < t.tamano_total_; ++i) {
        valores_resultado[i] = max(0.0, t.datos_[i]);
    }
    return Tensor(t.forma_, valores_resultado);
}

Tensor Sigmoid::aplicar(const Tensor& t) const {
    vector<double> valores_resultado(t.tamano_total_);
    for (size_t i = 0; i < t.tamano_total_; ++i) {
        valores_resultado[i] = 1.0 / (1.0 + exp(-t.datos_[i]));
    }
    return Tensor(t.forma_, valores_resultado);
}