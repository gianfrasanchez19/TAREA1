#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <initializer_list>
#include <string>
using namespace std;
class TransformacionTensor;
class TensorTransform;
class Tensor {
private:
    vector<size_t> forma_;
    vector<size_t> pasos_;
    size_t tamano_total_;
    double* datos_;
    size_t* contador_ref_;

    void validar_forma(const vector<size_t>& forma) const;
    void calcular_pasos();
    void asignar_y_llenar(const vector<double>& valores);
    void liberar();
    void retener();
    Tensor(const vector<size_t>& forma,
           double* datos_compartidos,
           size_t* contador_ref_compartido,
           bool compartir_almacenamiento_existente);

    static size_t calcular_tamano_total(const vector<size_t>& forma);
    static vector<size_t> calcular_forma_broadcast(const vector<size_t>& a,
                                                   const vector<size_t>& b);
    static size_t offset_broadcast(const vector<size_t>& forma_original,
                                   const vector<size_t>& pasos_originales,
                                   const vector<size_t>& forma_resultado,
                                   size_t indice_lineal);

public:
    Tensor();
    Tensor(const vector<size_t>& forma, const vector<double>& valores);
    Tensor(initializer_list<size_t> forma, initializer_list<double> valores);

    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor();

    // API en español
    static Tensor ceros(const vector<size_t>& forma);
    static Tensor unos(const vector<size_t>& forma);
    static Tensor aleatorio(const vector<size_t>& forma, double min, double max);
    static Tensor rango(int inicio, int fin);

    Tensor aplicar(const TransformacionTensor& transformacion) const;
    Tensor vista(const vector<size_t>& nueva_forma) const;
    Tensor expandir(size_t dim) const;
    static Tensor concatenar(const vector<Tensor>& tensores, size_t dim);

    const vector<size_t>& forma() const;
    size_t ndim() const;
    size_t tamano() const;
    double* datos();
    const double* datos() const;

    double& operator[](size_t index);
    const double& operator[](size_t index) const;
    static Tensor zeros(const vector<size_t>& shape);
    static Tensor ones(const vector<size_t>& shape);
    static Tensor random(const vector<size_t>& shape, double min, double max);
    static Tensor arange(int start, int end);

    Tensor apply(const TransformacionTensor& transformacion) const;
    Tensor view(const vector<size_t>& nueva_forma) const;
    Tensor unsqueeze(size_t dim) const;
    static Tensor concat(const vector<Tensor>& tensores, size_t dim);

    const vector<size_t>& shape() const;
    size_t size() const;
    double* data();
    const double* data() const;

    friend Tensor operator+(const Tensor& a, const Tensor& b);
    friend Tensor operator-(const Tensor& a, const Tensor& b);
    friend Tensor operator*(const Tensor& a, const Tensor& b);
    friend Tensor operator*(const Tensor& a, double scalar);
    friend Tensor operator*(double scalar, const Tensor& a);

    friend Tensor dot(const Tensor& a, const Tensor& b);
    friend Tensor matmul(const Tensor& a, const Tensor& b);

    friend class ReLU;
    friend class Sigmoid;
};

class TransformacionTensor {
public:
    virtual Tensor aplicar(const Tensor& t) const = 0;
    virtual ~TransformacionTensor() = default;
};

class TensorTransform {
public:
    virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU : public TransformacionTensor, public TensorTransform {
public:
    Tensor aplicar(const Tensor& t) const override;
    Tensor apply(const Tensor& t) const override { return aplicar(t); }
};

class Sigmoid : public TransformacionTensor, public TensorTransform {
public:
    Tensor aplicar(const Tensor& t) const override;
    Tensor apply(const Tensor& t) const override { return aplicar(t); }
};

#endif