//
// Created by nagato0614 on 2019-06-22.
//

#ifndef NAGATOLIB_SRC_NAGATO_H_
#define NAGATOLIB_SRC_NAGATO_H_

#include "math.hpp"
#include "vector.hpp"
#include "random.hpp"
#include "matrix.hpp"
#include "matrix_n.hpp"
#include "thread.hpp"
#include "tensor.hpp"
#include "network.hpp"
#include "../Metal/timer.hpp"

namespace nagato {
// Vector
using Vector4f = Vector<float, 4>;
using Vector3f = Vector<float, 3>;
using Vector2f = Vector<float, 2>;

// Matrix Vector
template<typename T, std::size_t size>
using MatrixVector = Matrix<T, 1, size>;
template<std::size_t size>
using Vectorf = Matrix<float, 1, size>;

// Matrix
template<std::size_t row, std::size_t col>
using Matrixf = Matrix<float, row, col>;
template<std::size_t row, std::size_t col>
using Matrixd = Matrix<double, row, col>;

// MatrixN
using MatrixNf = MatrixN<float>;

}

#endif //NAGATOLIB_SRC_NAGATO_H_
