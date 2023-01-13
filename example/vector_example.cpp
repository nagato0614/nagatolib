//
// Created by nagato0614 on 2019-06-22.
//
#include <iostream>
#include <vector>

#include "nagatolib.hpp"

template<typename T>
class TD;

using namespace nagato;
using vector3f = Vector<float, 3>;

void showVec(const vector3f &v) {
  std::cout << "---------------------" << std::endl;
  std::printf("%f, %f, %f\n", v[0], v[1], v[2]);
}

int main() {
  constexpr vector3f a{1, 2, 3};
  if constexpr (!a.HasNan()) {
    ;
  }
  vector3f b = {4, 5, 6};
  vector3f c = {7, 8, 9};


  vector3f d = 1 + a + b + c + 1;

  showVec(d);
  showVec(a * b);
  showVec(a - b);
  showVec(a / b);

  std::cout << Dot(a, b) << std::endl;
  std::cout << Dot(c * 2, b + 1) << std::endl;
  vector3f sqrtvec = Sqrt(a);
  showVec(sqrtvec);

  float norm_a = b.Sum();

  std::cout << norm_a << std::endl;
	std::cout << Normalize(a) << std::endl;
	std::cout << a << std::endl;

	std::cout << Vector3f(1, 2, 3) << std::endl;
	std::cout << Vector3f(1.0, 2.0f, 3) << std::endl;
}
