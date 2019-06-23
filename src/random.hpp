//
// Created by nagato0614 on 2019-06-24.
//

#ifndef NAGATOLIB_SRC_RANDOM_H_
#define NAGATOLIB_SRC_RANDOM_H_

#include <random>

namespace nagato {

template<typename Engine = std::mt19937>
class Random {
 public:
  Random(std::size_t seed = 0) : engine_(seed) {}
  ~Random() = default;

  template<class Return, class From, class To>
  Return uniform_int_distribution(From from, To to) {
	static_assert(std::is_arithmetic<Return>(),
				  "Return is not arithmetric");
	std::uniform_int_distribution<Return> distribution(from, to);
	return distribution(engine_);
  }

  template<class Return, class From, class To>
  Return uniform_real_distribution(From from, To to) {
	static_assert(std::is_arithmetic<Return>(),
				  "Return is not arithmetric");
	std::uniform_real_distribution<Return> distribution(from, to);
	return distribution(engine_);
  }

 private:
  Engine engine_;
};
}

#endif //NAGATOLIB_SRC_RANDOM_H_,
