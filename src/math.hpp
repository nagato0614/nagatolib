//
// Created by nagato0614 on 2019-06-23.
//

#ifndef NAGATOLIB_SRC_MATH_H_
#define NAGATOLIB_SRC_MATH_H_

#include <math.h>

namespace nagato {

// -----------------------------------------------------------------------------
// reference : https://cpplover.blogspot.com/2010/11/blog-post_20.html
template<typename Type>
Type sqrt(Type s) {

  #ifdef NAGATO_MATH
  Type x = s / 2.0; // Is there any better way to determine initial value?
  Type last_x = 0.0; // the value one before the last step

  while (x != last_x) // until the difference is not significant
  { // apply Babylonian method step
	last_x = x;
	x = (x + s / x) / 2.0;
  }

  return x;
  #else
  return std::sqrt(s);
  #endif
}
// -----------------------------------------------------------------------------

}
#endif //NAGATOLIB_SRC_MATH_H_
