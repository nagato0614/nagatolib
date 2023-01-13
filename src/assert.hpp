//
// Created by nagato0614 on 2019-06-24.
//

#ifndef NAGATOLI#pragma once
B_SRC_ASSERT_HPP_
#define NAGATOLIB_SRC_ASSERT_HPP_

#include <type_traits>

// -----------------------------------------------------------------------------

#define STATIC_ASSERT_IS_ARITHMETRIC(TYPE) \
      static_assert(std::is_arithmetic<TYPE>(), \
          "#TYPE is not arithmetic")

#define STATIC_ASSERT_IS_FLOATING_POINT(TYPE) \
      static_assert(std::is_floating_point<TYPE>(), \
          "#TYPE is not arithmetic")

#define STATIC_ASSERT_IS_INTEGER(TYPE) \
        static_assert(std::is_integral<TYPE>(), \
         "#TYPE is not arithmetic")

#define STATIC_ASSERT_IS_UNSIGNED(TYPE) \
        static_assert(std::is_unsigned<TYPE>(), \
         "#TYPE is not unsigned value!")

#define STATIC_ASSERT_IS_POSITIVE_NUMBER(NUMBER) \
      static_assert(NUMBER > 0.0, \
          "#NUMBER is not positive number")

#define ASSERT_INNER(FROM, NUMBER, TO) \
    assert(#FROM <= #NUMBER && #NUMBER < #TO)

#define STATIC_ASSERT_INNER(FROM, NUMBER, TO) \
    static_assert(#FROM <= #NUMBER && #NUMBER < #TO)

#define STATIC_ASSERT_EQUAL(A, B) \
			static_assert(A == B, \
					"NOT EQUEAL!")

// -----------------------------------------------------------------------------


#endif //NAGATOLIB_SRC_ASSERT_HPP_
