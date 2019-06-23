//
// Created by nagato0614 on 2019-06-24.
//

#ifndef NAGATOLIB_SRC_ASSERT_HPP_
#define NAGATOLIB_SRC_ASSERT_HPP_

#define STATIC_ASSERT_IS_ARITHMETRIC(TYPE) \
      static_assert(std::is_arithmetic<TYPE>(), \
          "Type is not arithmetic")

#endif //NAGATOLIB_SRC_ASSERT_HPP_
