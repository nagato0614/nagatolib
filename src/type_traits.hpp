//
// Created by nagato0614 on 2019/09/19.
//

#ifndef NAGATOLIB_SRC_TYPE_TRAITS_HPP_
#define NAGATOLIB_SRC_TYPE_TRAITS_HPP_

namespace nagato {

// -----------------------------------------------------------------------------
template<typename Type>
using remove_const_reference_t
= std::remove_const_t<std::remove_reference_t<Type>>;

// -----------------------------------------------------------------------------
// 参考 : https://ameblo.jp/michirushiina/entry-12386417744.html
// 可変テンプレート引数の型すべてが同じ型かどうか比較する

template<typename T, typename... Ts>
struct are_all_same {
	static constexpr bool value = true;
};

template<typename T, typename Head, typename... Rest>
struct are_all_same<T, Head, Rest...> {
	static constexpr bool value = false;
};

template<typename T, typename... Rest>
struct are_all_same<T, T, Rest...> {
	static constexpr bool value = are_all_same<T, Rest...>::value;
};

// -----------------------------------------------------------------------------
// 可変テンプレート引数のがすべて算術型か調べる

template<typename T, typename... Ts>
struct is_all_arithmetic {
	static constexpr bool value =
			std::is_arithmetic<remove_const_reference_t<T>>::value;
};

template<typename T, typename Head, typename... Rest>
struct is_all_arithmetic<T, Head, Rest...> {
	static constexpr bool value =
			std::is_arithmetic<remove_const_reference_t<T>>::value &&
					is_all_arithmetic<Head, Rest...>::value;
};

// -----------------------------------------------------------------------------

}

#endif //NAGATOLIB_SRC_TYPE_TRAITS_HPP_
