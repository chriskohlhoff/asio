//
// lazy.hpp
// ~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_LAZY_HPP
#define ASIO_LAZY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include <tuple>
#include <utility>
#include "asio/async_result.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

struct lazy_t {};

template <ASIO_COMPLETION_SIGNATURE Signature>
class async_result<lazy_t, Signature>
{
public:
  template <typename Initiation, typename... InitArgs>
  static auto initiate(Initiation initiation, lazy_t, InitArgs... init_args)
  {
    return [
        initiation = std::move(initiation),
        init_arg_pack = std::make_tuple(std::move(init_args)...)
      ](auto&& token) mutable
    {
      return std::apply(
          [&](auto&&... args)
          {
            return async_initiate<decltype(token), Signature>(
                std::move(initiation), token,
                std::forward<decltype(args)>(args)...);
          },
          std::move(init_arg_pack)
        );
    };
  }
};

/// A completion token object that indicates that an asynchronous object should
/// be packaged as a function object, for lazy initiation.
#if defined(ASIO_HAS_CONSTEXPR) || defined(GENERATING_DOCUMENTATION)
constexpr lazy_t lazy;
#elif defined(ASIO_MSVC)
__declspec(selectany) lazy_t lazy;
#endif

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_LAZY_HPP
