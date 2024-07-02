//
// error_as_deferred.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003-2022 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_ERROR_DEFERRED_HPP
#define ASIO_ERROR_DEFERRED_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)


#include "asio/deferred.hpp"
#include "asio/error.hpp"

#include "asio/detail/config.hpp"

#if (defined(ASIO_HAS_STD_TUPLE) \
    && defined(ASIO_HAS_DECLTYPE) \
    && defined(ASIO_HAS_VARIADIC_TEMPLATES)) \
  || defined(GENERATING_DOCUMENTATION)

#include <tuple>
#include "asio/associator.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/utility.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {


template <typename Signature>
class async_result<asio::error_code, Signature>
{
public:

  struct deferred_intercept_error_code
  {
    asio::error_code &ec;

    template<typename ... Args>
    auto operator()(asio::error_code ec_, Args && ... args) const
      -> deferred_values<Args...>
    {
      ec = ec_;
      return asio::deferred_t::values(std::forward<Args>(args)...);
    }

  };

  template <typename Initiation, typename... InitArgs>
  static auto
  initiate(ASIO_MOVE_ARG(Initiation) initiation,
      asio::error_code & ec, ASIO_MOVE_ARG(InitArgs)... init_args)
      -> decltype(
        deferred_sequence<
          deferred_async_operation<
           Signature, Initiation, InitArgs...>,
          deferred_intercept_error_code>(deferred_init_tag{},
            deferred_async_operation<
              Signature, Initiation, InitArgs...>(
                deferred_init_tag{},
                ASIO_MOVE_CAST(Initiation)(initiation),
                ASIO_MOVE_CAST(InitArgs)(init_args)...),
                std::declval<deferred_intercept_error_code>()))
  {
    return deferred_sequence<
        deferred_async_operation<
          Signature, Initiation, InitArgs...>,
        deferred_intercept_error_code>(deferred_init_tag{},
          deferred_async_operation<
            Signature, Initiation, InitArgs...>(
              deferred_init_tag{},
              ASIO_MOVE_CAST(Initiation)(initiation),
              ASIO_MOVE_CAST(InitArgs)(init_args)...),
          ASIO_MOVE_CAST(deferred_intercept_error_code)(deferred_intercept_error_code{ec}));
    }
};


template <typename Signature>
class async_result<std::exception_ptr, Signature>
{
public:

  struct deferred_intercept_error_code
  {
    std::exception_ptr &ec;

    template<typename ... Args>
    auto operator()(std::exception_ptr ec_, Args && ... args) const
      -> deferred_values<Args...>
    {
      ec = ec_;
      return asio::deferred_t::values(std::forward<Args>(args)...);
    }

  };

  template <typename Initiation, typename... InitArgs>
  static auto
  initiate(ASIO_MOVE_ARG(Initiation) initiation,
      std::exception_ptr & ec, ASIO_MOVE_ARG(InitArgs)... init_args)
      -> decltype(
        deferred_sequence<
          deferred_async_operation<
           Signature, Initiation, InitArgs...>,
          deferred_intercept_error_code>(deferred_init_tag{},
            deferred_async_operation<
              Signature, Initiation, InitArgs...>(
                deferred_init_tag{},
                ASIO_MOVE_CAST(Initiation)(initiation),
                ASIO_MOVE_CAST(InitArgs)(init_args)...),
                std::declval<deferred_intercept_error_code>()))
  {
    return deferred_sequence<
        deferred_async_operation<
          Signature, Initiation, InitArgs...>,
        deferred_intercept_error_code>(deferred_init_tag{},
          deferred_async_operation<
            Signature, Initiation, InitArgs...>(
              deferred_init_tag{},
              ASIO_MOVE_CAST(Initiation)(initiation),
              ASIO_MOVE_CAST(InitArgs)(init_args)...),
          ASIO_MOVE_CAST(deferred_intercept_error_code)(deferred_intercept_error_code{ec}));
    }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"


#endif // (defined(ASIO_HAS_STD_TUPLE)
       //     && defined(ASIO_HAS_DECLTYPE))
       //     && defined(ASIO_HAS_VARIADIC_TEMPLATES))
       //   || defined(GENERATING_DOCUMENTATION)

#endif // ASIO_ERROR_DEFERRED_HPP
