//
// use_fiber.hpp
// ~~~~~~~~~~~~~
//
// Copyright (c) 2003-2019 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_USE_FIBER_HPP
#define ASIO_USE_FIBER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include <functional>
#include <tuple>
#include <utility>
#include <boost/context/fiber.hpp>
#include "asio/async_result.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

struct use_fiber
{
  explicit use_fiber(boost::context::fiber& f) : fiber(f) {}
  boost::context::fiber& fiber;
};

template <typename R, typename... Args>
class async_result<use_fiber, R(Args...)>
{
public:
  template <typename Initiation, typename... InitArgs>
  static auto initiate(Initiation&& initiation,
      use_fiber u, InitArgs&&... init_args)
  {
    std::tuple<Args...>* result_ptr;

    u.fiber = std::move(u.fiber).resume_with(
        [&](boost::context::fiber f)
        {
          std::forward<Initiation>(initiation)(
              [&, f = std::move(f)](Args... results) mutable
              {
                std::tuple<Args...> result(std::move(results)...);
                result_ptr = &result;
                std::move(f).resume();
              },
              std::forward<InitArgs>(init_args)...
            );

          return boost::context::fiber{};
        }
      );

    return std::move(*result_ptr);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_USE_FIBER_HPP
