//
// experimental/linked_continuation.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2021 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_EXPERIMENTAL_LINKED_CONTINUATION_HPP
#define ASIO_EXPERIMENTAL_LINKED_CONTINUATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"
#include "asio/associator.hpp"
#include "asio/error.hpp"
#include <tuple>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {

/// A vocabulary type that links an operation with a dependent handler.
template <class CompletionHandler, class Init, class... InitArgs>
struct linked_continuation
{
  CompletionHandler completion_handler;
  Init init;
  std::tuple<InitArgs...> init_args;

  template <class... Args>
  void operator()(std::error_code e, Args&&...)
  {
    if (e)
    {
      std::move(completion_handler)(e);
    }
    else
    {
      std::apply(
          [this](auto&&... args)
          {
            std::move(init)(std::move(completion_handler),
                std::forward<decltype(args)>(args)...);
          }, std::move(init_args));
    }
  }
};

} // namespace experimental

template <template <typename, typename> class Associator,
    class CompletionHandler, class Init, class... InitArgs,
    class DefaultCandidate>
struct associator<Associator,
    experimental::linked_continuation<CompletionHandler, Init, InitArgs...>,
    DefaultCandidate>
  : Associator<CompletionHandler, DefaultCandidate>
{
  static typename Associator<CompletionHandler, DefaultCandidate>::type get(
      const experimental::linked_continuation<
        CompletionHandler, Init, InitArgs...>& op,
      const DefaultCandidate& c = DefaultCandidate()) noexcept
  {
    return Associator<CompletionHandler, DefaultCandidate>::get(
        op.completion_handler, c);
  }
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_EXPERIMENTAL_LINKED_CONTINUATION_HPP
