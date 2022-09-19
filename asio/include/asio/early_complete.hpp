//
// early_completion.hpp
// ~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2022 Klemens D. Morgenstern
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ASIO_EARLY_COMPLETE_HPP
#define ASIO_EARLY_COMPLETE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio
{

namespace detail
{

template<typename Signature>
struct early_completion_probe;

template<typename ... Args>
struct early_completion_probe<void(Args...)>
{
  void operator()(ASIO_MOVE_ARG(Args) ... );
};

}

template<typename Signature, typename Initiation, typename = void>
struct has_early_completion : false_type
{
};

template<typename Signature, typename Initiation>
struct has_early_completion<
        Signature, Initiation,
        decltype(std::declval<Initiation>().complete_early(
                std::declval<detail::early_completion_probe<Signature>>())
                )> : true_type
{
};

namespace detail
{

template<typename Signature, typename Initiation, typename Handler>
ASIO_CONSTEXPR bool invoke_early_completion_impl(ASIO_MOVE_ARG(Initiation), ASIO_MOVE_ARG(Handler), false_type)
{
  return false;
}

template<typename Signature, typename Initiation, typename Handler>
bool invoke_early_completion_impl(ASIO_MOVE_ARG(Initiation) init, ASIO_MOVE_ARG(Handler) handler, true_type)
{
  return ASIO_MOVE_CAST(Initiation)(init).complete_early(ASIO_MOVE_CAST(Handler)(handler));
}

}

template<typename Signature, typename Initiation, typename Handler>
bool invoke_early_completion(ASIO_MOVE_ARG(Initiation) init, ASIO_MOVE_ARG(Handler) handler)
{
  return detail::invoke_early_completion_impl(
          ASIO_MOVE_CAST(Initiation)(init),
          ASIO_MOVE_CAST(Handler)(handler),
          has_early_completion<Signature, typename std::decay<Initiation>::type>{});
}

}

#include "asio/detail/pop_options.hpp"


#endif //ASIO_EARLY_COMPLETE_HPP
