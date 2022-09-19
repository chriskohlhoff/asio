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

#include "asio/associator.hpp"
#include "asio/async_result.hpp"
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
  void operator()(ASIO_MOVE_ARG(Args) ... ) const;
};

}

template<typename Initiation, typename Signature,
         typename = void>
struct has_early_completion_step : false_type
{
};

template<typename Initiation, typename Signature>
struct has_early_completion_step<
        Initiation, Signature,
        decltype(declval<Initiation>().complete_early(declval<detail::early_completion_probe<Signature>>()))> : true_type
{
};

template<typename Initiation, typename ... Signatures>
struct has_early_completion
    : conjunction<has_early_completion_step<Initiation, Signatures>...>
{
};


namespace detail
{

template<typename Initiation, typename Handler>
ASIO_CONSTEXPR void invoke_early_completion_impl(ASIO_MOVE_ARG(Initiation), ASIO_MOVE_ARG(Handler), false_type)
{
}

template<typename Initiation, typename Handler>
void invoke_early_completion_impl(ASIO_MOVE_ARG(Initiation) init, ASIO_MOVE_ARG(Handler) handler, true_type)
{
  ASIO_MOVE_CAST(Initiation)(init).complete_early(ASIO_MOVE_CAST(Handler)(handler));
}

template<typename Initiation>
ASIO_CONSTEXPR bool check_early_completion_impl(ASIO_MOVE_ARG(Initiation), false_type)
{
  return false;
}

template<typename Initiation>
bool check_early_completion_impl(ASIO_MOVE_ARG(Initiation) init, true_type)
{
  return ASIO_MOVE_CAST(Initiation)(init).can_complete_early();
}

}

template<typename ... Signatures, typename Initiation, typename Handler>
void invoke_early_completion(ASIO_MOVE_ARG(Initiation) init, ASIO_MOVE_ARG(Handler) handler)
{
  detail::invoke_early_completion_impl(
          ASIO_MOVE_CAST(Initiation)(init),
          ASIO_MOVE_CAST(Handler)(handler),
          has_early_completion<typename decay<Initiation>::type, Signatures...>{});
}

template<typename ... Signatures, typename Initiation>
bool check_early_completion(ASIO_MOVE_ARG(Initiation) init)
{
  return detail::check_early_completion_impl(
          ASIO_MOVE_CAST(Initiation)(init),
          has_early_completion<typename decay<Initiation>::type, Signatures...>{});
}



template<typename Token>
struct allow_recursion_t
{
  template<typename T>
  allow_recursion_t(ASIO_MOVE_ARG(T) token) : token(ASIO_MOVE_CAST(T)(token)) {}
  Token token;
};

template<typename Token>
allow_recursion_t<typename decay<Token>::type> allow_recursion(ASIO_MOVE_ARG(Token) token)
{
  return allow_recursion_t<typename decay<Token>::type>
          (
              ASIO_MOVE_CAST(Token)(token)
          );
}


namespace detail
{

template<typename Initiation>
struct immediate_initiate
{
  Initiation init;
  template<typename Init_>
  immediate_initiate(ASIO_MOVE_ARG(Init_) init) : init(ASIO_MOVE_CAST(Init_)(init)) {}

  template<typename Handler>
  void operator()(ASIO_MOVE_ARG(Handler) handler)
  {
    invoke_early_completion(ASIO_MOVE_OR_LVALUE(Initiation)(init),
                            ASIO_MOVE_CAST(Handler)(handler));
  }

};

}

template <typename Init, typename... Signatures>
struct async_result<allow_recursion_t<Init>, Signatures ...>
{
  typedef typename async_result<Init, Signatures...>::return_type return_type;


  template <typename Initiation, typename RawCompletionToken, typename... InitArgs>
  static return_type initiate(ASIO_MOVE_ARG(Initiation) init,
                              ASIO_MOVE_ARG(RawCompletionToken) wrapped_token,
                              ASIO_MOVE_ARG(InitArgs)... init_args)
  {
    if (check_early_completion<Signatures...>(ASIO_MOVE_CAST(Initiation)(init)))
        return async_result<Init, Signatures...>::initiate(
                detail::immediate_initiate<Initiation>(ASIO_MOVE_CAST(Initiation)(init)),
                ASIO_MOVE_CAST(Init)(wrapped_token.token),
                ASIO_MOVE_CAST(InitArgs)(init_args)...);

    return async_result<Init, Signatures...>::initiate(
            ASIO_MOVE_CAST(Initiation)(init),
            ASIO_MOVE_CAST(Init)(wrapped_token.token),
            ASIO_MOVE_CAST(InitArgs)(init_args)...);
  }
};


}

#include "asio/detail/pop_options.hpp"


#endif //ASIO_EARLY_COMPLETE_HPP
