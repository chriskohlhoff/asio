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
// lazy solution
#include "asio/experimental/detail/channel_payload.hpp"
#include <optional>
#include <cassert>

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

template<typename Initiation, typename InitArgTuple, typename Signature,
         typename = bool>
struct has_early_completion_step : false_type
{
};

template<typename Initiation, typename ...InitArgs,typename Signature>
struct has_early_completion_step<
        Initiation, std::tuple<InitArgs...>, Signature,
        decltype(declval<Initiation>().
          try_complete(declval<detail::early_completion_probe<Signature>>(),
                       declval<InitArgs>()...))> : true_type
{
};

template<typename Initiation, typename InitArgTuple, typename ... Signatures>
struct has_early_completion
    : conjunction<has_early_completion_step<Initiation, InitArgTuple, Signatures>...>
{
};

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

template<typename Derived, typename Sig>
struct early_completion_helper_try_tpl_base;

template<typename Derived, typename ... Args>
struct early_completion_helper_try_tpl_base<Derived, void(Args...)>
{
  // not idealy, but does for now
  void operator()(Args ... args)
  {
    static_cast<Derived*>(this)->payload.emplace(experimental::detail::channel_message<void(Args...)>(0, ASIO_MOVE_CAST(Args)(args)...));
  }
};



}

template<typename Initiation, typename Token, typename ... Signatures>
struct early_completion_helper
{
  typedef typename async_result<Token, Signatures...>::return_type return_type;

  Initiation& init;

  std::optional<experimental::detail::channel_payload<Signatures...>> payload;

  struct try_tpl  : detail::early_completion_helper_try_tpl_base<try_tpl, Signatures> ...
  {
    std::optional<experimental::detail::channel_payload<Signatures...>>  &payload;

    using detail::early_completion_helper_try_tpl_base<try_tpl, Signatures> ::operator()...;

    template<typename ... Args>
    void operator()(ASIO_MOVE_ARG(Args) ... args)
    {
      payload = experimental::detail::channel_payload<Signatures...>({0, ASIO_MOVE_CAST(Args)(args)...});
    }
  };

  struct immediate_initiation
  {
    experimental::detail::channel_payload<Signatures...>  &payload;
    template<typename Handler>
    void operator()(ASIO_MOVE_ARG(Handler) handler)
    {
      payload.receive(handler);
    }
  };

  early_completion_helper(Initiation & init) : init(init) {}

  template<typename ... InitArgs>
  bool try_complete_impl(false_type,
                         ASIO_MOVE_ARG(InitArgs)... init_args)
  {
    return false;
  }

  template<typename ... InitArgs>
  bool try_complete_impl(true_type,
                         ASIO_MOVE_ARG(InitArgs)... init_args)
  {
    return ASIO_MOVE_CAST(Initiation)(init).try_complete(try_tpl{{},payload}, ASIO_MOVE_CAST(InitArgs)(init_args)...);
  }

  template<typename ... InitArgs>
  bool try_complete(ASIO_MOVE_ARG(InitArgs)... init_args)
  {
    return this->try_complete_impl(has_early_completion<Initiation, std::tuple<InitArgs...>, Signatures...>{},
                                   ASIO_MOVE_CAST(InitArgs)(init_args)...);
  }

  template<typename RawCompletionToken>
  return_type get_result(ASIO_MOVE_ARG(RawCompletionToken) token)
  {
    return async_result<Token, Signatures...>::initiate(
            immediate_initiation{*payload},
            ASIO_MOVE_CAST(RawCompletionToken)(token));
  }

  template<typename Handler>
  void receive(ASIO_MOVE_ARG(Handler) handler)
  {
    payload->receive(handler);
  }


};

template <typename Token, typename... Signatures>
struct async_result<allow_recursion_t<Token>, Signatures ...>
{
  typedef typename async_result<Token, Signatures...>::return_type return_type;


  template <typename Initiation, typename RawCompletionToken, typename... InitArgs>
  static return_type initiate(ASIO_MOVE_ARG(Initiation) init,
                              ASIO_MOVE_ARG(RawCompletionToken) wrapped_token,
                              ASIO_MOVE_ARG(InitArgs)... init_args)
  {
    early_completion_helper<Initiation,
                            Token,
                            Signatures...> ech(init);
    if (ech.try_complete(ASIO_MOVE_CAST(InitArgs)(init_args)...))
      return ech.get_result(ASIO_MOVE_CAST(Token)(wrapped_token.token));

    return async_result<Token, Signatures...>::initiate(
            ASIO_MOVE_CAST(Initiation)(init),
            ASIO_MOVE_CAST(Token)(wrapped_token.token),
            ASIO_MOVE_CAST(InitArgs)(init_args)...);
  }
};


}

#include "asio/detail/pop_options.hpp"


#endif //ASIO_EARLY_COMPLETE_HPP
