//
// handler_token.hpp
// ~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2012 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ARCHETYPES_HANDLER_TOKEN_HPP
#define ARCHETYPES_HANDLER_TOKEN_HPP

#include <asio/handler_token.hpp>
#include <asio/handler_type.hpp>

namespace archetypes {

struct lazy_handler
{
};

struct concrete_handler
{
  concrete_handler(lazy_handler)
  {
  }

  template <typename Arg1>
  void operator()(Arg1)
  {
  }

  template <typename Arg1, typename Arg2>
  void operator()(Arg1, Arg2)
  {
  }
};

} // namespace archetypes

namespace asio {

template <typename Signature>
struct handler_type<archetypes::lazy_handler, Signature>
{
  typedef archetypes::concrete_handler type;
};

template <>
class handler_token<archetypes::concrete_handler>
{
public:
  // The return type of the initiating function.
  typedef int type;

  // Construct a token from a given handler.
  explicit handler_token(archetypes::concrete_handler&)
  {
  }

  // Obtain the value to be returned from the initiating function.
  type get()
  {
    return 42;
  }
};

} // namespace asio

#endif // ARCHETYPES_HANDLER_TOKEN_HPP
