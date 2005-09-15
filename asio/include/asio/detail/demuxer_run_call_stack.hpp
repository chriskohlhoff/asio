//
// demuxer_run_call_stack.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_DEMUXER_RUN_CALL_STACK_HPP
#define ASIO_DETAIL_DEMUXER_RUN_CALL_STACK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/tss_ptr.hpp"

namespace asio {
namespace detail {

// Helper class to determine whether or not the current thread is inside an
// invocation of demuxer::run() for a specified demuxer.
template <typename Demuxer_Service>
class demuxer_run_call_stack
{
public:
  // Context class automatically pushes a demuxer on to the stack.
  class context
    : private boost::noncopyable
  {
  public:
    // Push the demuxer on to the stack.
    explicit context(Demuxer_Service* d)
      : demuxer_service_(d),
        next_(demuxer_run_call_stack<Demuxer_Service>::top_)
    {
      demuxer_run_call_stack<Demuxer_Service>::top_ = this;
    }

    // Pop the demuxer from the stack.
    ~context()
    {
      demuxer_run_call_stack<Demuxer_Service>::top_ = next_;
    }

  private:
    friend class demuxer_run_call_stack<Demuxer_Service>;

    // The demuxer service associated with the context.
    Demuxer_Service* demuxer_service_;

    // The next element in the stack.
    context* next_;
  };

  friend class context;

  // Determine whether the specified demuxer is on the stack.
  static bool contains(Demuxer_Service* d)
  {
    context* elem = top_;
    while (elem)
    {
      if (elem->demuxer_service_ == d)
        return true;
      elem = elem->next_;
    }
    return false;
  }

private:
  // The top of the stack of demuxer::run() calls for the current thread.
  static tss_ptr<context> top_;
};

template <typename Demuxer_Service>
tss_ptr<typename demuxer_run_call_stack<Demuxer_Service>::context>
demuxer_run_call_stack<Demuxer_Service>::top_;

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_DEMUXER_RUN_CALL_STACK_HPP
