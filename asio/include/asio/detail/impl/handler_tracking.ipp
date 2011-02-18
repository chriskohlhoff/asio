//
// detail/impl/handler_tracking.ipp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2011 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_IMPL_HANDLER_TRACKING_IPP
#define ASIO_DETAIL_IMPL_HANDLER_TRACKING_IPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_ENABLE_HANDLER_TRACKING)

#include <cstdio>
#include <unistd.h>
#include "asio/detail/handler_tracking.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct handler_tracking::tracking_state
{
  static_mutex mutex_;
  boost::uint64_t next_id_;
  tss_ptr<completion>* current_completion_;
};

handler_tracking::tracking_state* handler_tracking::get_state()
{
  static tracking_state state = { ASIO_STATIC_MUTEX_INIT, 1, 0 };
  return &state;
}

void handler_tracking::init()
{
  static tracking_state* state = get_state();

  state->mutex_.init();

  static_mutex::scoped_lock lock(state->mutex_);
  if (state->current_completion_ == 0)
    state->current_completion_ = new tss_ptr<completion>;
}

void handler_tracking::creation(handler_tracking::tracked_handler* h,
    const char* object_type, void* object, const char* op_name)
{
  using namespace std; // For sprintf (or equivalent).

  static tracking_state* state = get_state();

  static_mutex::scoped_lock lock(state->mutex_);
  h->id_ = state->next_id_++;
  lock.unlock();

  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration now =
    boost::posix_time::microsec_clock::universal_time() - epoch;

  boost::uint64_t current_id = 0;
  if (completion* current_completion = *state->current_completion_)
    current_id = current_completion->id_;

  char line[256] = "";
  int line_length = sprintf(line,
      "@asio|%llu.%06llu|%llu*%llu|%.20s@%p.%.20s\n",
      static_cast<boost::uint64_t>(now.total_seconds()),
      static_cast<boost::uint64_t>(now.total_microseconds() % 1000000),
      current_id, h->id_, object_type, object, op_name);

  ::write(STDERR_FILENO, line, line_length);
}

handler_tracking::completion::completion(handler_tracking::tracked_handler* h)
  : id_(h->id_),
    invoked_(false),
    next_(*get_state()->current_completion_)
{
  *get_state()->current_completion_ = this;
}

handler_tracking::completion::~completion()
{
  if (id_)
  {
    using namespace std; // For sprintf (or equivalent).

    boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
    boost::posix_time::time_duration now =
      boost::posix_time::microsec_clock::universal_time() - epoch;

    char line[256] = "";
    int line_length = sprintf(line, "@asio|%llu.%06llu|%c%llu|\n",
        static_cast<boost::uint64_t>(now.total_seconds()),
        static_cast<boost::uint64_t>(now.total_microseconds() % 1000000),
        invoked_ ? '!' : '~', id_);

    ::write(STDERR_FILENO, line, line_length);
  }

  *get_state()->current_completion_ = next_;
}

void handler_tracking::completion::invocation_begin()
{
  using namespace std; // For sprintf (or equivalent).

  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration now =
    boost::posix_time::microsec_clock::universal_time() - epoch;

  char line[256] = "";
  int line_length = sprintf(line, "@asio|%llu.%06llu|>%llu|\n",
      static_cast<boost::uint64_t>(now.total_seconds()),
      static_cast<boost::uint64_t>(now.total_microseconds() % 1000000), id_);

  ::write(STDERR_FILENO, line, line_length);

  invoked_ = true;
}

void handler_tracking::completion::invocation_begin(
    const asio::error_code& ec)
{
  using namespace std; // For sprintf (or equivalent).

  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration now =
    boost::posix_time::microsec_clock::universal_time() - epoch;

  char line[256] = "";
  int line_length = sprintf(line, "@asio|%llu.%06llu|>%llu|ec=%.20s:%d\n",
      static_cast<boost::uint64_t>(now.total_seconds()),
      static_cast<boost::uint64_t>(now.total_microseconds() % 1000000),
      id_, ec.category().name(), ec.value());

  ::write(STDERR_FILENO, line, line_length);

  invoked_ = true;
}

void handler_tracking::completion::invocation_begin(
    const asio::error_code& ec, std::size_t bytes_transferred)
{
  using namespace std; // For sprintf (or equivalent).

  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration now =
    boost::posix_time::microsec_clock::universal_time() - epoch;

  char line[256] = "";
  int line_length = sprintf(line,
      "@asio|%llu.%06llu|>%llu|ec=%.20s:%d,bytes_transferred=%llu\n",
      static_cast<boost::uint64_t>(now.total_seconds()),
      static_cast<boost::uint64_t>(now.total_microseconds() % 1000000),
      id_, ec.category().name(), ec.value(),
      static_cast<boost::uint64_t>(bytes_transferred));

  ::write(STDERR_FILENO, line, line_length);

  invoked_ = true;
}

void handler_tracking::completion::invocation_begin(
    const asio::error_code& ec, int signal_number)
{
  using namespace std; // For sprintf (or equivalent).

  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration now =
    boost::posix_time::microsec_clock::universal_time() - epoch;

  char line[256] = "";
  int line_length = sprintf(line,
      "@asio|%llu.%06llu|>%llu|ec=%.20s:%d,signal_number=%d\n",
      static_cast<boost::uint64_t>(now.total_seconds()),
      static_cast<boost::uint64_t>(now.total_microseconds() % 1000000),
      id_, ec.category().name(), ec.value(), signal_number);

  ::write(STDERR_FILENO, line, line_length);

  invoked_ = true;
}

void handler_tracking::completion::invocation_begin(
    const asio::error_code& ec, const char* arg)
{
  using namespace std; // For sprintf (or equivalent).

  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration now =
    boost::posix_time::microsec_clock::universal_time() - epoch;

  char line[256] = "";
  int line_length = sprintf(line,
      "@asio|%llu.%06llu|>%llu|ec=%.20s:%d,%.20s\n",
      static_cast<boost::uint64_t>(now.total_seconds()),
      static_cast<boost::uint64_t>(now.total_microseconds() % 1000000),
      id_, ec.category().name(), ec.value(), arg);

  ::write(STDERR_FILENO, line, line_length);

  invoked_ = true;
}

void handler_tracking::completion::invocation_end()
{
  if (id_)
  {
    using namespace std; // For sprintf (or equivalent).

    boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
    boost::posix_time::time_duration now =
      boost::posix_time::microsec_clock::universal_time() - epoch;

    char line[256] = "";
    int line_length = sprintf(line, "@asio|%llu.%06llu|<%llu|\n",
        static_cast<boost::uint64_t>(now.total_seconds()),
        static_cast<boost::uint64_t>(now.total_microseconds() % 1000000), id_);

    ::write(STDERR_FILENO, line, line_length);

    id_ = 0;
  }
}

void handler_tracking::operation(const char* object_type,
    void* object, const char* op_name)
{
  using namespace std; // For sprintf (or equivalent).

  static tracking_state* state = get_state();

  boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
  boost::posix_time::time_duration now =
    boost::posix_time::microsec_clock::universal_time() - epoch;

  boost::uint64_t current_id = 0;
  if (completion* current_completion = *state->current_completion_)
    current_id = current_completion->id_;

  char line[256] = "";
  int line_length = sprintf(line,
      "@asio|%llu.%06llu|%llu|%.20s@%p.%.20s\n",
      static_cast<boost::uint64_t>(now.total_seconds()),
      static_cast<boost::uint64_t>(now.total_microseconds() % 1000000),
      current_id, object_type, object, op_name);

  ::write(STDERR_FILENO, line, line_length);
}

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_ENABLE_HANDLER_TRACKING)

#endif // ASIO_DETAIL_IMPL_HANDLER_TRACKING_IPP
