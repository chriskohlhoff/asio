//
// win_thread.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_THREAD_HPP
#define ASIO_DETAIL_WIN_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include "asio/detail/pop_options.hpp"

#if defined(BOOST_WINDOWS) && !defined(UNDER_CE)

#include "asio/error.hpp"
#include "asio/system_error.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/throw_exception.hpp>
#include <memory>
#include <process.h>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

unsigned int __stdcall win_thread_function(void* arg);

class win_thread
  : private noncopyable
{
public:
  // The purpose of the thread.
  enum purpose { internal, external };

  // Constructor.
  template <typename Function>
  win_thread(Function f, purpose p = internal)
    : exit_event_(0)
  {
    std::auto_ptr<func_base> arg(new func<Function>(f));

    ::HANDLE entry_event = 0;
    if (p == internal)
    {
      arg->entry_event_ = entry_event = ::CreateEvent(0, true, false, 0);
      if (!entry_event)
      {
        DWORD last_error = ::GetLastError();
        asio::system_error e(
            asio::error_code(last_error,
              asio::error::get_system_category()),
            "thread.entry_event");
        boost::throw_exception(e);
      }

      arg->exit_event_ = exit_event_ = ::CreateEvent(0, true, false, 0);
      if (!exit_event_)
      {
        DWORD last_error = ::GetLastError();
        ::CloseHandle(entry_event);
        asio::system_error e(
            asio::error_code(last_error,
              asio::error::get_system_category()),
            "thread.exit_event");
        boost::throw_exception(e);
      }
    }

    unsigned int thread_id = 0;
    thread_ = reinterpret_cast<HANDLE>(::_beginthreadex(0, 0,
          win_thread_function, arg.get(), 0, &thread_id));
    if (!thread_)
    {
      DWORD last_error = ::GetLastError();
      if (entry_event)
        ::CloseHandle(entry_event);
      if (exit_event_)
        ::CloseHandle(exit_event_);
      asio::system_error e(
          asio::error_code(last_error,
            asio::error::get_system_category()),
          "thread");
      boost::throw_exception(e);
    }
    arg.release();

    if (entry_event)
    {
      ::WaitForSingleObject(entry_event, INFINITE);
      ::CloseHandle(entry_event);
    }
  }

  // Destructor.
  ~win_thread()
  {
    ::CloseHandle(thread_);

    // The exit_event_ handle is deliberately allowed to leak here since it
    // is an error for the owner of an internal thread not to join() it.
  }

  // Wait for the thread to exit.
  void join()
  {
    if (exit_event_)
    {
      ::WaitForSingleObject(exit_event_, INFINITE);
      ::CloseHandle(exit_event_);
      ::TerminateThread(thread_, 0);
    }
    else
    {
      ::WaitForSingleObject(thread_, INFINITE);
    }
  }

private:
  friend unsigned int __stdcall win_thread_function(void* arg);

  class func_base
  {
  public:
    virtual ~func_base() {}
    virtual void run() = 0;
    ::HANDLE entry_event_;
    ::HANDLE exit_event_;
  };

  template <typename Function>
  class func
    : public func_base
  {
  public:
    func(Function f)
      : f_(f)
    {
    }

    virtual void run()
    {
      f_();
    }

  private:
    Function f_;
  };

  ::HANDLE thread_;
  ::HANDLE exit_event_;
};

inline unsigned int __stdcall win_thread_function(void* arg)
{
  std::auto_ptr<win_thread::func_base> func(
      static_cast<win_thread::func_base*>(arg));

  if (func->entry_event_)
    ::SetEvent(func->entry_event_);

  func->run();

  if (HANDLE exit_event = func->exit_event_)
  {
    func.reset();
    ::SetEvent(exit_event);
    ::Sleep(INFINITE);
  }

  return 0;
}

} // namespace detail
} // namespace asio

#endif // defined(BOOST_WINDOWS) && !defined(UNDER_CE)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_THREAD_HPP
