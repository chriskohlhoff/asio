//
// win_thread.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
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

#if defined(_WIN32)

#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"
#include <new>
#include <process.h>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

extern "C" unsigned int __stdcall asio_detail_win_thread_function(void* arg);

class win_thread
  : private boost::noncopyable
{
public:
  // Constructor.
  template <typename Function>
  win_thread(Function f)
  {
    func_base* arg = new func<Function>(f);
    unsigned int thread_id = 0;
    thread_ = reinterpret_cast<HANDLE>(::_beginthreadex(0, 0,
          asio_detail_win_thread_function, arg, 0, &thread_id));
    if (!thread_)
      throw std::bad_alloc();
  }

  // Destructor.
  ~win_thread()
  {
    ::CloseHandle(thread_);
  }

  // Wait for the thread to exit.
  void join()
  {
    ::WaitForSingleObject(thread_, INFINITE);
  }

private:
  friend unsigned int __stdcall asio_detail_win_thread_function(void* arg);

  class func_base
  {
  public:
    virtual ~func_base() {}
    virtual void run() = 0;
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
};

inline unsigned int __stdcall asio_detail_win_thread_function(void* arg)
{
  win_thread::func_base* func =
    static_cast<win_thread::func_base*>(arg);
  func->run();
  delete func;
  return 0;
}

} // namespace detail
} // namespace asio

#endif // defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_THREAD_HPP
