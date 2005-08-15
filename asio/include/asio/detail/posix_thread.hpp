//
// posix_thread.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2005 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_POSIX_THREAD_HPP
#define ASIO_DETAIL_POSIX_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#if !defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <new>
#include <boost/noncopyable.hpp>
#include <pthread.h>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/scoped_lock.hpp"

namespace asio {
namespace detail {

extern "C" void* asio_detail_posix_thread_function(void* arg);

class posix_thread
  : private boost::noncopyable
{
public:
  // Constructor.
  template <typename Function>
  posix_thread(Function f)
    : joined_(false)
  {
    func_base* arg = new func<Function>(f);
    if (::pthread_create(&thread_, 0, asio_detail_posix_thread_function, arg))
      throw std::bad_alloc();
  }

  // Destructor.
  ~posix_thread()
  {
    if (!joined_)
      ::pthread_detach(thread_);
  }

  // Wait for the thread to exit.
  void join()
  {
    if (!joined_)
    {
      ::pthread_join(thread_, 0);
      joined_ = true;
    }
  }

private:
  friend void* asio_detail_posix_thread_function(void* arg);

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

  ::pthread_t thread_;
  bool joined_;
};

inline void* asio_detail_posix_thread_function(void* arg)
{
  posix_thread::func_base* f =
    static_cast<posix_thread::func_base*>(arg);
  f->run();
  delete f;
  return 0;
}

} // namespace detail
} // namespace asio

#endif // !defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_POSIX_THREAD_HPP
