//
// pthread_thread.hpp
// ~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_PTHREAD_THREAD_HPP
#define ASIO_DETAIL_PTHREAD_THREAD_HPP

#include "asio/detail/push_options.hpp"

#if !defined(_WIN32)

#include "asio/detail/push_options.hpp"
#include <boost/noncopyable.hpp>
#include <pthread.h>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/scoped_lock.hpp"

namespace asio {
namespace detail {

extern "C" void* asio_detail_pthread_thread_function(void* arg);

class pthread_thread
  : private boost::noncopyable
{
public:
  // Constructor.
  template <typename Function>
  pthread_thread(Function f)
    : joined_(false)
  {
    func_base* arg = new func<Function>(f);
    ::pthread_create(&thread_, 0, asio_detail_pthread_thread_function, arg);
  }

  // Destructor.
  ~pthread_thread()
  {
    if (!joined_)
      ::pthread_detach(thread_);
  }

  // Wait for the thread to exit.
  void join()
  {
    ::pthread_join(thread_, 0);
    joined_ = true;
  }

private:
  friend void* asio_detail_pthread_thread_function(void* arg);

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

inline void* asio_detail_pthread_thread_function(void* arg)
{
  pthread_thread::func_base* f =
    static_cast<pthread_thread::func_base*>(arg);
  f->run();
  delete f;
  return 0;
}

} // namespace detail
} // namespace asio

#endif // !defined(_WIN32)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_PTHREAD_THREAD_HPP
