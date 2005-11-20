//
// openssl_init.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2005 Voipster / Indrek dot Juhani at voipster dot com
// Copyright (c) 2005 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DETAIL_OPENSSL_INIT_HPP
#define ASIO_SSL_DETAIL_OPENSSL_INIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <vector>
#include <boost/shared_ptr.hpp>
#include "asio/detail/pop_options.hpp"

#include "asio/detail/mutex.hpp"
#include "asio/ssl/detail/openssl_types.hpp"

namespace asio {
namespace ssl {
namespace detail {

template <bool Do_Init = true>
class openssl_init
  : private boost::noncopyable
{
private:
  // Structure to perform the actual initialisation.
  class do_init
  {
  public:
    do_init()
    {
      if (Do_Init)
      {
        ::SSL_library_init();
        ::SSL_load_error_strings();        

        mutexes_.resize(::CRYPTO_num_locks());
        for (size_t i = 0; i < mutexes_.size(); ++i)
          mutexes_[i].reset(new asio::detail::mutex);
        ::CRYPTO_set_locking_callback(&do_init::openssl_locking_func);

        ::OpenSSL_add_ssl_algorithms();
      }
    }

    ~do_init()
    {
      if (Do_Init)
      {
        ::CRYPTO_set_locking_callback(0);
      }
    }

    // Helper function to manage a do_init singleton. The static instance of the
    // openssl_init object ensures that this function is always called before
    // main, and therefore before any other threads can get started. The do_init
    // instance must be static in this function to ensure that it gets
    // initialised before any other global objects try to use it.
    static boost::shared_ptr<do_init> instance()
    {
      static boost::shared_ptr<do_init> init(new do_init);
      return init;
    }

  private:
    static void openssl_locking_func(int mode, int n, 
      const char *file, int line)
    {
  	  if (mode & CRYPTO_LOCK)
        instance()->mutexes_[n]->lock();
	    else
        instance()->mutexes_[n]->unlock();
    }

    // Mutexes to be used in locking callbacks.
    std::vector<boost::shared_ptr<asio::detail::mutex> > mutexes_;
  };

public:
  // Constructor.
  openssl_init()
    : ref_(do_init::instance())
  {
    while (&instance_ == 0); // Ensure openssl_init::instance_ is linked in.
  }

  // Destructor.
  ~openssl_init()
  {
  }

private:
  // Instance to force initialisation of openssl at global scope.
  static openssl_init instance_;

  // Reference to singleton do_init object to ensure that openssl does not get
  // cleaned up until the last user has finished with it.
  boost::shared_ptr<do_init> ref_;
};

template <bool Do_Init>
openssl_init<Do_Init> openssl_init<Do_Init>::instance_;

} // namespace detail
} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DETAIL_OPENSSL_INIT_HPP
