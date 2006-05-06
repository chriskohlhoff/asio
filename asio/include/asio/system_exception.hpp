//
// error.hpp
// ~~~~~~~~~
//
// Copyright (c) 2003-2006 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SYSTEM_EXCEPTION_HPP
#define ASIO_SYSTEM_EXCEPTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <boost/scoped_ptr.hpp>
#include <cerrno>
#include <cstring>
#include <exception>
#include <string>
#include <boost/detail/workaround.hpp>
#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
# include <iostream>
#endif // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
#include "asio/detail/pop_options.hpp"

#include "asio/detail/win_local_free_on_block_exit.hpp"

namespace asio {

/// The system_exception class is used to represent system conditions that
/// prevent the library from operating correctly.
class system_exception
  : public std::exception
{
public:
  /// Construct with a specific context and error code.
  system_exception(const std::string& context, int code)
    : context_(context),
      code_(code)
  {
  }

  /// Copy constructor.
  system_exception(const system_exception& e)
    : std::exception(e),
      context_(e.context_),
      code_(e.code_)
  {
  }

  /// Destructor.
  virtual ~system_exception() throw ()
  {
  }

  /// Assignment operator.
  system_exception& operator=(const system_exception& e)
  {
    context_ = e.context_;
    code_ = e.code_;
    what_.reset();
    return *this;
  }

  /// Get a string representation of the exception.
  virtual const char* what() const throw ()
  {
#if defined(BOOST_WINDOWS) || defined(__CYGWIN__)
    try
    {
      if (!what_)
      {
        char* msg = 0;
        DWORD length = ::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER
            | FORMAT_MESSAGE_FROM_SYSTEM
            | FORMAT_MESSAGE_IGNORE_INSERTS, 0, code_,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (char*)&msg, 0, 0);
        detail::win_local_free_on_block_exit local_free_obj(msg);
        if (length && msg[length - 1] == '\n')
          msg[--length] = '\0';
        if (length && msg[length - 1] == '\r')
          msg[--length] = '\0';
        if (length)
        {
          std::string tmp(context_);
          tmp += ": ";
          tmp += msg;
          what_.reset(new std::string(tmp));
        }
        else
        {
          return "asio system_exception";
        }
      }
      return what_->c_str();
    }
    catch (std::exception&)
    {
      return "asio system_exception";
    }
#elif defined(__sun) || defined(__QNX__)
    return strerror(code_);
#elif defined(__MACH__) && defined(__APPLE__)
    try
    {
      char buf[256] = "";
      strerror_r(code_, buf, sizeof(buf));
      std::string tmp(context_);
      tmp += ": ";
      tmp += buf;
      what_.reset(new std::string(tmp));
      return what_->c_str();
    }
    catch (std::exception&)
    {
      return "asio system_exception";
    }
#else
    try
    {
      char buf[256] = "";
      std::string tmp(context_);
      tmp += ": ";
      tmp += strerror_r(code_, buf, sizeof(buf));
      what_.reset(new std::string(tmp));
      return what_->c_str();
    }
    catch (std::exception&)
    {
      return "asio system_exception";
    }
#endif
  }

  /// Get the implementation-defined context associated with the exception.
  const std::string& context() const
  {
    return context_;
  }

  /// Get the implementation-defined code associated with the exception.
  int code() const
  {
    return code_;
  }

private:
  // The context associated with the error.
  std::string context_;

  // The code associated with the error.
  int code_;

  // The string representation of the error.
  mutable boost::scoped_ptr<std::string> what_;
};

/// Output the string associated with a system exception.
/**
 * Used to output a human-readable string that is associated with a system
 * exception.
 *
 * @param os The output stream to which the string will be written.
 *
 * @param e The exception to be written.
 *
 * @return The output stream.
 *
 * @relates asio::system_exception
 */
#if BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
std::ostream& operator<<(std::ostream& os, const system_exception& e)
{
  os << e.what();
  return os;
}
#else // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))
template <typename Ostream>
Ostream& operator<<(Ostream& os, const system_exception& e)
{
  os << e.what();
  return os;
}
#endif // BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x564))

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SYSTEM_EXCEPTION_HPP
