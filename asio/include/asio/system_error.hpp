//
// system_error.hpp
// ~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2008 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SYSTEM_ERROR_HPP
#define ASIO_SYSTEM_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/config.hpp>
#include <boost/scoped_ptr.hpp>
#include <cerrno>
#include <exception>
#include <string>
#include "asio/detail/pop_options.hpp"

#include "asio/error_code.hpp"

namespace asio {

/// The system_error class is used to represent system conditions that
/// prevent the library from operating correctly.
class system_error
  : public std::exception
{
public:
  /// Construct with an error code.
  system_error(const error_code& code)
    : code_(code),
      context_()
  {
  }

  /// Construct with an error code and context.
  system_error(const error_code& code, const std::string& context)
    : code_(code),
      context_(context)
  {
  }

  /// Copy constructor.
  system_error(const system_error& other)
    : std::exception(other),
      code_(other.code_),
      context_(other.context_),
      what_()
  {
  }

  /// Destructor.
  virtual ~system_error() throw ()
  {
  }

  /// Assignment operator.
  system_error& operator=(const system_error& e)
  {
    context_ = e.context_;
    code_ = e.code_;
    what_.reset();
    return *this;
  }

  /// Get a string representation of the exception.
  virtual const char* what() const throw ()
  {
    try
    {
      if (!what_)
      {
        std::string tmp(context_);
        if (tmp.length())
          tmp += ": ";
        tmp += code_.message();
        what_.reset(new std::string(tmp));
      }
      return what_->c_str();
    }
    catch (std::exception&)
    {
      return "system_error";
    }
  }

  /// Get the error code associated with the exception.
  error_code code() const
  {
    return code_;
  }

private:
  // The code associated with the error.
  error_code code_;

  // The context associated with the error.
  std::string context_;

  // The string representation of the error.
  mutable boost::scoped_ptr<std::string> what_;
};

} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SYSTEM_ERROR_HPP
