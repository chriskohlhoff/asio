//
// socket_error.cpp
// ~~~~~~~~~~~~~~~~
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

#include "asio/socket_error.hpp"

namespace asio {

socket_error::
socket_error(
    int code)
  : std::runtime_error("Socket error"),
    code_(code)
{
}

int
socket_error::
code() const
{
  return code_;
}

std::string
socket_error::
message() const
{
#if defined(_WIN32)
  if (code_ == ENOMEM || code_ == EPERM || code_ == EAGAIN)
    return strerror(code_);

  void* msg_buf;
  ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
      | FORMAT_MESSAGE_IGNORE_INSERTS, 0, code_,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&msg_buf, 0, 0);
  std::string msg((LPCTSTR)msg_buf);
  ::LocalFree(msg_buf);
  return msg;
#else
  char buf[256] = "";
  std::string err_str(strerror_r(code_, buf, sizeof(buf)));
  return err_str;
#endif
}

socket_error::
operator void*() const
{
  return code_ == success ? 0 : reinterpret_cast<void*>(1);
}

bool
socket_error::
operator!() const
{
  return code_ == success;
}

} // namespace asio
