//
// select_interrupter.cpp
// ~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/select_interrupter.hpp"
#include <boost/throw_exception.hpp>
#include "asio/socket_error.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"
#if !defined(_WIN32)
#include <fcntl.h>
#endif // !defined(_WIN32)

namespace asio {
namespace detail {

select_interrupter::
select_interrupter()
  : read_descriptor_(-1),
    write_descriptor_(-1)
{
#if defined(_WIN32)
  socket_holder acceptor(socket_ops::socket(AF_INET, SOCK_STREAM,
        IPPROTO_TCP));
  if (acceptor.get() == invalid_socket)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  int opt = 1;
  socket_ops::setsockopt(acceptor.get(), SOL_SOCKET, SO_REUSEADDR, &opt,
      sizeof(opt));

  inet_addr_v4_type addr;
  socket_addr_len_type addr_len = sizeof(addr);
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = inet_addr("127.0.0.1");
  addr.sin_port = 0;
  if (socket_ops::bind(acceptor.get(), (const socket_addr_type*)&addr,
        addr_len) == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  if (getsockname(acceptor.get(), (socket_addr_type*)&addr, &addr_len)
      == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  if (socket_ops::listen(acceptor.get(), SOMAXCONN) == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  socket_holder client(socket_ops::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
  if (client.get() == invalid_socket)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  if (socket_ops::connect(client.get(), (const socket_addr_type*)&addr,
        addr_len) == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  socket_holder server(socket_ops::accept(acceptor.get(), 0, 0));
  if (server.get() == invalid_socket)
    boost::throw_exception(socket_error(socket_ops::get_error()));
  
  ioctl_arg_type non_blocking = 1;
  if (socket_ops::ioctl(client.get(), FIONBIO, &non_blocking))
    boost::throw_exception(socket_error(socket_ops::get_error()));

  opt = 1;
  socket_ops::setsockopt(client.get(), IPPROTO_TCP, TCP_NODELAY, &opt,
      sizeof(opt));

  non_blocking = 1;
  if (socket_ops::ioctl(server.get(), FIONBIO, &non_blocking))
    boost::throw_exception(socket_error(socket_ops::get_error()));

  opt = 1;
  socket_ops::setsockopt(server.get(), IPPROTO_TCP, TCP_NODELAY, &opt,
      sizeof(opt));

  read_descriptor_ = server.release();
  write_descriptor_ = client.release();
#else
  int pipe_fds[2];
  if (pipe(pipe_fds) == 0)
  {
    read_descriptor_ = pipe_fds[0];
    ::fcntl(read_descriptor_, F_SETFL, O_NONBLOCK);
    write_descriptor_ = pipe_fds[1];
    ::fcntl(write_descriptor_, F_SETFL, O_NONBLOCK);
  }
#endif
}

select_interrupter::
~select_interrupter()
{
#if defined(_WIN32)
  if (read_descriptor_ != invalid_socket)
    socket_ops::close(read_descriptor_);
  if (write_descriptor_ != invalid_socket)
    socket_ops::close(write_descriptor_);
#else
  if (read_descriptor_ != -1)
    close(read_descriptor_);
  if (write_descriptor_ != -1)
    close(write_descriptor_);
#endif
}

void
select_interrupter::
interrupt()
{
#if defined(_WIN32)
  char byte = 0;
  socket_ops::send(write_descriptor_, &byte, 1, 0);
#else
  char byte = 0;
  write(write_descriptor_, &byte, 1);
#endif
}

bool
select_interrupter::
reset()
{
#if defined(_WIN32)
  char data[1024];
  int bytes_read = socket_ops::recv(read_descriptor_, data, sizeof(data), 0);
  bool was_interrupted = (bytes_read > 0);
  while (bytes_read == sizeof(data))
    bytes_read = socket_ops::recv(read_descriptor_, data, sizeof(data), 0);
  return was_interrupted;
#else
  char data[1024];
  int bytes_read = ::read(read_descriptor_, data, sizeof(data));
  bool was_interrupted = (bytes_read > 0);
  while (bytes_read == sizeof(data))
    bytes_read = ::read(read_descriptor_, data, sizeof(data));
  return was_interrupted;
#endif
}

socket_type
select_interrupter::
read_descriptor() const
{
  return read_descriptor_;
}

} // namespace detail
} // namespace asio
