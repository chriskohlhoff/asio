//
// select_provider.cpp
// ~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/select_provider.hpp"
#include "asio/completion_context.hpp"
#include "asio/demuxer.hpp"
#include "asio/generic_address.hpp"
#include "asio/socket_error.hpp"
#include "asio/detail/select_op.hpp"
#include "asio/detail/socket_connector_impl.hpp"
#include "asio/detail/socket_holder.hpp"
#include "asio/detail/socket_ops.hpp"

#include "asio/detail/push_options.hpp"
#include <boost/bind.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

select_provider::
select_provider(
    demuxer& d)
  : demuxer_(d),
    selector_(d)
{
}

select_provider::
~select_provider()
{
}

service*
select_provider::
do_get_service(
    const service_type_id& service_type)
{
  if (service_type == dgram_socket_service::id
      || service_type == socket_acceptor_service::id
      || service_type == socket_connector_service::id
      || service_type == stream_socket_service::id)
    return this;
  return 0;
}

void
select_provider::
do_dgram_socket_create(
    dgram_socket_service::impl_type& impl,
    const socket_address& address)
{
  socket_holder sock(socket_ops::socket(address.family(), SOCK_DGRAM,
        IPPROTO_UDP));
  if (sock.get() == invalid_socket)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  int reuse = 1;
  socket_ops::setsockopt(sock.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
      sizeof(reuse));

  if (socket_ops::bind(sock.get(), address.native_address(),
        address.native_size()) == socket_error_retval)
    boost::throw_exception(socket_error(socket_ops::get_error()));

  impl = sock.release();
}

void
select_provider::
do_dgram_socket_destroy(
    dgram_socket_service::impl_type& impl)
{
  if (impl != dgram_socket_service::invalid_impl)
  {
    selector_.close_descriptor(impl);
    socket_ops::close(impl);
    impl = dgram_socket_service::invalid_impl;
  }
}

namespace 
{
  struct sendto_op : public select_op
  {
    demuxer* demuxer_;
    const void* data_;
    size_t length_;
    generic_address destination_;
    dgram_socket_service::sendto_handler handler_;
    completion_context* context_;

    sendto_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      int bytes = socket_ops::sendto(descriptor(), data_, length_, 0,
          destination_.native_address(), destination_.native_size());
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      demuxer_->operation_completed(boost::bind(&sendto_op::do_upcall,
            handler_, error, bytes), *context_);
      delete this;
    }

    virtual void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&sendto_op::do_upcall,
            handler_, error, 0), *context_);
      delete this;
    }

    static void do_upcall(const dgram_socket_service::sendto_handler& handler,
        const socket_error& error, size_t bytes_transferred)
    {
      handler(error, bytes_transferred);
    }
  };
} // namespace

void
select_provider::
do_dgram_socket_async_sendto(
    dgram_socket_service::impl_type& impl,
    const void* data,
    size_t length,
    const socket_address& destination,
    const sendto_handler& handler,
    completion_context& context)
{
  sendto_op* op = new sendto_op(impl);
  op->demuxer_ = &demuxer_;
  op->data_ = data;
  op->length_ = length;
  op->destination_ = destination;
  op->handler_ = handler;
  op->context_ = &context;

  demuxer_.operation_started();

  selector_.start_write_op(*op);
}

namespace
{
  struct recvfrom_op : public select_op
  {
    demuxer* demuxer_;
    void* data_;
    size_t max_length_;
    socket_address* sender_address_;
    dgram_socket_service::recvfrom_handler handler_;
    completion_context* context_;

    recvfrom_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      socket_addr_len_type addr_len = sender_address_->native_size();
      int bytes = socket_ops::recvfrom(descriptor(), data_, max_length_, 0,
          sender_address_->native_address(), &addr_len);
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      sender_address_->native_size(addr_len);
      demuxer_->operation_completed(boost::bind(&recvfrom_op::do_upcall,
            handler_, error, bytes), *context_);
      delete this;
    }

    virtual void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&recvfrom_op::do_upcall,
            handler_, error, 0), *context_);
      delete this;
    }

    static void do_upcall(
        const dgram_socket_service::recvfrom_handler& handler,
        const socket_error& error, size_t bytes_transferred)
    {
      handler(error, bytes_transferred);
    }
  };
} // namespace

void
select_provider::
do_dgram_socket_async_recvfrom(
    dgram_socket_service::impl_type& impl,
    void* data,
    size_t max_length,
    socket_address& sender_address,
    const recvfrom_handler& handler,
    completion_context& context)
{
  recvfrom_op* op = new recvfrom_op(impl);
  op->demuxer_ = &demuxer_;
  op->data_ = data;
  op->max_length_ = max_length;
  op->sender_address_ = &sender_address;
  op->handler_ = handler;
  op->context_ = &context;

  demuxer_.operation_started();

  selector_.start_read_op(*op);
}

void
select_provider::
do_socket_acceptor_destroy(
    socket_acceptor_service::impl_type& impl)
{
  if (impl != socket_acceptor_service::invalid_impl)
  {
    selector_.close_descriptor(impl);
    socket_ops::close(impl);
    impl = socket_acceptor_service::invalid_impl;
  }
}

namespace
{
  struct accept_op : public select_op
  {
    demuxer* demuxer_;
    socket_acceptor_service::peer_type* peer_socket_;
    socket_acceptor_service::accept_handler handler_;
    completion_context* context_;

    accept_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      socket_type new_socket = socket_ops::accept(descriptor(), 0, 0);
      socket_error error(new_socket == invalid_socket
          ? socket_ops::get_error() : socket_error::success);
      peer_socket_->set_impl(new_socket);
      demuxer_->operation_completed(boost::bind(&accept_op::do_upcall,
            handler_, error), *context_);
      delete this;
    }

    virtual void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&accept_op::do_upcall,
            handler_, error), *context_);
      delete this;
    }

    static void do_upcall(
        const socket_acceptor_service::accept_handler& handler,
        const socket_error& error)
    {
      handler(error);
    }
  };
} // namespace

void
select_provider::
do_socket_acceptor_async_accept(
    socket_acceptor_service::impl_type& impl,
    socket_acceptor_service::peer_type& peer_socket,
    const accept_handler& handler,
    completion_context& context)
{
  if (peer_socket.impl() != invalid_socket)
  {
    socket_error error(socket_error::already_connected);
    demuxer_.operation_immediate(
        boost::bind(&accept_op::do_upcall, handler, error));
    return;
  }

  accept_op* op = new accept_op(impl);
  op->demuxer_ = &demuxer_;
  op->peer_socket_ = &peer_socket;
  op->handler_ = handler;
  op->context_ = &context;

  demuxer_.operation_started();

  selector_.start_read_op(*op);
}

namespace
{
  struct accept_with_addr_op : public select_op
  {
    demuxer* demuxer_;
    socket_acceptor_service::peer_type* peer_socket_;
    socket_address* peer_address_;
    socket_acceptor_service::accept_handler handler_;
    completion_context* context_;

    accept_with_addr_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      socket_addr_len_type addr_len = peer_address_->native_size();
      socket_type new_socket = socket_ops::accept(descriptor(),
          peer_address_->native_address(), &addr_len);
      socket_error error(new_socket == invalid_socket
          ? socket_ops::get_error() : socket_error::success);
      peer_address_->native_size(addr_len);
      peer_socket_->set_impl(new_socket);
      demuxer_->operation_completed(boost::bind(&accept_op::do_upcall,
            handler_, error), *context_);
      delete this;
    }

    virtual void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&accept_op::do_upcall,
            handler_, error), *context_);
      delete this;
    }

    static void do_upcall(
        const socket_acceptor_service::accept_handler& handler,
        const socket_error& error)
    {
      handler(error);
    }
  };
} // namespace

void
select_provider::
do_socket_acceptor_async_accept(
    socket_acceptor_service::impl_type& impl,
    socket_acceptor_service::peer_type& peer_socket,
    socket_address& peer_address,
    const accept_handler& handler,
    completion_context& context)
{
  if (peer_socket.impl() != invalid_socket)
  {
    socket_error error(socket_error::already_connected);
    demuxer_.operation_immediate(
        boost::bind(&accept_with_addr_op::do_upcall, handler, error));
    return;
  }

  accept_with_addr_op* op = new accept_with_addr_op(impl);
  op->demuxer_ = &demuxer_;
  op->peer_socket_ = &peer_socket;
  op->peer_address_ = &peer_address;
  op->handler_ = handler;
  op->context_ = &context;

  demuxer_.operation_started();

  selector_.start_read_op(*op);
}

void
select_provider::
do_socket_connector_destroy(
    socket_connector_service::impl_type& impl)
{
  if (impl != socket_connector_service::invalid_impl)
  {
    socket_connector_impl::socket_set sockets;
    impl->get_sockets(sockets);
    socket_connector_impl::socket_set::iterator i = sockets.begin();
    while (i != sockets.end())
      selector_.close_descriptor(*i++);

    delete impl;
    impl = socket_connector_service::invalid_impl;
  }
}

namespace
{
  struct connect_op : public select_op
  {
    demuxer* demuxer_;
    socket_connector_service::impl_type connector_impl_;
    socket_connector_service::peer_type* peer_socket_;
    generic_address peer_address_;
    socket_connector_service::connect_handler handler_;
    completion_context* context_;

    connect_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      socket_holder new_socket(descriptor());
      connector_impl_->remove_socket(new_socket.get());

      // Get the error code from the connect operation.
      int connect_error = 0;
      socket_len_type connect_error_len = sizeof(connect_error);
      if (socket_ops::getsockopt(new_socket.get(), SOL_SOCKET, SO_ERROR,
            &connect_error, &connect_error_len) == socket_error_retval)
      {
        socket_error error(socket_ops::get_error());
        demuxer_->operation_completed(boost::bind(&connect_op::do_upcall,
              handler_, error), *context_);
        delete this;
        return;
      }

      // If connection failed then post a completion with the error code.
      if (connect_error)
      {
        socket_error error(connect_error);
        demuxer_->operation_completed(boost::bind(&connect_op::do_upcall,
              handler_, error), *context_);
        delete this;
        return;
      }

      // Make the socket blocking again (the default).
      ioctl_arg_type non_blocking = 0;
      if (socket_ops::ioctl(new_socket.get(), FIONBIO, &non_blocking))
      {
        socket_error error(socket_ops::get_error());
        demuxer_->operation_completed(boost::bind(&connect_op::do_upcall,
              handler_, error), *context_);
        delete this;
        return;
      }

      // Post the result of the successful connection operation.
      peer_socket_->set_impl(new_socket.release());
      socket_error error(socket_error::success);
      demuxer_->operation_completed(boost::bind(&connect_op::do_upcall,
            handler_, error), *context_);
      delete this;
    }

    virtual void do_cancel()
    {
      connector_impl_->remove_socket(descriptor());
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&connect_op::do_upcall,
            handler_, error), *context_);
      delete this;
    }

    static void do_upcall(
        const socket_connector_service::connect_handler& handler,
        const socket_error& error)
    {
      handler(error);
    }
  };
} // namespace

void
select_provider::
do_socket_connector_async_connect(
    socket_connector_service::impl_type& impl,
    socket_connector_service::peer_type& peer_socket,
    const socket_address& peer_address, const connect_handler& handler,
    completion_context& context)
{
  if (peer_socket.impl() != invalid_socket)
  {
    socket_error error(socket_error::already_connected);
    demuxer_.operation_immediate(
        boost::bind(&connect_op::do_upcall, handler, error));
    return;
  }

  // Create a new socket for the connection. This will not be put into the
  // stream_socket object until the connection has beenestablished.
  socket_holder new_socket(socket_ops::socket(peer_address.family(),
        SOCK_STREAM, IPPROTO_TCP));
  if (new_socket.get() == invalid_socket)
  {
    socket_error error(socket_ops::get_error());
    demuxer_.operation_immediate(boost::bind(&connect_op::do_upcall, handler,
          error), context);
    return;
  }

  // Mark the socket as non-blocking so that the connection will take place
  // asynchronously.
  ioctl_arg_type non_blocking = 1;
  if (socket_ops::ioctl(new_socket.get(), FIONBIO, &non_blocking))
  {
    socket_error error(socket_ops::get_error());
    demuxer_.operation_immediate(boost::bind(&connect_op::do_upcall, handler,
          error), context);
    return;
  }

  // Start the connect operation.
  if (socket_ops::connect(new_socket.get(), peer_address.native_address(),
        peer_address.native_size()) == 0)
  {
    // The connect operation has finished successfully so we need to post the
    // completion immediately.
    peer_socket.set_impl(new_socket.release());
    socket_error error(socket_error::success);
    demuxer_.operation_immediate(boost::bind(&connect_op::do_upcall, handler,
          error), context);
  }
  else if (socket_ops::get_error() == socket_error::in_progress
      || socket_ops::get_error() == socket_error::would_block)
  {
    // The connection is happening in the background, and we need to wait until
    // the socket becomes writeable.

    connect_op* op = new connect_op(new_socket.get());
    op->demuxer_ = &demuxer_;
    op->connector_impl_ = impl;
    op->peer_socket_ = &peer_socket;
    op->peer_address_ = peer_address;
    op->handler_ = handler;
    op->context_ = &context;

    impl->add_socket(new_socket.get());
    new_socket.release();

    demuxer_.operation_started();

    selector_.start_write_op(*op);
  }
  else
  {
    // The connect operation has failed, so post completion immediately.
    socket_error error(socket_ops::get_error());
    demuxer_.operation_immediate(boost::bind(&connect_op::do_upcall, handler,
          error), context);
  }
}

void
select_provider::
do_stream_socket_create(
    stream_socket_service::impl_type& impl,
    stream_socket_service::impl_type new_impl)
{
  impl = new_impl;
}

void
select_provider::
do_stream_socket_destroy(
    stream_socket_service::impl_type& impl)
{
  if (impl != stream_socket_service::invalid_impl)
  {
    selector_.close_descriptor(impl);
    socket_ops::close(impl);
    impl = stream_socket_service::invalid_impl;
  }
}

namespace
{
  struct send_op : public select_op
  {
    demuxer* demuxer_;
    const void* data_;
    size_t length_;
    stream_socket_service::send_handler handler_;
    completion_context* context_;

    send_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      int bytes = socket_ops::send(descriptor(), data_, length_, 0);
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      demuxer_->operation_completed(boost::bind(&send_op::do_upcall,
            handler_, error, bytes), *context_);
      delete this;
    }

    virtual void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&send_op::do_upcall,
            handler_, error, 0), *context_);
      delete this;
    }

    static void do_upcall(const stream_socket_service::send_handler& handler,
        const socket_error& error, size_t bytes_transferred)
    {
      handler(error, bytes_transferred);
    }
  };
} // namespace

void
select_provider::
do_stream_socket_async_send(
    stream_socket_service::impl_type& impl,
    const void* data,
    size_t length,
    const send_handler& handler,
    completion_context& context)
{
  send_op* op = new send_op(impl);
  op->demuxer_ = &demuxer_;
  op->data_ = data;
  op->length_ = length;
  op->handler_ = handler;
  op->context_ = &context;

  demuxer_.operation_started();

  selector_.start_write_op(*op);
}

namespace
{
  struct send_n_op : public select_op
  {
    demuxer* demuxer_;
    selector* selector_;
    const void* data_;
    size_t length_;
    size_t already_sent_;
    stream_socket_service::send_n_handler handler_;
    completion_context* context_;

    send_n_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      int bytes = socket_ops::send(descriptor(),
          static_cast<const char*>(data_) + already_sent_,
          length_ - already_sent_, 0);
      size_t last_bytes = (bytes > 0 ? bytes : 0);
      size_t total_bytes = already_sent_ + last_bytes;
      if (last_bytes == 0 || total_bytes == length_)
      {
        socket_error error(bytes < 0
            ? socket_ops::get_error() : socket_error::success);
        demuxer_->operation_completed(boost::bind(&send_n_op::do_upcall,
              handler_, error, total_bytes, last_bytes), *context_);
        delete this;
      }
      else
      {
        already_sent_ = total_bytes;
        selector_->restart_write_op(*this);
      }
    }

    virtual void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&send_n_op::do_upcall,
            handler_, error, already_sent_, 0), *context_);
      delete this;
    }

    static void do_upcall(const stream_socket_service::send_n_handler& handler,
        const socket_error& error, size_t total_bytes_transferred,
        size_t last_bytes_transferred)
    {
      handler(error, total_bytes_transferred, last_bytes_transferred);
    }
  };
} // namespace

void
select_provider::
do_stream_socket_async_send_n(
    stream_socket_service::impl_type& impl,
    const void* data,
    size_t length,
    const send_n_handler& handler,
    completion_context& context)
{
  send_n_op* op = new send_n_op(impl);
  op->demuxer_ = &demuxer_;
  op->selector_ = &selector_;
  op->data_ = data;
  op->length_ = length;
  op->already_sent_ = 0;
  op->handler_ = handler;
  op->context_ = &context;

  demuxer_.operation_started();

  selector_.start_write_op(*op);
}

namespace
{
  struct recv_op : public select_op
  {
    demuxer* demuxer_;
    void* data_;
    size_t max_length_;
    stream_socket_service::recv_handler handler_;
    completion_context* context_;

    recv_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      int bytes = socket_ops::recv(descriptor(), data_, max_length_, 0);
      socket_error error(bytes < 0
          ? socket_ops::get_error() : socket_error::success);
      demuxer_->operation_completed(boost::bind(&recv_op::do_upcall,
            handler_, error, bytes), *context_);
      delete this;
    }

    virtual void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&recv_op::do_upcall,
            handler_, error, 0), *context_);
      delete this;
    }

    static void do_upcall(const stream_socket_service::recv_handler& handler,
        const socket_error& error, size_t bytes_transferred)
    {
      handler(error, bytes_transferred);
    }
  };
} // namespace

void
select_provider::
do_stream_socket_async_recv(
    stream_socket_service::impl_type& impl,
    void* data,
    size_t max_length,
    const recv_handler& handler,
    completion_context& context)
{
  recv_op* op = new recv_op(impl);
  op->demuxer_ = &demuxer_;
  op->data_ = data;
  op->max_length_ = max_length;
  op->handler_ = handler;
  op->context_ = &context;

  demuxer_.operation_started();

  selector_.start_read_op(*op);
}

namespace
{
  struct recv_n_op : public select_op
  {
    demuxer* demuxer_;
    selector* selector_;
    void* data_;
    size_t length_;
    size_t already_recvd_;
    stream_socket_service::recv_n_handler handler_;
    completion_context* context_;

    recv_n_op(int d) : select_op(d) {}

    virtual void do_operation()
    {
      int bytes = socket_ops::recv(descriptor(),
          static_cast<char*>(data_) + already_recvd_,
          length_ - already_recvd_, 0);
      size_t last_bytes = (bytes > 0 ? bytes : 0);
      size_t total_bytes = already_recvd_ + last_bytes;
      if (last_bytes == 0 || total_bytes == length_)
      {
        socket_error error(bytes < 0
            ? socket_ops::get_error() : socket_error::success);
        demuxer_->operation_completed(boost::bind(&recv_n_op::do_upcall,
              handler_, error, total_bytes, last_bytes), *context_);
        delete this;
      }
      else
      {
        already_recvd_ = total_bytes;
        selector_->restart_read_op(*this);
      }
    }

    virtual void do_cancel()
    {
      socket_error error(socket_error::operation_aborted);
      demuxer_->operation_completed(boost::bind(&recv_n_op::do_upcall,
            handler_, error, already_recvd_, 0), *context_);
      delete this;
    }

    static void do_upcall(const stream_socket_service::recv_n_handler& handler,
        const socket_error& error, size_t total_bytes_transferred,
        size_t last_bytes_transferred)
    {
      handler(error, total_bytes_transferred, last_bytes_transferred);
    }
  };
} // namespace

void
select_provider::
do_stream_socket_async_recv_n(
    stream_socket_service::impl_type& impl,
    void* data,
    size_t length,
    const recv_n_handler& handler,
    completion_context& context)
{
  recv_n_op* op = new recv_n_op(impl);
  op->demuxer_ = &demuxer_;
  op->selector_ = &selector_;
  op->data_ = data;
  op->length_ = length;
  op->already_recvd_ = 0;
  op->handler_ = handler;
  op->context_ = &context;

  demuxer_.operation_started();

  selector_.start_read_op(*op);
}

} // namespace detail
} // namespace asio
