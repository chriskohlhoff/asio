//
// detail/reactive_descriptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_REACTIVE_DESCRIPTOR_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_DESCRIPTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#include "asio/buffer.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/descriptor_ops.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/null_buffers_op.hpp"
#include "asio/detail/reactor.hpp"
#include "asio/detail/reactor_op.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class reactive_descriptor_service
{
public:
  // The native type of a descriptor.
  typedef int native_type;

  // The implementation type of the descriptor.
  class implementation_type
    : private asio::detail::noncopyable
  {
  public:
    // Default constructor.
    implementation_type()
      : descriptor_(-1),
        state_(0)
    {
    }

  private:
    // Only this service will have access to the internal values.
    friend class reactive_descriptor_service;

    // The native descriptor representation.
    int descriptor_;

    // The current state of the descriptor.
    descriptor_ops::state_type state_;

    // Per-descriptor data used by the reactor.
    reactor::per_descriptor_data reactor_data_;
  };

  // Constructor.
  ASIO_DECL reactive_descriptor_service(asio::io_service& io_service);

  // Destroy all user-defined handler objects owned by the service.
  ASIO_DECL void shutdown_service();

  // Construct a new descriptor implementation.
  ASIO_DECL void construct(implementation_type& impl);

  // Destroy a descriptor implementation.
  ASIO_DECL void destroy(implementation_type& impl);

  // Assign a native descriptor to a descriptor implementation.
  ASIO_DECL asio::error_code assign(implementation_type& impl,
      const native_type& native_descriptor, asio::error_code& ec);

  // Determine whether the descriptor is open.
  bool is_open(const implementation_type& impl) const
  {
    return impl.descriptor_ != -1;
  }

  // Destroy a descriptor implementation.
  ASIO_DECL asio::error_code close(implementation_type& impl,
      asio::error_code& ec);

  // Get the native descriptor representation.
  native_type native(const implementation_type& impl) const
  {
    return impl.descriptor_;
  }

  // Cancel all operations associated with the descriptor.
  ASIO_DECL asio::error_code cancel(implementation_type& impl,
      asio::error_code& ec);

  // Perform an IO control command on the descriptor.
  template <typename IO_Control_Command>
  asio::error_code io_control(implementation_type& impl,
      IO_Control_Command& command, asio::error_code& ec)
  {
    descriptor_ops::ioctl(impl.descriptor_, impl.state_,
        command.name(), static_cast<ioctl_arg_type*>(command.data()), ec);
    return ec;
  }

  // Write some data to the descriptor.
  template <typename ConstBufferSequence>
  size_t write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    return descriptor_ops::sync_write(impl.descriptor_, impl.state_,
        bufs.buffers(), bufs.count(), bufs.all_empty(), ec);
  }

  // Wait until data can be written without blocking.
  size_t write_some(implementation_type& impl,
      const null_buffers&, asio::error_code& ec)
  {
    // Wait for descriptor to become ready.
    descriptor_ops::poll_write(impl.descriptor_, ec);

    return 0;
  }

  template <typename ConstBufferSequence>
  class write_op_base : public reactor_op
  {
  public:
    write_op_base(int descriptor,
        const ConstBufferSequence& buffers, func_type complete_func)
      : reactor_op(&write_op_base::do_perform, complete_func),
        descriptor_(descriptor),
        buffers_(buffers)
    {
    }

    static bool do_perform(reactor_op* base)
    {
      write_op_base* o(static_cast<write_op_base*>(base));

      buffer_sequence_adapter<asio::const_buffer,
          ConstBufferSequence> bufs(o->buffers_);

      return descriptor_ops::non_blocking_write(o->descriptor_,
          bufs.buffers(), bufs.count(), o->ec_, o->bytes_transferred_);
    }

  private:
    int descriptor_;
    ConstBufferSequence buffers_;
  };

  template <typename ConstBufferSequence, typename Handler>
  class write_op : public write_op_base<ConstBufferSequence>
  {
  public:
    write_op(int descriptor,
        const ConstBufferSequence& buffers, Handler handler)
      : write_op_base<ConstBufferSequence>(
          descriptor, buffers, &write_op::do_complete),
        handler_(handler)
    {
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code /*ec*/, std::size_t /*bytes_transferred*/)
    {
      // Take ownership of the handler object.
      write_op* o(static_cast<write_op*>(base));
      typedef handler_alloc_traits<Handler, write_op> alloc_traits;
      handler_ptr<alloc_traits> ptr(o->handler_, o);

      // Make the upcall if required.
      if (owner)
      {
        // Make a copy of the handler so that the memory can be deallocated
        // before the upcall is made. Even if we're not about to make an
        // upcall, a sub-object of the handler may be the true owner of the
        // memory associated with the handler. Consequently, a local copy of
        // the handler is required to ensure that any owning sub-object remains
        // valid until after we have deallocated the memory here.
        detail::binder2<Handler, asio::error_code, std::size_t>
          handler(o->handler_, o->ec_, o->bytes_transferred_);
        ptr.reset();
        asio::detail::fenced_block b;
        asio_handler_invoke_helpers::invoke(handler, handler.handler_);
      }
    }

  private:
    Handler handler_;
  };

  // Start an asynchronous write. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef write_op<ConstBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, impl.descriptor_, buffers, handler);

    start_op(impl, reactor::write_op, ptr.get(), true,
        buffer_sequence_adapter<asio::const_buffer,
          ConstBufferSequence>::all_empty(buffers));
    ptr.release();
  }

  // Start an asynchronous wait until data can be written without blocking.
  template <typename Handler>
  void async_write_some(implementation_type& impl,
      const null_buffers&, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef null_buffers_op<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, handler);

    start_op(impl, reactor::write_op, ptr.get(), false, false);
    ptr.release();
  }

  // Read some data from the stream. Returns the number of bytes read.
  template <typename MutableBufferSequence>
  size_t read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, asio::error_code& ec)
  {
    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    return descriptor_ops::sync_read(impl.descriptor_, impl.state_,
        bufs.buffers(), bufs.count(), bufs.all_empty(), ec);
  }

  // Wait until data can be read without blocking.
  size_t read_some(implementation_type& impl,
      const null_buffers&, asio::error_code& ec)
  {
    // Wait for descriptor to become ready.
    descriptor_ops::poll_read(impl.descriptor_, ec);

    return 0;
  }

  template <typename MutableBufferSequence>
  class read_op_base : public reactor_op
  {
  public:
    read_op_base(int descriptor,
        const MutableBufferSequence& buffers, func_type complete_func)
      : reactor_op(&read_op_base::do_perform, complete_func),
        descriptor_(descriptor),
        buffers_(buffers)
    {
    }

    static bool do_perform(reactor_op* base)
    {
      read_op_base* o(static_cast<read_op_base*>(base));

      buffer_sequence_adapter<asio::mutable_buffer,
          MutableBufferSequence> bufs(o->buffers_);

      return descriptor_ops::non_blocking_read(o->descriptor_,
          bufs.buffers(), bufs.count(), o->ec_, o->bytes_transferred_);
    }

  private:
    int descriptor_;
    MutableBufferSequence buffers_;
  };

  template <typename MutableBufferSequence, typename Handler>
  class read_op : public read_op_base<MutableBufferSequence>
  {
  public:
    read_op(int descriptor,
        const MutableBufferSequence& buffers, Handler handler)
      : read_op_base<MutableBufferSequence>(
          descriptor, buffers, &read_op::do_complete),
        handler_(handler)
    {
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code /*ec*/, std::size_t /*bytes_transferred*/)
    {
      // Take ownership of the handler object.
      read_op* o(static_cast<read_op*>(base));
      typedef handler_alloc_traits<Handler, read_op> alloc_traits;
      handler_ptr<alloc_traits> ptr(o->handler_, o);

      // Make the upcall if required.
      if (owner)
      {
        // Make a copy of the handler so that the memory can be deallocated
        // before the upcall is made. Even if we're not about to make an
        // upcall, a sub-object of the handler may be the true owner of the
        // memory associated with the handler. Consequently, a local copy of
        // the handler is required to ensure that any owning sub-object remains
        // valid until after we have deallocated the memory here.
        detail::binder2<Handler, asio::error_code, std::size_t>
          handler(o->handler_, o->ec_, o->bytes_transferred_);
        ptr.reset();
        asio::detail::fenced_block b;
        asio_handler_invoke_helpers::invoke(handler, handler.handler_);
      }
    }

  private:
    Handler handler_;
  };

  // Start an asynchronous read. The buffer for the data being read must be
  // valid for the lifetime of the asynchronous operation.
  template <typename MutableBufferSequence, typename Handler>
  void async_read_some(implementation_type& impl,
      const MutableBufferSequence& buffers, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef read_op<MutableBufferSequence, Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr,
        impl.descriptor_, buffers, handler);

    start_op(impl, reactor::read_op, ptr.get(), true,
        buffer_sequence_adapter<asio::mutable_buffer,
          MutableBufferSequence>::all_empty(buffers));
    ptr.release();
  }

  // Wait until data can be read without blocking.
  template <typename Handler>
  void async_read_some(implementation_type& impl,
      const null_buffers&, Handler handler)
  {
    // Allocate and construct an operation to wrap the handler.
    typedef null_buffers_op<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, handler);

    start_op(impl, reactor::read_op, ptr.get(), false, false);
    ptr.release();
  }

private:
  // Start the asynchronous operation.
  ASIO_DECL void start_op(implementation_type& impl, int op_type,
      reactor_op* op, bool non_blocking, bool noop);

  // The io_service implementation used to post completions.
  io_service_impl& io_service_impl_;

  // The selector that performs event demultiplexing for the service.
  reactor& reactor_;
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/reactive_descriptor_service.ipp"
#endif // defined(ASIO_HEADER_ONLY)

#endif // ASIO_DETAIL_REACTIVE_DESCRIPTOR_SERVICE_HPP
