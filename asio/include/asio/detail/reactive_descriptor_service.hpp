//
// reactive_descriptor_service.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#include "asio/detail/push_options.hpp"

#include "asio/buffer.hpp"
#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/descriptor_ops.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/null_buffers_op.hpp"
#include "asio/detail/reactor.hpp"
#include "asio/detail/reactor_op.hpp"

#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

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
        flags_(0)
    {
    }

  private:
    // Only this service will have access to the internal values.
    friend class reactive_descriptor_service;

    // The native descriptor representation.
    int descriptor_;

    enum
    {
      // The user wants a non-blocking descriptor.
      user_set_non_blocking = 1,

      // The descriptor has been set non-blocking.
      internal_non_blocking = 2,

      // Helper "flag" used to determine whether the descriptor is non-blocking.
      non_blocking = user_set_non_blocking | internal_non_blocking
    };

    // Flags indicating the current state of the descriptor.
    unsigned char flags_;

    // Per-descriptor data used by the reactor.
    reactor::per_descriptor_data reactor_data_;
  };

  // The maximum number of buffers to support in a single operation.
  enum { max_buffers = 64 < max_iov_len ? 64 : max_iov_len };

  // Constructor.
  reactive_descriptor_service(asio::io_service& io_service)
    : io_service_impl_(asio::use_service<io_service_impl>(io_service)),
      reactor_(asio::use_service<reactor>(io_service))
  {
    reactor_.init_task();
  }

  // Destroy all user-defined handler objects owned by the service.
  void shutdown_service()
  {
  }

  // Construct a new descriptor implementation.
  void construct(implementation_type& impl)
  {
    impl.descriptor_ = -1;
    impl.flags_ = 0;
  }

  // Destroy a descriptor implementation.
  void destroy(implementation_type& impl)
  {
    if (impl.descriptor_ != -1)
    {
      reactor_.close_descriptor(impl.descriptor_, impl.reactor_data_);

      if (impl.flags_ & implementation_type::internal_non_blocking)
      {
        ioctl_arg_type non_blocking = 0;
        asio::error_code ignored_ec;
        descriptor_ops::ioctl(impl.descriptor_,
            FIONBIO, &non_blocking, ignored_ec);
        impl.flags_ &= ~implementation_type::internal_non_blocking;
      }

      asio::error_code ignored_ec;
      descriptor_ops::close(impl.descriptor_, ignored_ec);

      impl.descriptor_ = -1;
    }
  }

  // Assign a native descriptor to a descriptor implementation.
  asio::error_code assign(implementation_type& impl,
      const native_type& native_descriptor, asio::error_code& ec)
  {
    if (is_open(impl))
    {
      ec = asio::error::already_open;
      return ec;
    }

    if (int err = reactor_.register_descriptor(
          native_descriptor, impl.reactor_data_))
    {
      ec = asio::error_code(err,
          asio::error::get_system_category());
      return ec;
    }

    impl.descriptor_ = native_descriptor;
    impl.flags_ = 0;
    ec = asio::error_code();
    return ec;
  }

  // Determine whether the descriptor is open.
  bool is_open(const implementation_type& impl) const
  {
    return impl.descriptor_ != -1;
  }

  // Destroy a descriptor implementation.
  asio::error_code close(implementation_type& impl,
      asio::error_code& ec)
  {
    if (is_open(impl))
    {
      reactor_.close_descriptor(impl.descriptor_, impl.reactor_data_);

      if (impl.flags_ & implementation_type::internal_non_blocking)
      {
        ioctl_arg_type non_blocking = 0;
        asio::error_code ignored_ec;
        descriptor_ops::ioctl(impl.descriptor_,
            FIONBIO, &non_blocking, ignored_ec);
        impl.flags_ &= ~implementation_type::internal_non_blocking;
      }

      if (descriptor_ops::close(impl.descriptor_, ec) == -1)
        return ec;

      impl.descriptor_ = -1;
    }

    ec = asio::error_code();
    return ec;
  }

  // Get the native descriptor representation.
  native_type native(const implementation_type& impl) const
  {
    return impl.descriptor_;
  }

  // Cancel all operations associated with the descriptor.
  asio::error_code cancel(implementation_type& impl,
      asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return ec;
    }

    reactor_.cancel_ops(impl.descriptor_, impl.reactor_data_);
    ec = asio::error_code();
    return ec;
  }

  // Perform an IO control command on the descriptor.
  template <typename IO_Control_Command>
  asio::error_code io_control(implementation_type& impl,
      IO_Control_Command& command, asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return ec;
    }

    descriptor_ops::ioctl(impl.descriptor_, command.name(),
        static_cast<ioctl_arg_type*>(command.data()), ec);

    // When updating the non-blocking mode we always perform the ioctl syscall,
    // even if the flags would otherwise indicate that the descriptor is
    // already in the correct state. This ensures that the underlying
    // descriptor is put into the state that has been requested by the user. If
    // the ioctl syscall was successful then we need to update the flags to
    // match.
    if (!ec && command.name() == static_cast<int>(FIONBIO))
    {
      if (*static_cast<ioctl_arg_type*>(command.data()))
      {
        impl.flags_ |= implementation_type::user_set_non_blocking;
      }
      else
      {
        // Clearing the non-blocking mode always overrides any internally-set
        // non-blocking flag. Any subsequent asynchronous operations will need
        // to re-enable non-blocking I/O.
        impl.flags_ &= ~(implementation_type::user_set_non_blocking
            | implementation_type::internal_non_blocking);
      }
    }

    return ec;
  }

  // Write some data to the descriptor.
  template <typename ConstBufferSequence>
  size_t write_some(implementation_type& impl,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return 0;
    }

    buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence> bufs(buffers);

    // A request to read_some 0 bytes on a stream is a no-op.
    if (bufs.all_empty())
    {
      ec = asio::error_code();
      return 0;
    }

    // Send the data.
    for (;;)
    {
      // Try to complete the operation without blocking.
      int bytes_sent = descriptor_ops::gather_write(
          impl.descriptor_, bufs.buffers(), bufs.count(), ec);

      // Check if operation succeeded.
      if (bytes_sent >= 0)
        return bytes_sent;

      // Operation failed.
      if ((impl.flags_ & implementation_type::user_set_non_blocking)
          || (ec != asio::error::would_block
            && ec != asio::error::try_again))
        return 0;

      // Wait for descriptor to become ready.
      if (descriptor_ops::poll_write(impl.descriptor_, ec) < 0)
        return 0;
    }
  }

  // Wait until data can be written without blocking.
  size_t write_some(implementation_type& impl,
      const null_buffers&, asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return 0;
    }

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

      for (;;)
      {
        // Write the data.
        asio::error_code ec;
        int bytes = descriptor_ops::gather_write(
            o->descriptor_, bufs.buffers(), bufs.count(), ec);

        // Retry operation if interrupted by signal.
        if (ec == asio::error::interrupted)
          continue;

        // Check if we need to run the operation again.
        if (ec == asio::error::would_block
            || ec == asio::error::try_again)
          return false;

        o->ec_ = ec;
        o->bytes_transferred_ = (bytes < 0 ? 0 : bytes);
        return true;
      }
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
        asio_handler_invoke_helpers::invoke(handler, handler);
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
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return 0;
    }

    buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence> bufs(buffers);

    // A request to read_some 0 bytes on a stream is a no-op.
    if (bufs.all_empty())
    {
      ec = asio::error_code();
      return 0;
    }

    // Read some data.
    for (;;)
    {
      // Try to complete the operation without blocking.
      int bytes_read = descriptor_ops::scatter_read(
          impl.descriptor_, bufs.buffers(), bufs.count(), ec);

      // Check if operation succeeded.
      if (bytes_read > 0)
        return bytes_read;

      // Check for EOF.
      if (bytes_read == 0)
      {
        ec = asio::error::eof;
        return 0;
      }

      // Operation failed.
      if ((impl.flags_ & implementation_type::user_set_non_blocking)
          || (ec != asio::error::would_block
            && ec != asio::error::try_again))
        return 0;

      // Wait for descriptor to become ready.
      if (descriptor_ops::poll_read(impl.descriptor_, ec) < 0)
        return 0;
    }
  }

  // Wait until data can be read without blocking.
  size_t read_some(implementation_type& impl,
      const null_buffers&, asio::error_code& ec)
  {
    if (!is_open(impl))
    {
      ec = asio::error::bad_descriptor;
      return 0;
    }

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

      for (;;)
      {
        // Read some data.
        asio::error_code ec;
        int bytes = descriptor_ops::scatter_read(
            o->descriptor_, bufs.buffers(), bufs.count(), ec);
        if (bytes == 0)
          ec = asio::error::eof;

        // Retry operation if interrupted by signal.
        if (ec == asio::error::interrupted)
          continue;

        // Check if we need to run the operation again.
        if (ec == asio::error::would_block
            || ec == asio::error::try_again)
          return false;

        o->ec_ = ec;
        o->bytes_transferred_ = (bytes < 0 ? 0 : bytes);
        return true;
      }
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
        asio_handler_invoke_helpers::invoke(handler, handler);
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
  void start_op(implementation_type& impl, int op_type,
      reactor_op* op, bool non_blocking, bool noop)
  {
    if (!noop)
    {
      if (is_open(impl))
      {
        if (is_non_blocking(impl) || set_non_blocking(impl, op->ec_))
        {
          reactor_.start_op(op_type, impl.descriptor_,
              impl.reactor_data_, op, non_blocking);
          return;
        }
      }
      else
        op->ec_ = asio::error::bad_descriptor;
    }

    io_service_impl_.post_immediate_completion(op);
  }

  // Determine whether the descriptor has been set non-blocking.
  bool is_non_blocking(implementation_type& impl) const
  {
    return (impl.flags_ & implementation_type::non_blocking);
  }

  // Set the internal non-blocking flag.
  bool set_non_blocking(implementation_type& impl,
      asio::error_code& ec)
  {
    ioctl_arg_type non_blocking = 1;
    if (descriptor_ops::ioctl(impl.descriptor_, FIONBIO, &non_blocking, ec))
      return false;
    impl.flags_ |= implementation_type::internal_non_blocking;
    return true;
  }

  // The io_service implementation used to post completions.
  io_service_impl& io_service_impl_;

  // The selector that performs event demultiplexing for the service.
  reactor& reactor_;
};

} // namespace detail
} // namespace asio

#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_REACTIVE_DESCRIPTOR_SERVICE_HPP
