//
// win_iocp_overlapped_ptr.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2010 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_DETAIL_WIN_IOCP_OVERLAPPED_PTR_HPP
#define ASIO_DETAIL_WIN_IOCP_OVERLAPPED_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/push_options.hpp"

#include "asio/detail/win_iocp_io_service_fwd.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/fenced_block.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/win_iocp_io_service.hpp"
#include "asio/detail/win_iocp_operation.hpp"

namespace asio {
namespace detail {

// Wraps a handler to create an OVERLAPPED object for use with overlapped I/O.
class win_iocp_overlapped_ptr
  : private noncopyable
{
public:
  // Construct an empty win_iocp_overlapped_ptr.
  win_iocp_overlapped_ptr()
    : ptr_(0),
      iocp_service_(0)
  {
  }

  // Construct an win_iocp_overlapped_ptr to contain the specified handler.
  template <typename Handler>
  explicit win_iocp_overlapped_ptr(
      asio::io_service& io_service, Handler handler)
    : ptr_(0),
      iocp_service_(0)
  {
    this->reset(io_service, handler);
  }

  // Destructor automatically frees the OVERLAPPED object unless released.
  ~win_iocp_overlapped_ptr()
  {
    reset();
  }

  // Reset to empty.
  void reset()
  {
    if (ptr_)
    {
      ptr_->destroy();
      ptr_ = 0;
      iocp_service_->work_finished();
      iocp_service_ = 0;
    }
  }

  // Reset to contain the specified handler, freeing any current OVERLAPPED
  // object.
  template <typename Handler>
  void reset(asio::io_service& io_service, Handler handler)
  {
    typedef overlapped_op<Handler> value_type;
    typedef handler_alloc_traits<Handler, value_type> alloc_traits;
    raw_handler_ptr<alloc_traits> raw_ptr(handler);
    handler_ptr<alloc_traits> ptr(raw_ptr, handler);
    io_service.impl_.work_started();
    reset();
    ptr_ = ptr.release();
    iocp_service_ = &io_service.impl_;
  }

  // Get the contained OVERLAPPED object.
  OVERLAPPED* get()
  {
    return ptr_;
  }

  // Get the contained OVERLAPPED object.
  const OVERLAPPED* get() const
  {
    return ptr_;
  }

  // Release ownership of the OVERLAPPED object.
  OVERLAPPED* release()
  {
    if (ptr_)
      iocp_service_->on_pending(ptr_);

    OVERLAPPED* tmp = ptr_;
    ptr_ = 0;
    iocp_service_ = 0;
    return tmp;
  }

  // Post completion notification for overlapped operation. Releases ownership.
  void complete(const asio::error_code& ec,
      std::size_t bytes_transferred)
  {
    if (ptr_)
    {
      iocp_service_->on_completion(ptr_, ec,
          static_cast<DWORD>(bytes_transferred));
      ptr_ = 0;
      iocp_service_ = 0;
    }
  }

private:
  template <typename Handler>
  struct overlapped_op : public win_iocp_operation
  {
    overlapped_op(Handler handler)
      : win_iocp_operation(&overlapped_op::do_complete),
        handler_(handler)
    {
    }

    static void do_complete(io_service_impl* owner, operation* base,
        asio::error_code ec, std::size_t bytes_transferred)
    {
      // Take ownership of the operation object.
      overlapped_op* o(static_cast<overlapped_op*>(base));
      typedef handler_alloc_traits<Handler, overlapped_op> alloc_traits;
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
          handler(o->handler_, ec, bytes_transferred);
        ptr.reset();
        asio::detail::fenced_block b;
        asio_handler_invoke_helpers::invoke(handler, handler);
      }
    }

  private:
    Handler handler_;
  };

  win_iocp_operation* ptr_;
  win_iocp_io_service* iocp_service_;
};

} // namespace detail
} // namespace asio

#endif // defined(ASIO_HAS_IOCP)

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_WIN_IOCP_OVERLAPPED_PTR_HPP
