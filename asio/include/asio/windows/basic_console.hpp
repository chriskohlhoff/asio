//
// windows/basic_object_handle.hpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2023 Klemens David Morgenstern (klemens dot morgenstern at gmx dot net)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_WINDOWS_BASIC_CONSOLE_HPP
#define ASIO_WINDOWS_BASIC_CONSOLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE) \
  || defined(GENERATING_DOCUMENTATION)

#include "asio/any_io_executor.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/io_object_impl.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/windows/basic_object_handle.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/compose.hpp"
#include "asio/append.hpp"
#include "asio/post.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"

#if defined(ASIO_HAS_MOVE)
# include <utility>

#endif // defined(ASIO_HAS_MOVE)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace windows {


/// Provides stream-oriented handle functionality.
/**
 * The windows::basic_console class provides asynchronous and blocking
 * stream-oriented handle functionality.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe.
 *
 * @par Concepts:
 * AsyncReadStream, AsyncWriteStream, Stream, SyncReadStream, SyncWriteStream.
 */
template <typename Executor = any_io_executor>
class basic_console
    : public basic_object_handle<Executor>
{
public:
  /// The type of the executor associated with the object.
  typedef Executor executor_type;

  /// Rebinds the handle type to another executor.
  template <typename Executor1>
  struct rebind_executor
  {
    /// The handle type when rebound to the specified executor.
    typedef basic_console<Executor1> other;
  };

  /// The native representation of a handle.
#if defined(GENERATING_DOCUMENTATION)
  typedef implementation_defined native_handle_type;
#else
  typedef basic_object_handle<Executor>::native_handle_type
      native_handle_type;
#endif

  /// Construct a stream handle without opening it.
  /**
   * This constructor creates a stream handle without opening it.
   *
   * @param ex The I/O executor that the stream handle will use, by default, to
   * dispatch handlers for any asynchronous operations performed on the stream
   * handle.
   */
  explicit basic_console(const executor_type& ex)
      : basic_object_handle<Executor>(ex)
  {
  }

  /// Construct a stream handle without opening it.
  /**
   * This constructor creates a stream handle without opening it. The handle
   * needs to be opened or assigned before data can be sent or received on it.
   *
   * @param context An execution context which provides the I/O executor that
   * the stream handle will use, by default, to dispatch handlers for any
   * asynchronous operations performed on the stream handle.
   */
  template <typename ExecutionContext>
  explicit basic_console(ExecutionContext& context,
                               typename constraint<
                                   is_convertible<ExecutionContext&, execution_context&>::value,
                                   defaulted_constraint
                               >::type = defaulted_constraint())
      : basic_object_handle<Executor>(context)
  {
  }

  /// Construct a stream handle on an existing native handle.
  /**
   * This constructor creates a stream handle object to hold an existing native
   * handle.
   *
   * @param ex The I/O executor that the stream handle will use, by default, to
   * dispatch handlers for any asynchronous operations performed on the stream
   * handle.
   *
   * @param handle The new underlying handle implementation.
   *
   * @throws asio::system_error Thrown on failure.
   */
  basic_console(const executor_type& ex, const native_handle_type& handle)
      : basic_object_handle<Executor>(ex, handle)
  {
  }

  /// Construct a stream handle on an existing native handle.
  /**
   * This constructor creates a stream handle object to hold an existing native
   * handle.
   *
   * @param context An execution context which provides the I/O executor that
   * the stream handle will use, by default, to dispatch handlers for any
   * asynchronous operations performed on the stream handle.
   *
   * @param handle The new underlying handle implementation.
   *
   * @throws asio::system_error Thrown on failure.
   */
  template <typename ExecutionContext>
  basic_console(ExecutionContext& context,
                      const native_handle_type& handle,
                      typename constraint<
                          is_convertible<ExecutionContext&, execution_context&>::value
                      >::type = 0)
      : basic_object_handle<Executor>(context, handle)
  {
  }

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Move-construct a stream handle from another.
  /**
   * This constructor moves a stream handle from one object to another.
   *
   * @param other The other stream handle object from which the move
   * will occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c basic_console(const executor_type&)
   * constructor.
   */
  basic_console(basic_console&& other)
      : basic_object_handle<Executor>(std::move(other))
  {
  }

  /// Move-assign a stream handle from another.
  /**
   * This assignment operator moves a stream handle from one object to
   * another.
   *
   * @param other The other stream handle object from which the move will occur.
   *
   * @note Following the move, the moved-from object is in the same state as if
   * constructed using the @c basic_console(const executor_type&)
   * constructor.
   */
  basic_console& operator=(basic_console&& other)
  {
    basic_object_handle<Executor>::operator=(std::move(other));
    return *this;
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Write some data to the handle.
  /**
   * This function is used to write data to the stream handle. The function call
   * will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the handle.
   *
   * @returns The number of bytes written.
   *
   * @throws asio::system_error Thrown on failure. An error code of
   * asio::error::eof indicates that the connection was closed by the
   * peer.
   *
   * @note The write_some operation may not transmit all of the data to the
   * peer. Consider using the @ref write function if you need to ensure that
   * all data is written before the blocking operation completes.
   *
   * @par Example
   * To write a single data buffer use the @ref buffer function as follows:
   * @code
   * handle.write_some(asio::buffer(data, size));
   * @endcode
   * See the @ref buffer documentation for information on writing multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename ConstBufferSequence>
  std::size_t write_some(const ConstBufferSequence& buffers)
  {
    asio::error_code ec;
    std::size_t s = write_some(buffers, ec);
    asio::detail::throw_error(ec, "write_some");
    return s;
  }

  /// Write some data to the handle.
  /**
   * This function is used to write data to the stream handle. The function call
   * will block until one or more bytes of the data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers One or more data buffers to be written to the handle.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns The number of bytes written. Returns 0 if an error occurred.
   *
   * @note The write_some operation may not transmit all of the data to the
   * peer. Consider using the @ref write function if you need to ensure that
   * all data is written before the blocking operation completes.
   */
  template <typename ConstBufferSequence>
  std::size_t write_some(const ConstBufferSequence& buffers,
                         asio::error_code& ec)
  {
    this->wait(ec);
    if (ec)
      return 0u;
    CONSOLE_SCREEN_BUFFER_INFO info;

    if (!GetConsoleScreenBufferInfo(this->native_handle(), &info))
    {
      ec.assign(static_cast<int>(::GetLastError()), asio::system_category());
      return 0u;
    }

    const auto max_size = static_cast<std::size_t>(info.dwSize.X * info.dwSize.Y);
    auto rbuf = detail::buffer_sequence_adapter<asio::const_buffer, ConstBufferSequence>::first(buffers);
    auto buffer = asio::const_buffer(rbuf.data(), (std::min)(max_size, rbuf.size()));
    DWORD written = 0u;
    if(!WriteFile(this->native_handle(), buffer.data(), static_cast<DWORD>(buffer.size()), &written, nullptr))
      ec.assign(static_cast<int>(::GetLastError()), asio::system_category());

    return static_cast<std::size_t>(written);
  }

  /// Start an asynchronous write.
  /**
   * This function is used to asynchronously write data to the stream handle.
   * The function call always returns immediately.
   *
   * @param buffers One or more data buffers to be written to the handle.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param handler The handler to be called when the write operation completes.
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   std::size_t bytes_transferred           // Number of bytes written.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. On
   * immediate completion, invocation of the handler will be performed in a
   * manner equivalent to using asio::post().
   *
   * @note The write operation may not transmit all of the data to the peer.
   * Consider using the @ref async_write function if you need to ensure that all
   * data is written before the asynchronous operation completes.
   *
   * @par Example
   * To write a single data buffer use the @ref buffer function as follows:
   * @code
   * handle.async_write_some(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on writing multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   *
   * @par Per-Operation Cancellation
   * This asynchronous operation supports cancellation for the following
   * asio::cancellation_type values:
   *
   * @li @c cancellation_type::terminal
   *
   * @li @c cancellation_type::partial
   *
   * @li @c cancellation_type::total
   */
  template <typename ConstBufferSequence,
      ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
                                    std::size_t)) WriteHandler
      ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
  ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
                               void (asio::error_code, std::size_t))
  async_write_some(const ConstBufferSequence& buffers,
                   ASIO_MOVE_ARG(WriteHandler) handler
                   ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
  {
    return asio::async_compose<
        WriteHandler,
        void (asio::error_code, std::size_t)>(
        initiate_async_write_some{this,
                                 detail::buffer_sequence_adapter<asio::const_buffer,
                                     ConstBufferSequence>::first(buffers)}, handler,
        this->get_executor());
  }

  /// Read some data from the handle.
  /**
   * This function is used to read data from the stream handle. The function
   * call will block until one or more bytes of data has been read successfully,
   * or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
   *
   * @returns The number of bytes read.
   *
   * @throws asio::system_error Thrown on failure. An error code of
   * asio::error::eof indicates that the connection was closed by the
   * peer.
   *
   * @note The read_some operation may not read all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that
   * the requested amount of data is read before the blocking operation
   * completes.
   *
   * @par Example
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code
   * handle.read_some(asio::buffer(data, size));
   * @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename MutableBufferSequence>
  std::size_t read_some(const MutableBufferSequence& buffers)
  {
    asio::error_code ec;
    std::size_t s = read_some(buffers, ec);
    asio::detail::throw_error(ec, "read_some");
    return s;
  }

  /// Read some data from the handle.
  /**
   * This function is used to read data from the stream handle. The function
   * call will block until one or more bytes of data has been read successfully,
   * or until an error occurs.
   *
   * @param buffers One or more buffers into which the data will be read.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns The number of bytes read. Returns 0 if an error occurred.
   *
   * @note The read_some operation may not read all of the requested number of
   * bytes. Consider using the @ref read function if you need to ensure that
   * the requested amount of data is read before the blocking operation
   * completes.
   */
  template <typename MutableBufferSequence>
  std::size_t read_some(const MutableBufferSequence& buffers,
                        asio::error_code& ec)
  {
    if (!SetConsoleMode(this->native_handle(), ENABLE_LINE_INPUT
                                               | ENABLE_ECHO_INPUT
                                               | ENABLE_VIRTUAL_TERMINAL_INPUT
                                               | ENABLE_PROCESSED_INPUT))
      ec.assign(static_cast<int>(::GetLastError()), asio::system_category());

    while (!ec)
    {
      this->wait(ec);
      if (ec)
        break;

      INPUT_RECORD buf[128];
      DWORD res = 0u;
      if (!PeekConsoleInputW(this->native_handle(), buf, 128u, &res))
      {
        ec.assign(static_cast<int>(::GetLastError()), asio::system_category());
        break;
      }

      auto inp = std::count_if(buf, buf + res,
                               [](const INPUT_RECORD & ke)
                               {
                                 return ke.EventType == KEY_EVENT
                                        && ke.Event.KeyEvent.uChar.AsciiChar != 0
                                        && ke.Event.KeyEvent.bKeyDown;
                               });
      if (inp == 0u)
      {
        if (!FlushConsoleInputBuffer(this->native_handle()))
        {
          ec.assign(static_cast<int>(::GetLastError()), asio::system_category());
          break;
        }
        continue;
      }
      mutable_buffer buffer =
          detail::buffer_sequence_adapter<asio::mutable_buffer, MutableBufferSequence>::first(buffers);

      if (!ReadFile(this->native_handle(), buffer.data(), static_cast<DWORD>(buffer.size()), &res, nullptr))
      {
        ec.assign(static_cast<int>(::GetLastError()), asio::system_category());
        break;
      }
      return static_cast<std::size_t>(res);
    }
    return 0u;
  }

  /// Start an asynchronous read.
  /**
   * This function is used to asynchronously read data from the stream handle.
   * The function call always returns immediately.
   *
   * @param buffers One or more buffers into which the data will be read.
   * Although the buffers object may be copied as necessary, ownership of the
   * underlying memory blocks is retained by the caller, which must guarantee
   * that they remain valid until the handler is called.
   *
   * @param handler The handler to be called when the read operation completes.
   * Copies will be made of the handler as required. The function signature of
   * the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   std::size_t bytes_transferred           // Number of bytes read.
   * ); @endcode
   * Regardless of whether the asynchronous operation completes immediately or
   * not, the handler will not be invoked from within this function. On
   * immediate completion, invocation of the handler will be performed in a
   * manner equivalent to using asio::post().
   *
   * @note The read operation may not read all of the requested number of bytes.
   * Consider using the @ref async_read function if you need to ensure that the
   * requested amount of data is read before the asynchronous operation
   * completes.
   *
   * @par Example
   * To read into a single data buffer use the @ref buffer function as follows:
   * @code
   * handle.async_read_some(asio::buffer(data, size), handler);
   * @endcode
   * See the @ref buffer documentation for information on reading into multiple
   * buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   *
   * @par Per-Operation Cancellation
   * This asynchronous operation supports cancellation for the following
   * asio::cancellation_type values:
   *
   * @li @c cancellation_type::terminal
   *
   * @li @c cancellation_type::partial
   *
   * @li @c cancellation_type::total
   */
  template <typename MutableBufferSequence,
      ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
                                    std::size_t)) ReadHandler
      ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
  ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
                               void (asio::error_code, std::size_t))
  async_read_some(const MutableBufferSequence& buffers,
                  ASIO_MOVE_ARG(ReadHandler) handler
                  ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
  {
    return asio::async_compose<
        ReadHandler,
        void (asio::error_code, std::size_t)>(
        initiate_async_read_some{this,
                                 detail::buffer_sequence_adapter<asio::mutable_buffer,
                                 MutableBufferSequence>::first(buffers)}, handler,
                                 this->get_executor());
  }

private:
  struct initiate_async_write_some
  {
    basic_console* self_;
    asio::const_buffer rbuf;

    template <typename Self>
    void operator()(Self && self) const
    {
      self_->async_wait(std::move(self));
    }

    template <typename Self>
    void operator()(Self && self, asio::error_code ec) const
    {
      if (ec)
        return self.complete(ec, 0u);
      CONSOLE_SCREEN_BUFFER_INFO info;

      if (!GetConsoleScreenBufferInfo(self_->native_handle(), &info))
      {
        ec.assign(static_cast<int>(::GetLastError()), asio::system_category());
        return self.complete(ec, 0u);
      }

      const auto max_size = static_cast<std::size_t>(info.dwSize.X * info.dwSize.Y);
      auto buffer = asio::const_buffer(rbuf.data(), (std::min)(max_size, rbuf.size()));
      DWORD written = 0u;
      if(!WriteFile(self_->native_handle(), buffer.data(), static_cast<DWORD>(buffer.size()), &written, nullptr))
        ec.assign(static_cast<int>(::GetLastError()), asio::system_category());

      self.complete(ec, static_cast<std::size_t>(written));
    }

  private:
  };

  struct initiate_async_read_some
  {
    basic_console* self_;
    asio::mutable_buffer buffer;

    template <typename Self>
    void operator()(Self && self) const
    {
      if (!SetConsoleMode(self_->native_handle(), ENABLE_LINE_INPUT
                                                 | ENABLE_ECHO_INPUT
                                                 | ENABLE_VIRTUAL_TERMINAL_INPUT
                                                 | ENABLE_PROCESSED_INPUT))
      {
        asio::post(self_->get_executor(),
                   asio::append(std::move(self),
                                asio::error_code(::GetLastError(), asio::system_category())));
      }
      else
        self_->async_wait(std::move(self));
    }

    template <typename Self>
    void operator()(Self && self, asio::error_code ec) const
    {


      if (ec)
        return self.complete(ec, 0u);

      INPUT_RECORD buf[128];
      DWORD res = 0u;
      if (!PeekConsoleInputW(self_->native_handle(), buf, 128u, &res))
        return self.complete(
            asio::error_code(static_cast<int>(::GetLastError()), asio::system_category()),
            0u);

      auto inp = std::count_if(buf, buf + res,
                               [](const INPUT_RECORD & ke)
                               {
                                 return ke.EventType == KEY_EVENT
                                     && ke.Event.KeyEvent.uChar.AsciiChar != 0
                                     && ke.Event.KeyEvent.bKeyDown;
                               });
      if (inp == 0u)
      {
        if (!FlushConsoleInputBuffer(self_->native_handle()))
          return self.complete(
              asio::error_code(static_cast<int>(::GetLastError()), asio::system_category()),
              0u);
        return self_->async_wait(std::move(self));
      }

      if (!ReadFile(self_->native_handle(), buffer.data(),
                    static_cast<DWORD>(buffer.size()), &res, nullptr))
      {
        ec.assign(static_cast<int>(::GetLastError()), asio::system_category());
        return self.complete(
            asio::error_code(static_cast<int>(::GetLastError()), asio::system_category()),
            0u);;
      }
      self.complete(asio::error_code{}, static_cast<std::size_t>(res));

    }
  };
};


} // namespace windows
} // namespace asio

#include "asio/detail/pop_options.hpp"


#endif

#endif //ASIO_WINDOWS_BASIC_CONSOLE_HPP
