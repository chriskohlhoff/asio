//
// ssl/dtls.hpp
// ~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2016
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef ASIO_SSL_DTLS_HPP
#define ASIO_SSL_DTLS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#include "asio/async_result.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/ssl/context.hpp"
#include "asio/ssl/detail/buffered_handshake_op.hpp"
#include "asio/ssl/detail/handshake_op.hpp"
#include "asio/ssl/detail/io.hpp"
#include "asio/ssl/detail/read_op.hpp"
#include "asio/ssl/detail/shutdown_op.hpp"
#include "asio/ssl/detail/stream_core.hpp"
#include "asio/ssl/detail/write_op.hpp"
#include "asio/ssl/ssl_base.hpp"
#include "asio/ssl/stream_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {

/// Provides Datagram-oriented functionality using SSL.
/**
 * The dtls class template provides asynchronous and blocking stream-oriented
 * functionality using SSL.
 *
 * @par Thread Safety
 * @e Distinct @e objects: Safe.@n
 * @e Shared @e objects: Unsafe. The application must also ensure that all
 * asynchronous operations are performed within the same implicit or explicit
 * strand.
 *
 * @par Example
 * To use the SSL dtls template with an ip::udp::socket, you would write:
 * @code
 * asio::io_context io_context;
 * asio::ssl::context ctx(asio::ssl::context::dtlsv12);
 * asio::ssl::stream<asio:ip::udp::socket> sock(io_context, ctx);
 * @endcode
 */
template <typename datagram_socket>
class dtls :
  public ssl_base<datagram_socket>,
  private noncopyable
{
public:
  /// The native handle type of the SSL stream.
  typedef SSL* native_handle_type;

  /// The type of the next layer.
  typedef typename remove_reference<datagram_socket>::type next_layer_type;

  /// The type of the lowest layer.
  typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

  /// The type of the executor associated with the object.
  typedef typename lowest_layer_type::executor_type executor_type;

  template <typename Arg>
  dtls(Arg &arg, context& ctx)
    : ssl_base<datagram_socket>(arg, ctx)
  {
  }

  ~dtls()
  {
  }

  /// Receive some data on a dtls connection.
  /**
   * This function is used to receive data on the dtls connection.
   * The function call will block until data has been received successfully
   * or an error occurs.
   *
   * @param buffers One or more buffers into which the data will be received.
   *
   * @returns The number of bytes received.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @par Example
   * To receive into a single data buffer use the @ref buffer function as
   * follows:
   * @code socket.receive(asio::buffer(data, size)); @endcode
   * See the @ref buffer documentation for information on receiving into
   * multiple buffers in one go, and how to use it with arrays, boost::array or
   * std::vector.
   */
  template <typename BufferSequence>
  size_t receive(BufferSequence mb, asio::error_code &ec)
  {
    return detail::io(this->next_layer_, this->core_, detail::read_op<BufferSequence>(mb), ec);
  }

  /// Receive some data from the stream.
  /**
   * This function is used to read data from the stream. The function call will
   * block until one or more bytes of data has been read successfully, or until
   * an error occurs.
   *
   * @param buffers The buffers into which the data will be read.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns The number of bytes read. Returns 0 if an error occurred.
   */
  template <typename BufferSequence>
  size_t receive(BufferSequence mb)
  {
    asio::error_code ec;
    std::size_t res = detail::io(this->next_layer_, this->core_, detail::read_op<BufferSequence>(mb), ec);
    asio::detail::throw_error(ec, "receive");
    return res;
  }

  /// Start an asynchronous receive.
  /**
   * This function is used to asynchronously read one or more bytes of data from
   * the dtls connection. The function call always returns immediately.
   *
   * @param buffers The buffers into which the data will be read. Although the
   * buffers object may be copied as necessary, ownership of the underlying
   * buffers is retained by the caller, which must guarantee that they remain
   * valid until the handler is called.
   *
   * @param handler The handler to be called when the read operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   std::size_t bytes_transferred           // Number of bytes read.
   * ); @endcode
   */
  template <typename MutableBufferSequence, typename ReadHandler>
  ASIO_INITFN_RESULT_TYPE(ReadHandler,
                          void (asio::error_code, std::size_t))
  async_receive(const MutableBufferSequence& buffers,
                ASIO_MOVE_ARG(ReadHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a ReadHandler.
    ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

    asio::async_completion<ReadHandler,
      void (asio::error_code, std::size_t)> init(handler);

    detail::async_io(this->next_layer_, this->core_,
        detail::read_op<MutableBufferSequence>(buffers), init.handler);

    return init.result.get();
  }

  /// Send data on the dtls connection.
  /**
   * This function is used to write data on the dtls connection. The function
   * call will block until one or more bytes of data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers The data to be written to the dtls connection.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @returns The number of bytes written. Returns 0 if an error occurred.
   *
   */
  template <typename BufferSequence>
  size_t send(BufferSequence cb, asio::error_code &ec)
  {
    return detail::io(this->next_layer_, this->core_, detail::write_op<BufferSequence>(cb), ec);
  }

  /// Send data on the dtls connection.
  /**
   * This function is used to write data on the dtls connection. The function
   * call will block until one or more bytes of data has been written
   * successfully, or until an error occurs.
   *
   * @param buffers The data to be written to the dtls connection.
   *
   * @returns The number of bytes written. Returns 0 if an error occurred.
   *
   */
  template <typename BufferSequence>
  size_t send(BufferSequence cb)
  {
    asio::error_code ec;
    std::size_t res = detail::io(this->next_layer_, this->core_, detail::write_op<BufferSequence>(cb), ec);
    asio::detail::throw_error(ec, "send");
    return res;
  }

  /// Start an asynchronous write.
  /**
   * This function is used to asynchronously write one or more bytes of data to
   * the stream. The function call always returns immediately.
   *
   * @param buffers The data to be written to the stream. Although the buffers
   * object may be copied as necessary, ownership of the underlying buffers is
   * retained by the caller, which must guarantee that they remain valid until
   * the handler is called.
   *
   * @param handler The handler to be called when the write operation completes.
   * Copies will be made of the handler as required. The equivalent function
   * signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   std::size_t bytes_transferred           // Number of bytes written.
   * ); @endcode
   *
   * @note The async_write_some operation may not transmit all of the data to
   * the peer. Consider using the @ref async_write function if you need to
   * ensure that all data is written before the blocking operation completes.
   */
  template <typename ConstBufferSequence, typename WriteHandler>
  ASIO_INITFN_RESULT_TYPE(WriteHandler,
      void (asio::error_code, std::size_t))
  async_send(const ConstBufferSequence& buffers,
             ASIO_MOVE_ARG(WriteHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a WriteHandler.
    ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

    asio::async_completion<WriteHandler,
      void (asio::error_code, std::size_t)> init(handler);

    detail::async_io(this->next_layer_, this->core_,
        detail::write_op<ConstBufferSequence>(buffers), init.handler);

    return init.result.get();
  }

};

} // namespace ssl
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_DTLS_HPP
