#ifndef ASIO_SSL_BASE_HPP
#define ASIO_SSL_BASE_HPP

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
#include "asio/ssl/stream_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {

template <typename lower_layer>
class ssl_base
{
public:
  /// The native handle type of the SSL stream.
  typedef SSL* native_handle_type;

  /// The type of the next layer.
  typedef typename remove_reference<lower_layer>::type next_layer_type;

  /// The type of the lowest layer.
  typedef typename ssl_base<lower_layer>::next_layer_type::lowest_layer_type lowest_layer_type;

  /// The type of the executor associated with the object.
  typedef typename lowest_layer_type::executor_type executor_type;

  typedef stream_base::handshake_type handshake_type;

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  /// Construct a stream.
  /**
   * This constructor creates a SSL object and initializes the underlying
   * transport object.
   *
   * @param arg The argument to be passed to initialise the underlying
   * transport.
   *
   * @param ctx The SSL context to be used for the stream.
   */
  template <typename Arg>
  ssl_base(Arg &&arg, context &ctx) :
    next_layer_(ASIO_MOVE_CAST(Arg)(arg)),
    core_(ctx.native_handle(),
    next_layer_.lowest_layer().get_executor().context())
  {
  }
#else // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
  template <typename Arg>
  ssl_base(Arg &arg, context &ctx) :
    next_layer_(arg),
    core_(ctx.native_handle(),
    next_layer_.lowest_layer().get_executor().context())
  {
  }
#endif // defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

  /// Get the executor associated with the object.
  /**
   * This function may be used to obtain the executor object that the stream
   * uses to dispatch handlers for asynchronous operations.
   *
   * @return A copy of the executor that stream will use to dispatch handlers.
   */
  executor_type get_executor() ASIO_NOEXCEPT
  {
    return next_layer_.lowest_layer().get_executor();
  }

  /// Get the underlying implementation in the native type.
  /**
   * This function may be used to obtain the underlying implementation of the
   * context. This is intended to allow access to context functionality that is
   * not otherwise provided.
   *
   * @par Example
   * The native_handle() function returns a pointer of type @c SSL* that is
   * suitable for passing to functions such as @c SSL_get_verify_result and
   * @c SSL_get_peer_certificate:
   * @code
   * asio::ssl::stream<asio:ip::tcp::socket> sock(io_context, ctx);
   *
   * // ... establish connection and perform handshake ...
   *
   * if (X509* cert = SSL_get_peer_certificate(sock.native_handle()))
   * {
   *   if (SSL_get_verify_result(sock.native_handle()) == X509_V_OK)
   *   {
   *     // ...
   *   }
   * }
   * @endcode
   */
  native_handle_type native_handle()
  {
    return core_.engine_.native_handle();
  }

  /// Get a reference to the next layer.
  /**
   * This function returns a reference to the next layer in a stack of stream
   * layers.
   *
   * @return A reference to the next layer in the stack of stream layers.
   * Ownership is not transferred to the caller.
   */
  const next_layer_type& next_layer() const
  {
    return next_layer_;
  }

  /// Get a reference to the next layer.
  /**
   * This function returns a reference to the next layer in a stack of stream
   * layers.
   *
   * @return A reference to the next layer in the stack of stream layers.
   * Ownership is not transferred to the caller.
   */
  next_layer_type& next_layer()
  {
    return next_layer_;
  }

  /// Get a reference to the lowest layer.
  /**
   * This function returns a reference to the lowest layer in a stack of
   * stream layers.
   *
   * @return A reference to the lowest layer in the stack of stream layers.
   * Ownership is not transferred to the caller.
   */
  lowest_layer_type& lowest_layer()
  {
    return next_layer_.lowest_layer();
  }

  /// Get a reference to the lowest layer.
  /**
   * This function returns a reference to the lowest layer in a stack of
   * stream layers.
   *
   * @return A reference to the lowest layer in the stack of stream layers.
   * Ownership is not transferred to the caller.
   */
  const lowest_layer_type& lowest_layer() const
  {
    return next_layer_.lowest_layer();
  }

  /// Set the peer verification mode.
  /**
   * This function may be used to configure the peer verification mode used by
   * the stream. The new mode will override the mode inherited from the context.
   *
   * @param v A bitmask of peer verification modes. See @ref verify_mode for
   * available values.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_set_verify.
   */
  void set_verify_mode(verify_mode v)
  {
    asio::error_code ec;
    set_verify_mode(v, ec);
    asio::detail::throw_error(ec, "set_verify_mode");
  }

  /// Set the peer verification mode.
  /**
   * This function may be used to configure the peer verification mode used by
   * the stream. The new mode will override the mode inherited from the context.
   *
   * @param v A bitmask of peer verification modes. See @ref verify_mode for
   * available values.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_set_verify.
   */
  ASIO_SYNC_OP_VOID set_verify_mode(
      verify_mode v, asio::error_code& ec)
  {
    core_.engine_.set_verify_mode(v, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Set the peer verification depth.
  /**
   * This function may be used to configure the maximum verification depth
   * allowed by the stream.
   *
   * @param depth Maximum depth for the certificate chain verification that
   * shall be allowed.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_set_verify_depth.
   */
  void set_verify_depth(int depth)
  {
    asio::error_code ec;
    set_verify_depth(depth, ec);
    asio::detail::throw_error(ec, "set_verify_depth");
  }

  /// Set the peer verification depth.
  /**
   * This function may be used to configure the maximum verification depth
   * allowed by the stream.
   *
   * @param depth Maximum depth for the certificate chain verification that
   * shall be allowed.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_set_verify_depth.
   */
  ASIO_SYNC_OP_VOID set_verify_depth(
      int depth, asio::error_code& ec)
  {
    core_.engine_.set_verify_depth(depth, ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Set the callback used to verify peer certificates.
  /**
   * This function is used to specify a callback function that will be called
   * by the implementation when it needs to verify a peer certificate.
   *
   * @param callback The function object to be used for verifying a certificate.
   * The function signature of the handler must be:
   * @code bool verify_callback(
   *   bool preverified, // True if the certificate passed pre-verification.
   *   verify_context& ctx // The peer certificate and other context.
   * ); @endcode
   * The return value of the callback is true if the certificate has passed
   * verification, false otherwise.
   *
   * @throws asio::system_error Thrown on failure.
   *
   * @note Calls @c SSL_set_verify.
   */
  template <typename VerifyCallback>
  void set_verify_callback(VerifyCallback callback)
  {
    asio::error_code ec;
    this->set_verify_callback(callback, ec);
    asio::detail::throw_error(ec, "set_verify_callback");
  }

  /// Set the callback used to verify peer certificates.
  /**
   * This function is used to specify a callback function that will be called
   * by the implementation when it needs to verify a peer certificate.
   *
   * @param callback The function object to be used for verifying a certificate.
   * The function signature of the handler must be:
   * @code bool verify_callback(
   *   bool preverified, // True if the certificate passed pre-verification.
   *   verify_context& ctx // The peer certificate and other context.
   * ); @endcode
   * The return value of the callback is true if the certificate has passed
   * verification, false otherwise.
   *
   * @param ec Set to indicate what error occurred, if any.
   *
   * @note Calls @c SSL_set_verify.
   */
  template <typename VerifyCallback>
  ASIO_SYNC_OP_VOID set_verify_callback(VerifyCallback callback,
      asio::error_code& ec)
  {
    core_.engine_.set_verify_callback(
        new detail::verify_callback<VerifyCallback>(callback), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Perform SSL handshaking.
  /**
   * This function is used to perform SSL handshaking on the stream. The
   * function call will block until handshaking is complete or an error occurs.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void handshake(handshake_type type)
  {
    asio::error_code ec;
    handshake(type, ec);
    asio::detail::throw_error(ec, "handshake");
  }

  /// Perform SSL handshaking.
  /**
   * This function is used to perform SSL handshaking on the stream. The
   * function call will block until handshaking is complete or an error occurs.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID handshake(handshake_type type,
      asio::error_code& ec)
  {
    detail::io(next_layer_, core_, detail::handshake_op(type), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Perform SSL handshaking.
  /**
   * This function is used to perform SSL handshaking on the stream. The
   * function call will block until handshaking is complete or an error occurs.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @param buffers The buffered data to be reused for the handshake.
   *
   * @throws asio::system_error Thrown on failure.
   */
  template <typename ConstBufferSequence>
  void handshake(handshake_type type, const ConstBufferSequence& buffers)
  {
    asio::error_code ec;
    handshake(type, buffers, ec);
    asio::detail::throw_error(ec, "handshake");
  }

  /// Perform SSL handshaking.
  /**
   * This function is used to perform SSL handshaking on the stream. The
   * function call will block until handshaking is complete or an error occurs.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @param buffers The buffered data to be reused for the handshake.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  template <typename ConstBufferSequence>
  ASIO_SYNC_OP_VOID handshake(handshake_type type,
      const ConstBufferSequence& buffers, asio::error_code& ec)
  {
    detail::io(next_layer_, core_,
        detail::buffered_handshake_op<ConstBufferSequence>(type, buffers), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Start an asynchronous SSL handshake.
  /**
   * This function is used to asynchronously perform an SSL handshake on the
   * stream. This function call always returns immediately.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @param handler The handler to be called when the handshake operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error // Result of operation.
   * ); @endcode
   */
  template <typename HandshakeHandler>
  ASIO_INITFN_RESULT_TYPE(HandshakeHandler,
      void (asio::error_code))
  async_handshake(handshake_type type,
      ASIO_MOVE_ARG(HandshakeHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a HandshakeHandler.
    ASIO_HANDSHAKE_HANDLER_CHECK(HandshakeHandler, handler) type_check;

    asio::async_completion<HandshakeHandler,
      void (asio::error_code)> init(handler);

    detail::async_io(next_layer_, core_,
        detail::handshake_op(type), init.completion_handler);

    return init.result.get();
  }

  /// Start an asynchronous SSL handshake.
  /**
   * This function is used to asynchronously perform an SSL handshake on the
   * stream. This function call always returns immediately.
   *
   * @param type The type of handshaking to be performed, i.e. as a client or as
   * a server.
   *
   * @param buffers The buffered data to be reused for the handshake. Although
   * the buffers object may be copied as necessary, ownership of the underlying
   * buffers is retained by the caller, which must guarantee that they remain
   * valid until the handler is called.
   *
   * @param handler The handler to be called when the handshake operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error, // Result of operation.
   *   std::size_t bytes_transferred // Amount of buffers used in handshake.
   * ); @endcode
   */
  template <typename ConstBufferSequence, typename BufferedHandshakeHandler>
  ASIO_INITFN_RESULT_TYPE(BufferedHandshakeHandler,
      void (asio::error_code, std::size_t))
  async_handshake(handshake_type type, const ConstBufferSequence& buffers,
      ASIO_MOVE_ARG(BufferedHandshakeHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a BufferedHandshakeHandler.
    ASIO_BUFFERED_HANDSHAKE_HANDLER_CHECK(
        BufferedHandshakeHandler, handler) type_check;

    asio::async_completion<BufferedHandshakeHandler,
      void (asio::error_code, std::size_t)> init(handler);

    detail::async_io(next_layer_, core_,
        detail::buffered_handshake_op<ConstBufferSequence>(type, buffers),
        init.completion_handler);

    return init.result.get();
  }

  /// Shut down SSL on the stream.
  /**
   * This function is used to shut down SSL on the stream. The function call
   * will block until SSL has been shut down or an error occurs.
   *
   * @throws asio::system_error Thrown on failure.
   */
  void shutdown()
  {
    asio::error_code ec;
    shutdown(ec);
    asio::detail::throw_error(ec, "shutdown");
  }

  /// Shut down SSL on the stream.
  /**
   * This function is used to shut down SSL on the stream. The function call
   * will block until SSL has been shut down or an error occurs.
   *
   * @param ec Set to indicate what error occurred, if any.
   */
  ASIO_SYNC_OP_VOID shutdown(asio::error_code& ec)
  {
    detail::io(next_layer_, core_, detail::shutdown_op(), ec);
    ASIO_SYNC_OP_VOID_RETURN(ec);
  }

  /// Asynchronously shut down SSL on the stream.
  /**
   * This function is used to asynchronously shut down SSL on the stream. This
   * function call always returns immediately.
   *
   * @param handler The handler to be called when the handshake operation
   * completes. Copies will be made of the handler as required. The equivalent
   * function signature of the handler must be:
   * @code void handler(
   *   const asio::error_code& error // Result of operation.
   * ); @endcode
   */
  template <typename ShutdownHandler>
  ASIO_INITFN_RESULT_TYPE(ShutdownHandler,
      void (asio::error_code))
  async_shutdown(ASIO_MOVE_ARG(ShutdownHandler) handler)
  {
    // If you get an error on the following line it means that your handler does
    // not meet the documented type requirements for a ShutdownHandler.
    ASIO_SHUTDOWN_HANDLER_CHECK(ShutdownHandler, handler) type_check;

    asio::async_completion<ShutdownHandler,
      void (asio::error_code)> init(handler);

    detail::async_io(next_layer_, core_, detail::shutdown_op(),
        init.completion_handler);

    return init.result.get();
  }

protected:
  lower_layer next_layer_;
  detail::stream_core core_;
};

}
}

#include "asio/detail/pop_options.hpp"

#endif // ASIO_SSL_BASE_HPP
