#ifndef SERVICES_STREAM_SOCKET_SERVICE_HPP
#define SERVICES_STREAM_SOCKET_SERVICE_HPP

#include <asio.hpp>
#include <boost/noncopyable.hpp>
#include "logger.hpp"

namespace services {

/// Debugging stream socket service that wraps the normal stream socket service.
template <typename Protocol>
class stream_socket_service
  : private boost::noncopyable
{
private:
  /// The type of the wrapped stream socket service.
  typedef asio::stream_socket_service<Protocol> service_impl_type;

public:
  /// The protocol type.
  typedef Protocol protocol_type;

  /// The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  /// The implementation type of a stream socket.
  typedef typename service_impl_type::implementation_type implementation_type;

  /// The native type of a stream socket.
  typedef typename service_impl_type::native_type native_type;

  /// Construct a new stream socket service for the specified io_service.
  explicit stream_socket_service(asio::io_service& io_service)
    : service_impl_(io_service.get_service(
          asio::service_factory<service_impl_type>())),
      logger_(io_service, "stream_socket")
  {
  }

  /// Get the io_service associated with the service.
  asio::io_service& io_service()
  {
    return service_impl_.io_service();
  }

  /// Construct a new stream socket implementation.
  void construct(implementation_type& impl)
  {
    service_impl_.construct(impl);
  }

  /// Destroy a stream socket implementation.
  void destroy(implementation_type& impl)
  {
    service_impl_.destroy(impl);
  }

  /// Open a new stream socket implementation.
  template <typename Error_Handler>
  void open(implementation_type& impl, const protocol_type& protocol,
      Error_Handler error_handler)
  {
    logger_.log("Opening new socket");
    service_impl_.open(impl, protocol, error_handler);
  }

  /// Open a stream socket from an existing native socket.
  template <typename Error_Handler>
  void open(implementation_type& impl, const native_type& native_socket,
      Error_Handler error_handler)
  {
    logger_.log("Opening native socket");
    service_impl_.open(impl, native_socket, error_handler);
  }

  /// Close a stream socket implementation.
  template <typename Error_Handler>
  void close(implementation_type& impl, Error_Handler error_handler)
  {
    logger_.log("Closing socket");
    service_impl_.close(impl, error_handler);
  }

  /// Bind the stream socket to the specified local endpoint.
  template <typename Error_Handler>
  void bind(implementation_type& impl, const endpoint_type& endpoint,
      Error_Handler error_handler)
  {
    logger_.log("Binding socket");
    service_impl_.bind(impl, endpoint, error_handler);
  }

  /// Connect the stream socket to the specified endpoint.
  template <typename Error_Handler>
  void connect(implementation_type& impl, const endpoint_type& peer_endpoint,
      Error_Handler error_handler)
  {
    logger_.log("Connecting socket");
    service_impl_.connect(impl, peer_endpoint, error_handler);
  }

  /// Handler to wrap asynchronous connect completion.
  template <typename Handler>
  class connect_handler
  {
  public:
    connect_handler(Handler h, logger& l)
      : handler_(h),
        logger_(l)
    {
    }

    void operator()(const asio::error& e)
    {
      if (e)
      {
        std::string msg = "Asynchronous connect failed: ";
        msg += e.what();
        logger_.log(msg);
      }
      else
      {
        logger_.log("Asynchronous connect succeeded");
      }

      handler_(e);
    }

  private:
    Handler handler_;
    logger& logger_;
  };

  /// Start an asynchronous connect.
  template <typename Handler>
  void async_connect(implementation_type& impl,
      const endpoint_type& peer_endpoint, Handler handler)
  {
    logger_.log("Starting asynchronous connect");
    service_impl_.async_connect(impl, peer_endpoint, 
        connect_handler<Handler>(handler, logger_));
  }

  /// Set a socket option.
  template <typename Option, typename Error_Handler>
  void set_option(implementation_type& impl, const Option& option,
      Error_Handler error_handler)
  {
    logger_.log("Setting socket option");
    service_impl_.set_option(impl, option, error_handler);
  }

  /// Get a socket option.
  template <typename Option, typename Error_Handler>
  void get_option(const implementation_type& impl, Option& option,
      Error_Handler error_handler) const
  {
    logger_.log("Getting socket option");
    service_impl_.get_option(impl, option, error_handler);
  }

  /// Perform an IO control command on the socket.
  template <typename IO_Control_Command, typename Error_Handler>
  void io_control(implementation_type& impl, IO_Control_Command& command,
      Error_Handler error_handler)
  {
    logger_.log("Performing IO control command on socket");
    service_impl_.io_control(impl, command, error_handler);
  }

  /// Get the local endpoint.
  template <typename Error_Handler>
  void local_endpoint(const implementation_type& impl,
      Error_Handler error_handler) const
  {
    logger_.log("Getting socket's local endpoint");
    return service_impl_.local_endpoint(impl, error_handler);
  }

  /// Get the remote endpoint.
  template <typename Error_Handler>
  endpoint_type remote_endpoint(const implementation_type& impl,
      Error_Handler error_handler) const
  {
    logger_.log("Getting socket's remote endpoint");
    return service_impl_.remote_endpoint(impl, error_handler);
  }

  /// Disable sends or receives on the socket.
  template <typename Error_Handler>
  void shutdown(implementation_type& impl,
      asio::socket_base::shutdown_type what, Error_Handler error_handler)
  {
    logger_.log("Shutting down socket");
    service_impl_.shutdown(impl, what, error_handler);
  }

  /// Send the given data to the peer.
  template <typename Const_Buffers, typename Error_Handler>
  std::size_t send(implementation_type& impl, const Const_Buffers& buffers,
      asio::socket_base::message_flags flags,
      Error_Handler error_handler)
  {
    logger_.log("Sending data on socket");
    return service_impl_.send(impl, buffers, flags, error_handler);
  }

  /// Handler to wrap asynchronous send completion.
  template <typename Handler>
  class send_handler
  {
  public:
    send_handler(Handler h, logger& l)
      : handler_(h),
        logger_(l)
    {
    }

    void operator()(const asio::error& e, std::size_t bytes_transferred)
    {
      if (e)
      {
        std::string msg = "Asynchronous send failed: ";
        msg += e.what();
        logger_.log(msg);
      }
      else
      {
        logger_.log("Asynchronous send succeeded");
      }

      handler_(e, bytes_transferred);
    }

  private:
    Handler handler_;
    logger& logger_;
  };

  /// Start an asynchronous send.
  template <typename Const_Buffers, typename Handler>
  void async_send(implementation_type& impl, const Const_Buffers& buffers,
      asio::socket_base::message_flags flags, Handler handler)
  {
    logger_.log("Starting asynchronous send");
    service_impl_.async_send(impl, buffers, flags,
        send_handler<Handler>(handler, logger_));
  }

  /// Receive some data from the peer.
  template <typename Mutable_Buffers, typename Error_Handler>
  std::size_t receive(implementation_type& impl, const Mutable_Buffers& buffers,
      asio::socket_base::message_flags flags,
      Error_Handler error_handler)
  {
    logger_.log("Receiving data on socket");
    return service_impl_.receive(impl, buffers, flags, error_handler);
  }

  /// Handler to wrap asynchronous receive completion.
  template <typename Handler>
  class receive_handler
  {
  public:
    receive_handler(Handler h, logger& l)
      : handler_(h),
        logger_(l)
    {
    }

    void operator()(const asio::error& e, std::size_t bytes_transferred)
    {
      if (e)
      {
        std::string msg = "Asynchronous receive failed: ";
        msg += e.what();
        logger_.log(msg);
      }
      else
      {
        logger_.log("Asynchronous receive succeeded");
      }

      handler_(e, bytes_transferred);
    }

  private:
    Handler handler_;
    logger& logger_;
  };

  /// Start an asynchronous receive.
  template <typename Mutable_Buffers, typename Handler>
  void async_receive(implementation_type& impl, const Mutable_Buffers& buffers,
      asio::socket_base::message_flags flags, Handler handler)
  {
    logger_.log("Starting asynchronous receive");
    service_impl_.async_receive(impl, buffers, flags,
        receive_handler<Handler>(handler, logger_));
  }

private:
  /// The wrapped stream socket service.
  service_impl_type& service_impl_;

  /// The logger used for writing debug messages.
  logger logger_;
};

} // namespace services

#endif // SERVICES_STREAM_SOCKET_SERVICE_HPP
