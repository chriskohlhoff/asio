#ifndef ASIO_DETAIL_WINRT_DSOCKET_SERVICE_HPP
#define ASIO_DETAIL_WINRT_DSOCKET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif // defined(_MSC_VER) && (_MSC_VER >= 1200)

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)

#include "asio/error.hpp"
#include "asio/io_service.hpp"
#include "asio/detail/addressof.hpp"
#include "asio/detail/winrt_socket_connect_op.hpp"
#include "asio/detail/winrt_dsocket_service_base.hpp"
#include "asio/detail/winrt_utils.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class winrt_dsocket_service :
  public winrt_dsocket_service_base
{
public:
  // The protocol type.
  typedef Protocol protocol_type;

  // The endpoint type.
  typedef typename Protocol::endpoint endpoint_type;

  // The native type of a socket.
  typedef Windows::Networking::Sockets::DatagramSocket^ native_handle_type;

  // The implementation type of the socket.
  struct implementation_type : base_implementation_type
  {
    // Default constructor.
    implementation_type()
      : base_implementation_type(),
        protocol_(endpoint_type().protocol())
    {
    }

    // The protocol associated with the socket.
    protocol_type protocol_;
  };

  // Constructor.
  winrt_dsocket_service(asio::io_service& io_service)
    : winrt_dsocket_service_base(io_service)
  {
  }

  // Move-construct a new socket implementation.
  void move_construct(implementation_type& impl,
      implementation_type& other_impl)
  {
    this->base_move_construct(impl, other_impl);

    impl.protocol_ = other_impl.protocol_;
    other_impl.protocol_ = endpoint_type().protocol();
  }

  // Move-assign from another socket implementation.
  void move_assign(implementation_type& impl,
      winrt_dsocket_service& other_service,
      implementation_type& other_impl)
  {
    this->base_move_assign(impl, other_service, other_impl);

    impl.protocol_ = other_impl.protocol_;
    other_impl.protocol_ = endpoint_type().protocol();
  }

  // Move-construct a new socket implementation from another protocol type.
  template <typename Protocol1>
  void converting_move_construct(implementation_type& impl,
      typename winrt_dsocket_service<
        Protocol1>::implementation_type& other_impl)
  {
    this->base_move_construct(impl, other_impl);

    impl.protocol_ = protocol_type(other_impl.protocol_);
    other_impl.protocol_ = typename Protocol1::endpoint().protocol();
  }

  // Open a new socket implementation.
  asio::error_code open(implementation_type& impl,
      const protocol_type& protocol, asio::error_code& ec)
  {
    if (is_open(impl))
    {
      ec = asio::error::already_open;
      return ec;
    }

    try
    {
      using namespace Windows::Networking::Sockets;
      using namespace Windows::Foundation;
      impl.socket_ = ref new DatagramSocket;
      receive_handler& recv_handler = *impl.receive_handler_;
      impl.socket_->MessageReceived +=
          ref new TypedEventHandler<DatagramSocket^, DatagramSocketMessageReceivedEventArgs^>(
              [&recv_handler](DatagramSocket^, DatagramSocketMessageReceivedEventArgs^ args) -> void
              {
                recv_handler(args);
              });
      impl.protocol_ = protocol;
      ec = asio::error_code();
    }
    catch (Platform::Exception^ e)
    {
      ec = asio::error_code(e->HResult,
            asio::system_category());
    }

    return ec;
  }

  // Assign a native socket to a socket implementation.
  asio::error_code assign(implementation_type& impl,
      const protocol_type& protocol, const native_handle_type& native_socket,
      asio::error_code& ec)
  {
    if (is_open(impl))
    {
      ec = asio::error::already_open;
      return ec;
    }

    impl.socket_ = native_socket;
    impl.socket_->MessageReceived += create_recv_event_handler();
    impl.protocol_ = protocol;
    ec = asio::error_code();

    return ec;
  }

  // Bind the socket to the specified local endpoint.
  asio::error_code bind(implementation_type& impl,
      const endpoint_type& peer_endpoint, asio::error_code& ec)
  {
      return do_bind(impl, peer_endpoint.data(), ec);
  }

  // Get the local endpoint.
  endpoint_type local_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    endpoint_type endpoint;
    endpoint.resize(do_get_endpoint(impl, true,
          endpoint.data(), endpoint.size(), ec));
    return endpoint;
  }

  // Get the remote endpoint.
  endpoint_type remote_endpoint(const implementation_type& impl,
      asio::error_code& ec) const
  {
    endpoint_type endpoint;
    endpoint.resize(do_get_endpoint(impl, false,
          endpoint.data(), endpoint.size(), ec));
    return endpoint;
  }

  // Set a socket option.
  template <typename Option>
  asio::error_code set_option(implementation_type& impl,
      const Option& option, asio::error_code& ec)
  {
    return do_set_option(impl, option.level(impl.protocol_),
        option.name(impl.protocol_), option.data(impl.protocol_),
        option.size(impl.protocol_), ec);
  }

  // Get a socket option.
  template <typename Option>
  asio::error_code get_option(const implementation_type& impl,
      Option& option, asio::error_code& ec) const
  {
    std::size_t size = option.size(impl.protocol_);
    do_get_option(impl, option.level(impl.protocol_),
        option.name(impl.protocol_),
        option.data(impl.protocol_), &size, ec);
    if (!ec)
      option.resize(impl.protocol_, size);
    return ec;
  }

  // Connect the socket to the specified endpoint.
  asio::error_code connect(implementation_type& impl,
      const endpoint_type& peer_endpoint, asio::error_code& ec)
  {
    return do_connect(impl, peer_endpoint.data(), ec);
  }

  // Start an asynchronous connect.
  template <typename Handler>
  void async_connect(implementation_type& impl,
      const endpoint_type& peer_endpoint, Handler& handler)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef winrt_socket_connect_op<Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(handler);

    ASIO_HANDLER_CREATION((p.p, "socket", &impl, "async_connect"));

    start_connect_op(impl, peer_endpoint.data(), p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Send a datagram to the specified endpoint.
  template <typename ConstBufferSequence>
  std::size_t send_to(implementation_type& impl,
      const ConstBufferSequence& buffers, const endpoint_type& destination,
      socket_base::message_flags flags, asio::error_code& ec)
  {
    return do_send_to(impl, buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence>::first(buffers), destination.data(), flags, ec);
  }

  // Wait until data can be sent without blocking.
  std::size_t send_to(base_implementation_type&, const null_buffers&,
      const endpoint_type&, socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous send. The data being sent must be valid for the
  // lifetime of the asynchronous operation.
  template <typename ConstBufferSequence, typename Handler>
  void async_send_to(base_implementation_type& impl,
      const ConstBufferSequence& buffers, const endpoint_type& destination,
      socket_base::message_flags flags, Handler& handler)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef winrt_socket_send_op<ConstBufferSequence, Handler> op;
    typename op::ptr p = { asio::detail::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0 };
    p.p = new (p.v) op(buffers, handler);

    ASIO_HANDLER_CREATION((p.p, "socket", &impl, "async_send_to"));

    start_send_to_op(impl,
      buffer_sequence_adapter<asio::const_buffer,
        ConstBufferSequence>::first(buffers), destination.data(),
      flags, p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Start an asynchronous wait until data can be sent without blocking.
  template <typename Handler>
  void async_send_to(base_implementation_type&, const null_buffers&,
      const endpoint_type& destination, socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_service_.get_io_service().post(
      detail::bind_handler(handler, ec, bytes_transferred));
  }

  // Receive a datagram with the endpoint of the sender. Returns the number of
  // bytes received.
  template <typename MutableBufferSequence>
  size_t receive_from(implementation_type& impl,
      const MutableBufferSequence& buffers,
      endpoint_type& sender_endpoint, socket_base::message_flags flags,
      asio::error_code& ec)
  {
    return do_receive_from(impl, buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence>::first(buffers), sender_endpoint.data(), flags, ec);
  }

  // Wait until data can be received without blocking.
  size_t receive_from(implementation_type&,
      const null_buffers&, endpoint_type&,
      socket_base::message_flags, asio::error_code& ec)
  {
    ec = asio::error::operation_not_supported;
    return 0;
  }

  // Start an asynchronous receive that will get the endpoint of the sender.
  template <typename MutableBufferSequence, typename Handler>
  void async_receive_from(base_implementation_type& impl,
      const MutableBufferSequence& buffers, endpoint_type& sender_endpoint,
      socket_base::message_flags flags, Handler& handler)
  {
    bool is_continuation =
      asio_handler_cont_helpers::is_continuation(handler);

    // Allocate and construct an operation to wrap the handler.
    typedef winrt_socket_recv_op<MutableBufferSequence, Handler> op;
    typename op::ptr p = {asio::detail::addressof(handler),
      asio_handler_alloc_helpers::allocate(
        sizeof(op), handler), 0};
    p.p = new (p.v) op(buffers, handler);

    ASIO_HANDLER_CREATION((p.p, "socket", &impl, "async_receive_from"));

    start_receive_op(impl,
      buffer_sequence_adapter<asio::mutable_buffer,
        MutableBufferSequence>::first(buffers), sender_endpoint.data(),
        flags, p.p, is_continuation);
    p.v = p.p = 0;
  }

  // Wait until data can be received without blocking.
  template <typename Handler>
  void async_receive_from(implementation_type&,
      const null_buffers&, endpoint_type&,
      socket_base::message_flags, Handler& handler)
  {
    asio::error_code ec = asio::error::operation_not_supported;
    const std::size_t bytes_transferred = 0;
    io_service_.get_io_service().post(
        detail::bind_handler(handler, ec, bytes_transferred));
  }
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // defined(ASIO_WINDOWS_RUNTIME)

#endif // ASIO_DETAIL_WINRT_DSOCKET_SERVICE_HPP
