// Compile with:
//
//   g++ -std=c++2a -fcoroutines-ts -Wall -Wextra -Ipath-to-asio/include \
//     -DASIO_HAS_APPLE_NETWORK_FRAMEWORK -o https_client https_client.cpp -lNetwork

#include <net.hpp>
#include <netx.hpp>
#include <iostream>

using default_token = netx::as_single_t<netx::use_awaitable_t<>>;
using socket_type = default_token::as_default_on_t<netx::generic_stream_socket>;

netx::awaitable<void> run(net::io_context& ctx)
{
  socket_type socket(ctx);
  co_await socket.async_connect(netx::host(netx::ip::tls_tcp::any(), "www.boost.org", "443"));

  std::cout << "Sending request" << std::endl;
  std::string request("GET /LICENSE_1_0.txt HTTP/1.0\r\nHost: www.boost.org\r\n\r\n");
  auto [err1, n1] = co_await net::async_write(socket, net::buffer(request));
  if (err1)
  {
    std::cerr << "failed to send request" << std::endl;
    co_return;
  }

  std::cout << "Sent request, waiting for response" << std::endl;
  std::string response;
  auto [err2, n2] = co_await net::async_read(socket, net::dynamic_buffer(response));
  if (err2 && err2 != net::stream_errc::eof)
  {
    std::cerr << "failed to receive response" << std::endl;
    co_return;
  }

  std::cout << "Received response" << std::endl;
  std::cout << response << std::endl;
}

int main()
{
  net::io_context ctx;
  netx::co_spawn(ctx, run(ctx), netx::detached);
  ctx.run();
}
