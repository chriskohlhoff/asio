#include <ctime>
#include <iostream>
#include <boost/bind.hpp>
#include <asio.hpp>

void handle_tcp_write(asio::ip::tcp::socket* socket, char* write_buf)
{
  using namespace std; // For free.
  free(write_buf);
  delete socket;
}

void handle_tcp_accept(asio::ip::tcp::acceptor* acceptor,
    asio::ip::tcp::socket* socket, const asio::error& error)
{
  if (!error)
  {
    using namespace std; // For time_t, time, ctime, strdup and strlen.
    time_t now = time(0);
    char* write_buf = strdup(ctime(&now));
    size_t write_length = strlen(write_buf);

    asio::async_write(*socket,
        asio::buffer(write_buf, write_length),
        boost::bind(handle_tcp_write, socket, write_buf));

    socket = new asio::ip::tcp::socket(acceptor->io_service());

    acceptor->async_accept(*socket,
        boost::bind(handle_tcp_accept, acceptor, socket,
          asio::placeholders::error));
  }
  else
  {
    delete socket;
  }
}

void handle_udp_send_to(char* send_buf)
{
  using namespace std; // For free.
  free(send_buf);
}

void handle_udp_receive_from(asio::ip::udp::socket* socket,
    char* recv_buf, size_t recv_length,
    asio::ip::udp::endpoint* remote_endpoint,
    const asio::error& error)
{
  if (!error || error == asio::error::message_size)
  {
    using namespace std; // For time_t, time, ctime, strdup and strlen.
    time_t now = time(0);
    char* send_buf = strdup(ctime(&now));
    size_t send_length = strlen(send_buf);

    socket->async_send_to(
        asio::buffer(send_buf, send_length), *remote_endpoint,
        boost::bind(handle_udp_send_to, send_buf));

    socket->async_receive_from(
        asio::buffer(recv_buf, recv_length), *remote_endpoint,
        boost::bind(handle_udp_receive_from, socket, recv_buf, recv_length,
          remote_endpoint, asio::placeholders::error));
  }
}

int main()
{
  try
  {
    asio::io_service io_service;

    asio::ip::tcp::acceptor tcp_acceptor(io_service,
        asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 13));

    asio::ip::tcp::socket* tcp_socket
      = new asio::ip::tcp::socket(io_service);

    tcp_acceptor.async_accept(*tcp_socket,
        boost::bind(handle_tcp_accept, &tcp_acceptor, tcp_socket,
          asio::placeholders::error));

    asio::ip::udp::socket udp_socket(io_service,
        asio::ip::udp::endpoint(asio::ip::udp::v4(), 13));

    char recv_buf[1];
    size_t recv_length = sizeof(recv_buf);
    asio::ip::udp::endpoint remote_endpoint;

    udp_socket.async_receive_from(
        asio::buffer(recv_buf, recv_length), remote_endpoint,
        boost::bind(handle_udp_receive_from, &udp_socket, recv_buf, recv_length,
          &remote_endpoint, asio::placeholders::error));

    io_service.run();
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
