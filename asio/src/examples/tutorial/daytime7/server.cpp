#include <ctime>
#include <iostream>
#include "boost/bind.hpp"
#include "asio.hpp"

void handle_tcp_send(asio::stream_socket* socket, char* send_buf)
{
  using namespace std; // For free.
  free(send_buf);
  delete socket;
}

void handle_tcp_accept(asio::socket_acceptor* acceptor,
    asio::stream_socket* socket, const asio::error& error)
{
  if (!error)
  {
    using namespace std; // For time_t, time, ctime, strdup and strlen.
    time_t now = time(0);
    char* send_buf = strdup(ctime(&now));
    size_t send_length = strlen(send_buf);

    asio::async_send_n(*socket, send_buf, send_length,
        boost::bind(handle_tcp_send, socket, send_buf));

    socket = new asio::stream_socket(acceptor->demuxer());

    acceptor->async_accept(*socket,
        boost::bind(handle_tcp_accept, acceptor, socket, asio::arg::error));
  }
  else
  {
    delete socket;
  }
}

void handle_udp_sendto(char* send_buf)
{
  using namespace std; // For free.
  free(send_buf);
}

void handle_udp_recvfrom(asio::dgram_socket* socket, char* recv_buf,
    size_t recv_length, asio::ipv4::udp::endpoint* remote_endpoint,
    const asio::error& error)
{
  if (!error || error == asio::error::message_size)
  {
    using namespace std; // For time_t, time, ctime, strdup and strlen.
    time_t now = time(0);
    char* send_buf = strdup(ctime(&now));
    size_t send_length = strlen(send_buf);

    socket->async_sendto(send_buf, send_length, *remote_endpoint,
        boost::bind(handle_udp_sendto, send_buf));

    socket->async_recvfrom(recv_buf, recv_length, *remote_endpoint,
        boost::bind(handle_udp_recvfrom, socket, recv_buf, recv_length,
          remote_endpoint, asio::arg::error));
  }
}

int main()
{
  try
  {
    asio::demuxer demuxer;

    asio::socket_acceptor tcp_acceptor(demuxer, asio::ipv4::tcp::endpoint(13));

    asio::stream_socket* tcp_socket = new asio::stream_socket(demuxer);

    tcp_acceptor.async_accept(*tcp_socket,
        boost::bind(handle_tcp_accept, &tcp_acceptor, tcp_socket,
          asio::arg::error));

    asio::dgram_socket udp_socket(demuxer, asio::ipv4::udp::endpoint(13));

    char recv_buf[1];
    size_t recv_length = sizeof(recv_buf);
    asio::ipv4::udp::endpoint remote_endpoint;

    udp_socket.async_recvfrom(recv_buf, recv_length, remote_endpoint,
        boost::bind(handle_udp_recvfrom, &udp_socket, recv_buf, recv_length,
          &remote_endpoint, asio::arg::error));

    demuxer.run();
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
