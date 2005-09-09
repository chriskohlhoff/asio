#include <ctime>
#include <iostream>
#include "boost/bind.hpp"
#include "asio.hpp"

void handle_send_to(char* send_buf, const asio::error& /*error*/,
    size_t /*bytes_transferred*/)
{
  using namespace std; // For free.
  free(send_buf);
}

void handle_receive_from(asio::datagram_socket* socket, char* recv_buf,
    size_t recv_length, asio::ipv4::udp::endpoint* remote_endpoint,
    const asio::error& error, size_t /*bytes_transferred*/)
{
  if (!error || error == asio::error::message_size)
  {
    using namespace std; // For time_t, time, ctime, strdup and strlen.
    time_t now = time(0);
    char* send_buf = strdup(ctime(&now));
    size_t send_length = strlen(send_buf);

    socket->async_send_to(
        asio::buffers(send_buf, send_length), 0, *remote_endpoint,
        boost::bind(handle_send_to, send_buf, asio::placeholders::error,
          asio::placeholders::bytes_transferred));

    socket->async_receive_from(
        asio::buffers(recv_buf, recv_length), 0, *remote_endpoint,
        boost::bind(handle_receive_from, socket, recv_buf, recv_length,
          remote_endpoint, asio::placeholders::error,
          asio::placeholders::bytes_transferred));
  }
}

int main()
{
  try
  {
    asio::demuxer demuxer;

    asio::datagram_socket socket(demuxer, asio::ipv4::udp::endpoint(13));

    char recv_buf[1];
    size_t recv_length = sizeof(recv_buf);
    asio::ipv4::udp::endpoint remote_endpoint;

    socket.async_receive_from(
        asio::buffers(recv_buf, recv_length), 0, remote_endpoint,
        boost::bind(handle_receive_from, &socket, recv_buf, recv_length,
          &remote_endpoint, asio::placeholders::error,
          asio::placeholders::bytes_transferred));

    demuxer.run();
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
