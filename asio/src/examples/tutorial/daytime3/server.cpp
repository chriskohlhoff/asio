#include <ctime>
#include <iostream>
#include "boost/bind.hpp"
#include "asio.hpp"

void handle_send(asio::stream_socket* socket, char* send_buf,
    const asio::socket_error& /*error*/, size_t /*last_bytes_sent*/,
    size_t /*total_bytes_sent*/)
{
  using namespace std; // For free.
  free(send_buf);
  delete socket;
}

void handle_accept(asio::socket_acceptor* acceptor,
    asio::stream_socket* socket, const asio::socket_error& error)
{
  if (!error)
  {
    using namespace std; // For time_t, time, ctime, strdup and strlen.
    time_t now = time(0);
    char* send_buf = strdup(ctime(&now));
    size_t send_length = strlen(send_buf);

    asio::async_send_n(*socket, send_buf, send_length,
        boost::bind(handle_send, socket, send_buf, asio::arg::error,
          asio::arg::last_bytes_sent, asio::arg::total_bytes_sent));

    socket = new asio::stream_socket(acceptor->demuxer());

    acceptor->async_accept(*socket,
        boost::bind(handle_accept, acceptor, socket, asio::arg::error));
  }
  else
  {
    delete socket;
  }
}

int main()
{
  try
  {
    asio::demuxer demuxer;

    asio::socket_acceptor acceptor(demuxer, asio::ipv4::address(13));

    asio::stream_socket* socket = new asio::stream_socket(demuxer);

    acceptor.async_accept(*socket,
        boost::bind(handle_accept, &acceptor, socket, asio::arg::error));

    demuxer.run();
  }
  catch (asio::socket_error& e)
  {
    std::cerr << e.what() << ": " << e.message() << std::endl;
  }

  return 0;
}
