#include <ctime>
#include <iostream>
#include "boost/bind.hpp"
#include "asio.hpp"

void handle_write(asio::stream_socket* socket, char* write_buf,
    const asio::error& /*error*/, size_t /*last_bytes_transferred*/,
    size_t /*total_bytes_transferred*/)
{
  using namespace std; // For free.
  free(write_buf);
  delete socket;
}

void handle_accept(asio::socket_acceptor* acceptor,
    asio::stream_socket* socket, const asio::error& error)
{
  if (!error)
  {
    using namespace std; // For time_t, time, ctime, strdup and strlen.
    time_t now = time(0);
    char* write_buf = strdup(ctime(&now));
    size_t write_length = strlen(write_buf);

    asio::async_write_n(*socket, asio::buffer(write_buf, write_length),
        boost::bind(handle_write, socket, write_buf,
          asio::placeholders::error,
          asio::placeholders::last_bytes_transferred,
          asio::placeholders::total_bytes_transferred));

    socket = new asio::stream_socket(acceptor->demuxer());

    acceptor->async_accept(*socket,
        boost::bind(handle_accept, acceptor, socket,
          asio::placeholders::error));
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

    asio::socket_acceptor acceptor(demuxer, asio::ipv4::tcp::endpoint(13));

    asio::stream_socket* socket = new asio::stream_socket(demuxer);

    acceptor.async_accept(*socket,
        boost::bind(handle_accept, &acceptor, socket,
          asio::placeholders::error));

    demuxer.run();
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}

