#include <ctime>
#include <iostream>
#include <boost/bind.hpp>
#include <asio.hpp>

void handle_write(asio::ip::tcp::socket* socket, char* write_buf,
    const asio::error& /*error*/, size_t /*bytes_transferred*/)
{
  using namespace std; // For free.
  free(write_buf);
  delete socket;
}

void handle_accept(asio::ip::tcp::acceptor* acceptor,
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
        boost::bind(handle_write, socket, write_buf,
          asio::placeholders::error,
          asio::placeholders::bytes_transferred));

    socket = new asio::ip::tcp::socket(acceptor->io_service());

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
    asio::io_service io_service;

    asio::ip::tcp::acceptor acceptor(io_service,
        asio::ip::tcp::endpoint(asio::ip::tcp::v4(), 13));

    asio::ip::tcp::socket* socket
      = new asio::ip::tcp::socket(io_service);

    acceptor.async_accept(*socket,
        boost::bind(handle_accept, &acceptor, socket,
          asio::placeholders::error));

    io_service.run();
  }
  catch (asio::error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}

