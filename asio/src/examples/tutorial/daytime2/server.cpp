#include <ctime>
#include <iostream>
#include <string>
#include "asio.hpp"

int main()
{
  try
  {
    asio::demuxer demuxer;

    asio::socket_acceptor acceptor(demuxer, asio::ipv4::address(13));

    for (;;)
    {
      asio::stream_socket socket(demuxer);
      acceptor.accept(socket);

      using namespace std; // For time_t, time and ctime.
      time_t now = time(0);
      std::string msg = ctime(&now);

      asio::send_n(socket, msg.c_str(), msg.length(), 0, asio::ignore_error());
    }
  }
  catch (asio::socket_error& e)
  {
    std::cerr << e << std::endl;
  }

  return 0;
}
