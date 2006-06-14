#include <ctime>
#include <iostream>
#include <string>
#include <asio.hpp>

using asio::ip::tcp;

std::string make_daytime_string()
{
  using namespace std; // For time_t, time and ctime;
  time_t now = time(0);
  return ctime(&now);
}

int main()
{
  try
  {
    asio::io_service io_service;

    tcp::endpoint endpoint(tcp::v4(), 13);
    tcp::acceptor acceptor(io_service, endpoint);

    for (;;)
    {
      tcp::iostream stream;
      acceptor.accept(*stream.rdbuf());
      stream << make_daytime_string();
    }
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
