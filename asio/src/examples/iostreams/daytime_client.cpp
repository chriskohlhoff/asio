#include <iostream>
#include <string>
#include "socket_stream.hpp"

int main(int argc, char* argv[])
{
  try
  {
    if (argc != 2)
    {
      std::cerr << "Usage: daytime_client <host>" << std::endl;
      return 1;
    }

    socket_stream s(13, argv[1]);
    std::string line;
    std::getline(s, line);
    std::cout << line << std::endl;
  }
  catch (asio::error& e)
  {
    std::cout << e << std::endl;
  }

  return 0;
}
