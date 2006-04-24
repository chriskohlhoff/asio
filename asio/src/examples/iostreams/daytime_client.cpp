// This example is disabled for gcc 4.0.2 because Boost.Iostreams and that
// compiler version don't seem to get along very well.
#if defined(__GNUC__) \
  && (__GNUC__ == 4 && __GNUC_MINOR__ == 0 && __GNUC_PATCHLEVEL__ == 2)
#warning iostreams example is disabled for gcc 4.0.2
int main() {}
#else

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

    socket_stream s(argv[1], "daytime");
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

#endif
