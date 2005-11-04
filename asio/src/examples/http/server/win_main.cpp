#include <iostream>
#include <string>
#include <asio.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/lexical_cast.hpp>
#include "server.hpp"

#if defined(_WIN32)

boost::function0<void> console_ctrl_function;

BOOL WINAPI console_ctrl_handler(DWORD ctrl_type)
{
  switch (ctrl_type)
  {
  case CTRL_C_EVENT:
  case CTRL_BREAK_EVENT:
  case CTRL_CLOSE_EVENT:
  case CTRL_SHUTDOWN_EVENT:
    console_ctrl_function();
    return TRUE;
  default:
    return FALSE;
  }
}

int main(int argc, char* argv[])
{
  try
  {
    // Check command line arguments.
    if (argc != 3)
    {
      std::cerr << "Usage: http_server <port> <doc_root>\n";
      return 1;
    }
    unsigned short port = boost::lexical_cast<unsigned short>(argv[1]);
    std::string doc_root = argv[2];

    // Initialise server.
    http::server::server s(port, doc_root);

    // Set console control handler to allow server to be stopped.
    console_ctrl_function = boost::bind(&http::server::server::stop, &s);
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);

    // Run the server until stopped.
    s.run();
  }
  catch (asio::error& e)
  {
    std::cerr << "asio error: " << e << "\n";
  }
  catch (std::exception& e)
  {
    std::cerr << "exception: " << e.what() << "\n";
  }

  return 0;
}

#endif // defined(_WIN32)
