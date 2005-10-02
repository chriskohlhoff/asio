#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <string>
#include "asio.hpp"
#include "boost/bind.hpp"
#include "boost/lexical_cast.hpp"
#include "server.hpp"

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
    short port = boost::lexical_cast<short>(argv[1]);
    std::string doc_root = argv[2];

    // Block all signals for background thread.
    sigset_t new_mask;
    sigfillset(&new_mask);
    sigset_t old_mask;
    pthread_sigmask(SIG_BLOCK, &new_mask, &old_mask);

    // Run server in background thread.
    http::server s(port, doc_root);
    asio::thread t(boost::bind(&http::server::run, &s));

    // Restore previous signals.
    pthread_sigmask(SIG_SETMASK, &old_mask, 0);

    // Wait for signal indicating time to shut down.
    sigset_t wait_mask;
    sigemptyset(&wait_mask);
    sigaddset(&wait_mask, SIGINT);
    sigaddset(&wait_mask, SIGQUIT);
    sigaddset(&wait_mask, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &wait_mask, 0);
    int sig = 0;
    sigwait(&wait_mask, &sig);

    // Stop the server.
    s.stop();
    t.join();
  }
  catch (asio::error& e)
  {
    std::cerr << "asio error: " << e << "\n";
  }

  return 0;
}
