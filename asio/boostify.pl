#!/usr/bin/perl -w

use strict;
use File::Path;

my $boost_dir;

sub determine_boost_dir
{
  # Parse the configure.ac to determine the asio version.
  my $asio_version = "unknown";
  open(my $input, "<configure.ac") or die("Can't read configure.ac");
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /AM_INIT_AUTOMAKE\(asio, \[([^\]]*)\]\)/)
    {
      $asio_version = $1;
      $asio_version =~ s/\./_/g;
    }
  }
  close($input);

  # Create the boost directory name.
  our $boost_dir;
  $boost_dir = "boost_asio_$asio_version";
}

sub print_line
{
  my ($output, $line, $from, $lineno) = @_;

  # Warn if the resulting line is >80 characters wide.
  if (length($line) > 80)
  {
    if ($from =~ /\.[chi]pp$/)
    {
      print("Warning: $from:$lineno: output >80 characters wide.\n");
    }
  }

  # Write the output.
  print($output $line . "\n");
}

sub source_contains_asio_thread_usage
{
  my ($from) = @_;

  # Open the input file.
  open(my $input, "<$from") or die("Can't open $from for reading");

  # Check file for use of asio::thread.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /asio::thread/)
    {
      close($input);
      return 1;
    }
    elsif ($line =~ /^ *thread /)
    {
      close($input);
      return 1;
    }
  }

  close($input);
  return 0;
}

sub source_contains_asio_error_code_include
{
  my ($from) = @_;

  # Open the input file.
  open(my $input, "<$from") or die("Can't open $from for reading");

  # Check file for inclusion of asio/error_code.hpp.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /# *include [<"]asio\/error_code\.hpp[>"]/)
    {
      close($input);
      return 1;
    }
  }

  close($input);
  return 0;
}

sub source_contains_asio_system_error_include
{
  my ($from) = @_;

  # Open the input file.
  open(my $input, "<$from") or die("Can't open $from for reading");

  # Check file for inclusion of asio/system_error.hpp.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /# *include [<"]asio\/system_error\.hpp[>"]/)
    {
      close($input);
      return 1;
    }
  }

  close($input);
  return 0;
}

sub source_contains_boostify_error_categories
{
  my ($from) = @_;

  # Open the input file.
  open(my $input, "<$from") or die("Can't open $from for reading");

  # Check file for boostify error category directive.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /boostify: error category/)
    {
      close($input);
      return 1;
    }
  }

  close($input);
  return 0;
}

my $error_cat_decls = <<"EOF";
#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)
  static boost::system::error_category netdb_ecat;
  static int netdb_ed(const boost::system::error_code& ec);
  static std::string netdb_md(const boost::system::error_code& ec);
  static boost::system::wstring_t netdb_wmd(
      const boost::system::error_code& ec);

  static boost::system::error_category addrinfo_ecat;
  static int addrinfo_ed(const boost::system::error_code& ec);
  static std::string addrinfo_md(const boost::system::error_code& ec);
  static boost::system::wstring_t addrinfo_wmd(
      const boost::system::error_code& ec);
#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

  static boost::system::error_category misc_ecat;
  static int misc_ed(const boost::system::error_code& ec);
  static std::string misc_md(const boost::system::error_code& ec);
  static boost::system::wstring_t misc_wmd(const boost::system::error_code& ec);

  static boost::system::error_category ssl_ecat;
  static int ssl_ed(const boost::system::error_code& ec);
  static std::string ssl_md(const boost::system::error_code& ec);
  static boost::system::wstring_t ssl_wmd(const boost::system::error_code& ec);
EOF

my $error_cat_defns = <<"EOF";
#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

template <typename T>
boost::system::error_category error_base<T>::netdb_ecat(
    boost::system::error_code::new_category(&error_base<T>::netdb_ed,
      &error_base<T>::netdb_md, &error_base<T>::netdb_wmd));

template <typename T>
int error_base<T>::netdb_ed(const boost::system::error_code& ec)
{
  return EOTHER;
}

template <typename T>
std::string error_base<T>::netdb_md(const boost::system::error_code& ec)
{
  if (ec == error_base<T>::host_not_found)
    return "Host not found (authoritative)";
  if (ec == error_base<T>::host_not_found_try_again)
    return "Host not found (non-authoritative), try again later";
  if (ec == error_base<T>::no_data)
    return "The query is valid, but it does not have associated data";
  if (ec == error_base<T>::no_recovery)
    return "A non-recoverable error occurred during database lookup";
  return "EINVAL";
}

template <typename T>
boost::system::wstring_t error_base<T>::netdb_wmd(
    const boost::system::error_code& ec)
{
  if (ec == error_base<T>::host_not_found)
    return L"Host not found (authoritative)";
  if (ec == error_base<T>::host_not_found_try_again)
    return L"Host not found (non-authoritative), try again later";
  if (ec == error_base<T>::no_data)
    return L"The query is valid, but it does not have associated data";
  if (ec == error_base<T>::no_recovery)
    return L"A non-recoverable error occurred during database lookup";
  return L"EINVAL";
}

template <typename T>
boost::system::error_category error_base<T>::addrinfo_ecat(
    boost::system::error_code::new_category(&error_base<T>::addrinfo_ed,
      &error_base<T>::addrinfo_md, &error_base<T>::addrinfo_wmd));

template <typename T>
int error_base<T>::addrinfo_ed(const boost::system::error_code& ec)
{
  return EOTHER;
}

template <typename T>
std::string error_base<T>::addrinfo_md(const boost::system::error_code& ec)
{
  if (ec == error_base<T>::service_not_found)
    return "Service not found";
  if (ec == error_base<T>::socket_type_not_supported)
    return "Socket type not supported";
  return "EINVAL";
}

template <typename T>
boost::system::wstring_t error_base<T>::addrinfo_wmd(
    const boost::system::error_code& ec)
{
  if (ec == error_base<T>::service_not_found)
    return L"Service not found";
  if (ec == error_base<T>::socket_type_not_supported)
    return L"Socket type not supported";
  return L"EINVAL";
}

#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

template <typename T>
boost::system::error_category error_base<T>::misc_ecat(
    boost::system::error_code::new_category(&error_base<T>::misc_ed,
      &error_base<T>::misc_md, &error_base<T>::misc_wmd));

template <typename T>
int error_base<T>::misc_ed(const boost::system::error_code& ec)
{
  return EOTHER;
}

template <typename T>
std::string error_base<T>::misc_md(const boost::system::error_code& ec)
{
  if (ec == error_base<T>::already_open)
    return "Already open";
  if (ec == error_base<T>::eof)
    return "End of file";
  if (ec == error_base<T>::not_found)
    return "Element not found";
  return "EINVAL";
}

template <typename T>
boost::system::wstring_t error_base<T>::misc_wmd(
    const boost::system::error_code& ec)
{
  if (ec == error_base<T>::eof)
    return L"End of file";
  if (ec == error_base<T>::not_found)
    return L"Element not found";
  return L"EINVAL";
}

template <typename T>
boost::system::error_category error_base<T>::ssl_ecat(
    boost::system::error_code::new_category(&error_base<T>::ssl_ed,
      &error_base<T>::ssl_md, &error_base<T>::ssl_wmd));

template <typename T>
int error_base<T>::ssl_ed(const boost::system::error_code& ec)
{
  return EOTHER;
}

template <typename T>
std::string error_base<T>::ssl_md(const boost::system::error_code& ec)
{
  return "SSL error";
}

template <typename T>
boost::system::wstring_t error_base<T>::ssl_wmd(
    const boost::system::error_code& ec)
{
  return L"SSL error";
}
EOF

sub copy_source_file
{
  my ($from, $to) = @_;

  # Ensure the output directory exists.
  my $dir = $to;
  $dir =~ s/[^\/]*$//;
  mkpath($dir);

  # First determine whether the file makes any use of asio::thread.
  my $uses_asio_thread = source_contains_asio_thread_usage($from);

  # Check whether the file includes error handling header files.
  my $includes_error_code = source_contains_asio_error_code_include($from);
  my $includes_system_error = source_contains_asio_system_error_include($from);
  my $includes_boostify_ecats = source_contains_boostify_error_categories($from);

  # Open the files.
  open(my $input, "<$from") or die("Can't open $from for reading");
  open(my $output, ">$to") or die("Can't open $to for writing");

  # Copy the content.
  my $lineno = 1;
  while (my $line = <$input>)
  {
    chomp($line);

    # Unconditional replacements.
    $line =~ s/[\\@]ref boost_bind/boost::bind()/g;
    if ($from =~ /.*\.txt$/)
    {
      $line =~ s/[\\@]ref async_read/boost::asio::async_read()/g;
      $line =~ s/[\\@]ref async_write/boost::asio::async_write()/g;
    }

    # Conditional replacements.
    if ($line =~ /^namespace asio {/)
    {
      print_line($output, "namespace boost {", $from, $lineno);
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /^} \/\/ namespace asio$/)
    {
      print_line($output, $line, $from, $lineno);
      print_line($output, "} // namespace boost", $from, $lineno);
    }
    elsif ($line =~ /^(# *include )[<"](asio\.hpp)[>"]$/)
    {
      print_line($output, $1 . "<boost/" . $2 . ">", $from, $lineno);
      if ($uses_asio_thread)
      {
        print_line($output, $1 . "<boost/thread.hpp>", $from, $lineno);
      }
    }
    elsif ($line =~ /^(# *include )[<"](asio\/detail\/pop_options\.hpp)[>"]$/)
    {
      if ($includes_boostify_ecats)
      {
        $includes_boostify_ecats = 0;
        print_line($output, $1 . "<boost/cerrno.hpp>", $from, $lineno);
      }
      if ($includes_error_code)
      {
        $includes_error_code = 0;
        print_line($output, $1 . "<boost/system/error_code.hpp>", $from, $lineno);
      }
      if ($includes_system_error)
      {
        $includes_system_error = 0;
        print_line($output, $1 . "<boost/system/system_error.hpp>", $from, $lineno);
      }
      print_line($output, $1 . "<boost/" . $2 . ">", $from, $lineno);
    }
    elsif ($line =~ /# *include <cerrno>/)
    {
      if ($includes_boostify_ecats)
      {
        # Line is removed.
      }
      else
      {
        print_line($output, $line, $from, $lineno);
      }
    }
    elsif ($line =~ /# *include [<"]asio\/thread\.hpp[>"]/)
    {
      # Line is removed.
    }
    elsif ($line =~ /# *include [<"]asio\/error_code\.hpp[>"]/)
    {
      # Line is removed.
    }
    elsif ($line =~ /# *include [<"]asio\/impl\/error_code\.ipp[>"]/)
    {
      # Line is removed.
    }
    elsif ($line =~ /# *include [<"]asio\/system_error\.hpp[>"]/)
    {
      # Line is removed.
    }
    elsif ($line =~ /^(# *include )[<"](asio\/.*)[>"](.*)$/)
    {
      print_line($output, $1 . "<boost/" . $2 . ">" . $3, $from, $lineno);
    }
    elsif ($line =~ /ASIO_/ && !($line =~ /BOOST_ASIO_/))
    {
      $line =~ s/ASIO_/BOOST_ASIO_/g;
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /asio::thread/)
    {
      $line =~ s/asio::thread/boost::thread/g;
      if (!($line =~ /boost::asio::/))
      {
        $line =~ s/asio::/boost::asio::/g;
      }
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /^( *)thread( .*)$/)
    {
      if (!($line =~ /boost::asio::/))
      {
        $line =~ s/asio::/boost::asio::/g;
      }
      print_line($output, $1 . "boost::thread" . $2, $from, $lineno);
    }
    elsif ($line =~ /boostify: error category declarations go here/)
    {
      print($output $error_cat_decls);
    }
    elsif ($line =~ /boostify: error category definitions go here/)
    {
      print($output $error_cat_defns);
    }
    elsif ($line =~ /asio::/ && !($line =~ /boost::asio::/))
    {
      $line =~ s/asio::error_code/boost::system::error_code/g;
      $line =~ s/asio::system_error/boost::system::system_error/g;
      $line =~ s/asio::native_ecat/boost::system::native_ecat/g;
      if ($from =~ /error\.hpp/)
      {
        $line =~ s/asio::netdb_ecat/asio::detail::error_base<T>::netdb_ecat/g;
        $line =~ s/asio::addrinfo_ecat/asio::detail::error_base<T>::addrinfo_ecat/g;
        $line =~ s/asio::misc_ecat/asio::detail::error_base<T>::misc_ecat/g;
        $line =~ s/asio::ssl_ecat/asio::detail::error_base<T>::ssl_ecat/g;
      }
      else
      {
        $line =~ s/asio::netdb_ecat/asio::error::netdb_ecat/g;
        $line =~ s/asio::addrinfo_ecat/asio::error::addrinfo_ecat/g;
        $line =~ s/asio::misc_ecat/asio::error::misc_ecat/g;
        $line =~ s/asio::ssl_ecat/asio::error::ssl_ecat/g;
      }
      $line =~ s/asio::/boost::asio::/g;
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /using namespace asio/)
    {
      $line =~ s/using namespace asio/using namespace boost::asio/g;
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /asio_handler_alloc_helpers/)
    {
      $line =~ s/asio_handler_alloc_helpers/boost_asio_handler_alloc_helpers/g;
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /asio_handler_dispatch_helpers/)
    {
      $line =~ s/asio_handler_invoke_helpers/boost_asio_handler_invoke_helpers/g;
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /[\\@]ref boost_bind/)
    {
      $line =~ s/[\\@]ref boost_bind/boost::bind()/g;
      print_line($output, $line, $from, $lineno);
    }
    else
    {
      print_line($output, $line, $from, $lineno);
    }
    ++$lineno;
  }

  # Ok, we're done.
  close($input);
  close($output);
}

sub copy_include_files
{
  my @dirs = (
      "include",
      "include/asio",
      "include/asio/detail",
      "include/asio/impl",
      "include/asio/ip",
      "include/asio/ip/detail",
      "include/asio/ssl",
      "include/asio/ssl/detail");

  foreach my $dir (@dirs)
  {
    our $boost_dir;
    my @files = ( glob("$dir/*.hpp"), glob("$dir/*.ipp") );
    foreach my $file (@files)
    {
      if ($file ne "include/asio/thread.hpp"
          and $file ne "include/asio/error_code.hpp"
          and $file ne "include/asio/system_error.hpp"
          and $file ne "include/asio/impl/error_code.ipp")
      {
        my $from = $file;
        my $to = $file;
        $to =~ s/^include\//$boost_dir\/boost\//;
        copy_source_file($from, $to);
      }
    }
  }
}

sub create_lib_directory
{
  my @dirs = (
      "doc",
      "example",
      "test");

  our $boost_dir;
  foreach my $dir (@dirs)
  {
    mkpath("$boost_dir/libs/asio/$dir");
  }
}

sub copy_unit_tests
{
  my @dirs = (
      "src/tests/unit",
      "src/tests/unit/ip",
      "src/tests/unit/ssl");

  our $boost_dir;
  foreach my $dir (@dirs)
  {
    my @files = ( glob("$dir/*.*pp"), glob("$dir/Jamfile*") );
    foreach my $file (@files)
    {
      if ($file ne "src/tests/unit/thread.cpp"
          and $file ne "src/tests/unit/error_handler.cpp")
      {
        my $from = $file;
        my $to = $file;
        $to =~ s/^src\/tests\/unit\//$boost_dir\/libs\/asio\/test\//;
        copy_source_file($from, $to);
      }
    }
  }
}

sub copy_examples
{
  my @dirs = (
      "src/examples/allocation",
      "src/examples/buffers",
      "src/examples/chat",
      "src/examples/echo",
      "src/examples/http/client",
      "src/examples/http/doc_root",
      "src/examples/http/server",
      "src/examples/http/server2",
      "src/examples/http/server3",
      "src/examples/iostreams",
      "src/examples/multicast",
      "src/examples/serialization",
      "src/examples/services",
      "src/examples/socks4",
      "src/examples/ssl",
      "src/examples/timeouts",
      "src/examples/timers",
      "src/examples/tutorial",
      "src/examples/tutorial/daytime1",
      "src/examples/tutorial/daytime2",
      "src/examples/tutorial/daytime3",
      "src/examples/tutorial/daytime4",
      "src/examples/tutorial/daytime5",
      "src/examples/tutorial/daytime6",
      "src/examples/tutorial/daytime7",
      "src/examples/tutorial/timer1",
      "src/examples/tutorial/timer2",
      "src/examples/tutorial/timer3",
      "src/examples/tutorial/timer4",
      "src/examples/tutorial/timer5");

  our $boost_dir;
  foreach my $dir (@dirs)
  {
    my @files = (
        glob("$dir/*.*pp"),
        glob("$dir/*.html"),
        glob("$dir/Jamfile*"),
        glob("$dir/*.pem"),
        glob("$dir/README*"),
        glob("$dir/*.txt"));
    foreach my $file (@files)
    {
      my $from = $file;
      my $to = $file;
      $to =~ s/^src\/examples\//$boost_dir\/libs\/asio\/example\//;
      copy_source_file($from, $to);
    }
  }
}

determine_boost_dir();
copy_include_files();
create_lib_directory();
copy_unit_tests();
copy_examples();
