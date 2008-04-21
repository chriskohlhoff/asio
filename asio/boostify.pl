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

sub source_contains_asio_include
{
  my ($from) = @_;

  # Open the input file.
  open(my $input, "<$from") or die("Can't open $from for reading");

  # Check file for inclusion of asio.hpp.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /# *include [<"]asio\.hpp[>"]/)
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

my $error_cat_defns = <<"EOF";
inline const boost::system::error_category& get_system_category()
{
  return boost::system::get_system_category();
}

#if !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

namespace detail {

class netdb_category : public boost::system::error_category
{
public:
  const char* name() const
  {
    return "asio.netdb";
  }

  std::string message(int value) const
  {
    if (value == error::host_not_found)
      return "Host not found (authoritative)";
    if (value == error::host_not_found_try_again)
      return "Host not found (non-authoritative), try again later";
    if (value == error::no_data)
      return "The query is valid, but it does not have associated data";
    if (value == error::no_recovery)
      return "A non-recoverable error occurred during database lookup";
    return "asio.netdb error";
  }
};

} // namespace detail

inline const boost::system::error_category& get_netdb_category()
{
  static detail::netdb_category instance;
  return instance;
}

namespace detail {

class addrinfo_category : public boost::system::error_category
{
public:
  const char* name() const
  {
    return "asio.addrinfo";
  }

  std::string message(int value) const
  {
    if (value == error::service_not_found)
      return "Service not found";
    if (value == error::socket_type_not_supported)
      return "Socket type not supported";
    return "asio.addrinfo error";
  }
};

} // namespace detail

inline const boost::system::error_category& get_addrinfo_category()
{
  static detail::addrinfo_category instance;
  return instance;
}

#else // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

inline const boost::system::error_category& get_netdb_category()
{
  return get_system_category();
}

inline const boost::system::error_category& get_addrinfo_category()
{
  return get_system_category();
}

#endif // !defined(BOOST_WINDOWS) && !defined(__CYGWIN__)

namespace detail {

class misc_category : public boost::system::error_category
{
public:
  const char* name() const
  {
    return "asio.misc";
  }

  std::string message(int value) const
  {
    if (value == error::already_open)
      return "Already open";
    if (value == error::eof)
      return "End of file";
    if (value == error::not_found)
      return "Element not found";
    if (value == error::fd_set_failure)
      return "The descriptor does not fit into the select call's fd_set";
    return "asio.misc error";
  }
};

} // namespace detail

inline const boost::system::error_category& get_misc_category()
{
  static detail::misc_category instance;
  return instance;
}

namespace detail {

class ssl_category : public boost::system::error_category
{
public:
  const char* name() const
  {
    return "asio.ssl";
  }

  std::string message(int) const
  {
    return "asio.ssl error";
  }
};

} // namespace detail

inline const boost::system::error_category& get_ssl_category()
{
  static detail::ssl_category instance;
  return instance;
}

static const boost::system::error_category& system_category
  = boost::asio::error::get_system_category();
static const boost::system::error_category& netdb_category
  = boost::asio::error::get_netdb_category();
static const boost::system::error_category& addrinfo_category
  = boost::asio::error::get_addrinfo_category();
static const boost::system::error_category& misc_category
  = boost::asio::error::get_misc_category();
static const boost::system::error_category& ssl_category
  = boost::asio::error::get_ssl_category();

} // namespace error
} // namespace asio

namespace system {

template<> struct is_error_code_enum<boost::asio::error::basic_errors>
{
  static const bool value = true;
};

template<> struct is_error_code_enum<boost::asio::error::netdb_errors>
{
  static const bool value = true;
};

template<> struct is_error_code_enum<boost::asio::error::addrinfo_errors>
{
  static const bool value = true;
};

template<> struct is_error_code_enum<boost::asio::error::misc_errors>
{
  static const bool value = true;
};

template<> struct is_error_code_enum<boost::asio::error::ssl_errors>
{
  static const bool value = true;
};

} // namespace system

namespace asio {
namespace error {
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
  my $includes_asio = source_contains_asio_include($from);

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
        $uses_asio_thread = 0;
      }
    }
    elsif ($line =~ /^(# *include )[<"]boost\/.*[>"].*$/)
    {
      if (!$includes_asio && $uses_asio_thread)
      {
        print_line($output, $1 . "<boost/thread.hpp>", $from, $lineno);
        $uses_asio_thread = 0;
      }
      print_line($output, $line, $from, $lineno);
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
    elsif ($line =~ /boostify: error category definitions go here/)
    {
      print($output $error_cat_defns);
    }
    elsif ($line =~ /asio::/ && !($line =~ /boost::asio::/))
    {
      $line =~ s/asio::error_code/boost::system::error_code/g;
      $line =~ s/asio::system_error/boost::system::system_error/g;
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
    elsif ($line =~ /asio_handler_invoke_helpers/)
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
      "include/asio/local",
      "include/asio/posix",
      "include/asio/ssl",
      "include/asio/ssl/detail",
      "include/asio/windows");

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
      "src/tests/unit/local",
      "src/tests/unit/posix",
      "src/tests/unit/ssl",
      "src/tests/unit/windows");

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
      "src/examples/invocation",
      "src/examples/iostreams",
      "src/examples/multicast",
      "src/examples/nonblocking",
      "src/examples/porthopper",
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
