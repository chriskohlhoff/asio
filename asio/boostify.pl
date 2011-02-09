#!/usr/bin/perl -w

use strict;
use File::Path;

our $boost_dir = "boostified";

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

  my $is_asio_hpp = 0;
  $is_asio_hpp = 1 if ($from =~ /asio\.hpp/);

  my $needs_doc_link = 0;
  $needs_doc_link = 1 if ($is_asio_hpp);

  my $is_error_hpp = 0;
  $is_error_hpp = 1 if ($from =~ /asio\/error\.hpp/);

  my $is_qbk = 0;
  $is_qbk = 1 if ($from =~ /.qbk$/);

  my $is_xsl = 0;
  $is_xsl = 1 if ($from =~ /.xsl$/);

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
    if ($line =~ /asio_detail_posix_thread_function/)
    {
      $line =~ s/asio_detail_posix_thread_function/boost_asio_detail_posix_thread_function/g;
    }
    if ($line =~ /ASIO_/ && !($line =~ /BOOST_ASIO_/))
    {
      $line =~ s/ASIO_/BOOST_ASIO_/g;
    }

    # Extra replacements for quickbook and XSL source only.
    if ($is_qbk || $is_xsl)
    {
      $line =~ s/asio\.examples/boost_asio.examples/g;
      $line =~ s/asio\.history/boost_asio.history/g;
      $line =~ s/asio\.index/boost_asio.index/g;
      $line =~ s/asio\.overview/boost_asio.overview/g;
      $line =~ s/asio\.reference/boost_asio.reference/g;
      $line =~ s/asio\.tutorial/boost_asio.tutorial/g;
      $line =~ s/asio\.using/boost_asio.using/g;
      $line =~ s/Asio/Boost.Asio/g;
      $line =~ s/changes made in each release/changes made in each Boost release/g;
      $line =~ s/\[\$/[\$boost_asio\//g;
      $line =~ s/\[@\.\.\/src\/examples/[\@boost_asio\/example/g;
      $line =~ s/include\/asio/boost\/asio/g;
      $line =~ s/\^asio/^boost\/asio/g;
    }

    # Conditional replacements.
    if ($line =~ /^( *)namespace asio {/)
    {
      if ($is_qbk)
      {
        print_line($output, $1 . "namespace boost { namespace asio {", $from, $lineno);
      }
      else
      {
        print_line($output, $1 . "namespace boost {", $from, $lineno);
        print_line($output, $line, $from, $lineno);
      }
    }
    elsif ($line =~ /^( *)} \/\/ namespace asio$/)
    {
      if ($is_qbk)
      {
        print_line($output, $1 . "} } // namespace boost::asio", $from, $lineno);
      }
      else
      {
        print_line($output, $line, $from, $lineno);
        print_line($output, $1 . "} // namespace boost", $from, $lineno);
      }
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
    elsif ($line =~ /# *include [<"]asio\/thread\.hpp[>"]/)
    {
      # Line is removed.
    }
    elsif ($line =~ /(# *include )[<"]asio\/error_code\.hpp[>"]/)
    {
      if ($is_asio_hpp)
      {
        # Line is removed.
      }
      else
      {
        print_line($output, $1 . "<boost/cerrno.hpp>", $from, $lineno) if ($is_error_hpp);
        print_line($output, $1 . "<boost/system/error_code.hpp>", $from, $lineno);
      }
    }
    elsif ($line =~ /# *include [<"]asio\/impl\/error_code\.[hi]pp[>"]/)
    {
      # Line is removed.
    }
    elsif ($line =~ /(# *include )[<"]asio\/system_error\.hpp[>"]/)
    {
      if ($is_asio_hpp)
      {
        # Line is removed.
      }
      else
      {
        print_line($output, $1 . "<boost/system/system_error.hpp>", $from, $lineno);
      }
    }
    elsif ($line =~ /(^.*# *include )[<"](asio\/.*)[>"](.*)$/)
    {
      print_line($output, $1 . "<boost/" . $2 . ">" . $3, $from, $lineno);
    }
    elsif ($line =~ /#.*defined\(.*ASIO_HAS_STD_SYSTEM_ERROR\)$/)
    {
      # Line is removed.
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
    elsif ($line =~ /^( *)thread( .*)$/ && !$is_qbk)
    {
      if (!($line =~ /boost::asio::/))
      {
        $line =~ s/asio::/boost::asio::/g;
      }
      print_line($output, $1 . "boost::thread" . $2, $from, $lineno);
    }
    elsif ($line =~ /namespace std {/)
    {
      print_line($output, "namespace boost {", $from, $lineno);
      print_line($output, "namespace system {", $from, $lineno);
    }
    elsif ($line =~ /} \/\/ namespace std/)
    {
      print_line($output, "} // namespace system", $from, $lineno);
      print_line($output, "} // namespace boost", $from, $lineno);
    }
    elsif ($line =~ /asio::/ && !($line =~ /boost::asio::/))
    {
      $line =~ s/asio::error_code/boost::system::error_code/g;
      $line =~ s/asio::error_category/boost::system::error_category/g;
      $line =~ s/asio::system_category/boost::system::system_category/g;
      $line =~ s/asio::system_error/boost::system::system_error/g;
      $line =~ s/asio::/boost::asio::/g if !$is_xsl;
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
    elsif ($line =~ /( *)\[category template\]/)
    {
      print_line($output, $1 . "[authors [Kohlhoff, Christopher]]", $from, $lineno);
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /^$/ && $needs_doc_link)
    {
      $needs_doc_link = 0;
      print_line($output, "//  See www.boost.org/libs/asio for documentation.", $from, $lineno);
      print_line($output, "//", $from, $lineno);
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
      "include/asio/detail/impl",
      "include/asio/impl",
      "include/asio/ip",
      "include/asio/ip/impl",
      "include/asio/ip/detail",
      "include/asio/ip/detail/impl",
      "include/asio/local",
      "include/asio/local/detail",
      "include/asio/local/detail/impl",
      "include/asio/posix",
      "include/asio/ssl",
      "include/asio/ssl/detail",
      "include/asio/ssl/impl",
      "include/asio/windows");

  foreach my $dir (@dirs)
  {
    our $boost_dir;
    my @files = ( glob("$dir/*.hpp"), glob("$dir/*.ipp"), glob("$dir/*cpp") );
    foreach my $file (@files)
    {
      if ($file ne "include/asio/thread.hpp"
          and $file ne "include/asio/error_code.hpp"
          and $file ne "include/asio/system_error.hpp"
          and $file ne "include/asio/impl/error_code.hpp"
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
      "src/tests/unit/archetypes",
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
          and $file ne "src/tests/unit/error_handler.cpp"
          and $file ne "src/tests/unit/unit_test.cpp")
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
      "src/examples/http/server4",
      "src/examples/icmp",
      "src/examples/invocation",
      "src/examples/iostreams",
      "src/examples/local",
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
      "src/examples/tutorial/timer5",
      "src/examples/windows");

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

sub copy_doc
{
  our $boost_dir;
  my @files = (
      "src/doc/asio.qbk",
      "src/doc/examples.qbk",
      "src/doc/reference.xsl",
      "src/doc/tutorial.xsl",
      glob("src/doc/overview/*.qbk"),
      glob("src/doc/requirements/*.qbk"));
  foreach my $file (@files)
  {
    my $from = $file;
    my $to = $file;
    $to =~ s/^src\/doc\//$boost_dir\/libs\/asio\/doc\//;
    copy_source_file($from, $to);
  }
}

copy_include_files();
create_lib_directory();
copy_unit_tests();
copy_examples();
copy_doc();
