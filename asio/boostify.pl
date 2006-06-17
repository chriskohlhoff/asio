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
    print("Warning: $from:$lineno: output >80 characters wide.\n");
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

sub copy_source_file
{
  my ($from, $to) = @_;

  # Ensure the output directory exists.
  my $dir = $to;
  $dir =~ s/[^\/]*$//;
  mkpath($dir);

  # First determine whether the file makes any use of asio::thread.
  my $uses_asio_thread = source_contains_asio_thread_usage($from);

  # Open the files.
  open(my $input, "<$from") or die("Can't open $from for reading");
  open(my $output, ">$to") or die("Can't open $to for writing");

  # Copy the content.
  my $lineno = 1;
  while (my $line = <$input>)
  {
    chomp($line);
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
    elsif ($line =~ /# *include [<"]asio\/thread\.hpp[>"]/)
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
    elsif ($line =~ /asio::/ && !($line =~ /boost::asio::/))
    {
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
      if ($file ne "include/asio/thread.hpp")
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
      if ($file ne "src/tests/unit/thread_test.cpp")
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
      "src/examples/http/server",
      "src/examples/iostreams",
      "src/examples/multicast",
      "src/examples/serialization",
      "src/examples/services",
      "src/examples/ssl",
      "src/examples/timeouts",
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

sub copy_docs
{
  my @dirs = (
      "src/doc/boost",
      "src/doc/concepts",
      "src/doc/design");

  our $boost_dir;
  foreach my $dir (@dirs)
  {
    my @files = (
        glob("$dir/*.*pp"),
        glob("$dir/*.dox"),
        glob("$dir/*.dot"),
        glob("$dir/*.htm"),
        glob("$dir/*.qbk"),
        glob("$dir/*.txt"),
        glob("$dir/*.v2"));
    foreach my $file (@files)
    {
      my $from = $file;
      my $to = $file;
      if ($to =~ /src\/doc\/boost\/.*\.qbk/)
      {
        $to =~ s/^src\/doc\/boost\//$boost_dir\/libs\/asio\/doc\//;
      }
      elsif ($to =~ /src\/doc\/boost\/.*\.v2/)
      {
        $to =~ s/^src\/doc\/boost\//$boost_dir\/libs\/asio\/doc\//;
      }
      elsif ($to =~ /src\/doc\/boost/)
      {
        $to =~ s/^src\/doc\/boost\//$boost_dir\/libs\/asio\/doc\/doxygen\//;
      }
      elsif ($to =~ /src\/doc\/design/)
      {
        $to =~ s/^src\/doc\/design\//$boost_dir\/libs\/asio\/doc\/doxygen\/design\//;
      }
      else
      {
        $to =~ s/^src\/doc\//$boost_dir\/libs\/asio\/doc\//;
      }
      copy_source_file($from, $to);
    }
  }

  copy_source_file("src/doc/boost/asio.css",
      "$boost_dir/libs/asio/doc/asio.css");
  copy_source_file("src/doc/boost/asio.css",
      "$boost_dir/libs/asio/doc/design/asio.css");
  copy_source_file("src/doc/boost/asio.css",
      "$boost_dir/libs/asio/doc/doxygen/asio.css");
  copy_source_file("src/doc/boost/asio.css",
      "$boost_dir/libs/asio/doc/examples/asio.css");
  copy_source_file("src/doc/boost/asio.css",
      "$boost_dir/libs/asio/doc/reference/asio.css");
  copy_source_file("src/doc/boost/asio.css",
      "$boost_dir/libs/asio/doc/tutorial/asio.css");
  copy_source_file("src/doc/external.doxytags",
      "$boost_dir/libs/asio/doc/doxygen/external.doxytags");
}

sub create_root_html
{
  our $boost_dir;
  open(my $output, ">$boost_dir/libs/asio/index.html")
    or die("Can't open $boost_dir/libs/asio/index.html for writing");
  print($output "<html>\n");
  print($output "<head>\n");
  print($output "<meta http-equiv=\"refresh\"");
  print($output " content=\"0; URL=doc/html/index.html\">\n");
  print($output "</head>\n");
  print($output "<body>\n");
  print($output "Automatic redirection failed, please go to\n");
  print($output "<a href=\"doc/html/index.html\">doc/html/index.html</a>\n");
  print($output "</body>\n");
  print($output "</html>\n");
  close($output);
}

sub create_doc_html
{
  our $boost_dir;
  mkdir("$boost_dir/libs/asio/doc");
  open(my $output, ">$boost_dir/libs/asio/doc/index.html")
    or die("Can't open $boost_dir/libs/asio/doc/index.html for writing");
  print($output "<html>\n");
  print($output "<head>\n");
  print($output "<meta http-equiv=\"refresh\"");
  print($output " content=\"0; URL=html/index.html\">\n");
  print($output "</head>\n");
  print($output "<body>\n");
  print($output "Automatic redirection failed, please go to\n");
  print($output "<a href=\"html/index.html\">html/index.html</a>\n");
  print($output "</body>\n");
  print($output "</html>\n");
  close($output);
}

sub execute_doxygen
{
  our $boost_dir;
  chdir("$boost_dir/libs/asio/doc/doxygen");
  system("doxygen reference.dox");
  system("doxygen design.dox");
  system("doxygen examples.dox");
  system("doxygen tutorial.dox");
  unlink("asio.doxytags");
  unlink("design.doxytags");
  chdir("..");
  my @rm_files = (
      glob("*/*.map"),
      glob("*/*.md5"),
      glob("*/doxygen.png"));
  foreach my $rm_file (@rm_files)
  {
    unlink($rm_file);
  }
}

determine_boost_dir();
copy_include_files();
create_lib_directory();
copy_unit_tests();
copy_examples();
copy_docs();
create_root_html();
create_doc_html();
execute_doxygen();
