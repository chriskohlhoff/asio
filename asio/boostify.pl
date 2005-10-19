#!/usr/bin/perl -w

use strict;
use File::Path;

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

sub copy_source_file
{
  my ($from, $to) = @_;

  # Ensure the output directory exists.
  my $dir = $to;
  $dir =~ s/[^\/]*$//;
  mkpath($dir);

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
    elsif ($line =~ /^} \/\/ namespace asio/)
    {
      print_line($output, $line, $from, $lineno);
      print_line($output, "} // namespace boost", $from, $lineno);
    }
    elsif ($line =~ /^(#include ")(asio[.\/].*)$/)
    {
      print_line($output, $1 . "boost/" . $2, $from, $lineno);
    }
    elsif ($line =~ /^(#include )<(boost\/[^>]*)>$/)
    {
      print_line($output, $1 . "\"" . $2 . "\"", $from, $lineno);
    }
    elsif ($line =~ /ASIO_/)
    {
      $line =~ s/ASIO_/BOOST_ASIO_/g;
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /asio::/)
    {
      $line =~ s/asio::/boost::asio::/g;
      print_line($output, $line, $from, $lineno);
    }
    elsif ($line =~ /using namespace asio/)
    {
      $line =~ s/using namespace asio/using namespace boost::asio/g;
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
      "include/asio/ipv4",
      "include/asio/ipv4/detail",
      "include/asio/ssl",
      "include/asio/ssl/detail");

  foreach my $dir (@dirs)
  {
    my @files = glob("$dir/*.hpp");
    foreach my $file (@files)
    {
      my $from = $file;
      my $to = $file;
      $to =~ s/^include\//boost\/boost\//;
      copy_source_file($from, $to);
    }
  }
}

sub create_lib_directory
{
  my @dirs = (
      "doc",
      "example",
      "test");

  foreach my $dir (@dirs)
  {
    mkpath("boost/libs/asio/$dir");
  }
}

sub copy_unit_tests
{
  my @dirs = (
      "src/tests/unit",
      "src/tests/unit/ipv4",
      "src/tests/unit/ssl");

  foreach my $dir (@dirs)
  {
    my @files = ( glob("$dir/*.*pp"), glob("$dir/Jamfile*") );
    foreach my $file (@files)
    {
      my $from = $file;
      my $to = $file;
      $to =~ s/^src\/tests\/unit\//boost\/libs\/asio\/test\//;
      copy_source_file($from, $to);
    }
  }
}

sub create_root_html
{
  open(my $output, ">boost/libs/asio/index.html")
    or die("Can't open boost/libs/asio/index.html for writing");
  print($output "<html>\n");
  print($output "<head>\n");
  print($output "<meta http-equiv=\"refresh\"");
  print($output " content=\"0; URL=doc/index.html\">\n");
  print($output "</head>\n");
  print($output "<body>\n");
  print($output "Automatic redirection failed, please go to\n");
  print($output "<a href=\"doc/index.html\">doc/index.html</a>\n");
  print($output "</body>\n");
  print($output "</html>\n");
  close($output);
}

copy_include_files();
create_lib_directory();
copy_unit_tests();
create_root_html();
