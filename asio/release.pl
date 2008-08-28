#!/usr/bin/perl -w

use strict;
use Cwd 'abs_path';
use Date::Format;
use File::Path;
use File::Copy;

our $version_major;
our $version_minor;
our $version_sub_minor;

sub print_usage_and_exit
{
  print("Usage: ./release.pl <version>\n");
  print("  Example: ./release.pl 1.2.0\n");
  exit(1);
}

sub determine_version($)
{
  my $version_string = shift;
  if ($version_string =~ /^([0-9]+)\.([0-9]+)\.([0-9]+)$/)
  {
    our $version_major = $1;
    our $version_minor = $2;
    our $version_sub_minor = $3;
  }
  else
  {
    print_usage_and_exit();
  }
}

sub update_configure_ac
{
  # Open the files.
  my $from = "configure.ac";
  my $to = $from . ".tmp";
  open(my $input, "<$from") or die("Can't open $from for reading");
  open(my $output, ">$to") or die("Can't open $to for writing");

  # Copy the content.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /^AC_INIT\(asio.*\)$/)
    {
      $line = "AC_INIT(asio, [";
      $line .= "$version_major.$version_minor.$version_sub_minor";
      $line .= "])";
    }
    print($output "$line\n");
  }

  # Close the files and move the temporary output into position.
  close($input);
  close($output);
  move($to, $from);
  unlink($to);
}

sub update_readme
{
  # Open the files.
  my $from = "README";
  my $to = $from . ".tmp";
  open(my $input, "<$from") or die("Can't open $from for reading");
  open(my $output, ">$to") or die("Can't open $to for writing");

  # Copy the content.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /^asio version/)
    {
      $line = "asio version ";
      $line .= "$version_major.$version_minor.$version_sub_minor";
    }
    elsif ($line =~ /^Released/)
    {
      my @time = localtime;
      $line = "Released " . strftime("%A, %d %B %Y", @time);
    }
    print($output "$line\n");
  }

  # Close the files and move the temporary output into position.
  close($input);
  close($output);
  move($to, $from);
  unlink($to);
}

sub update_asio_version_hpp
{
  # Open the files.
  my $from = "include/asio/version.hpp";
  my $to = $from . ".tmp";
  open(my $input, "<$from") or die("Can't open $from for reading");
  open(my $output, ">$to") or die("Can't open $to for writing");

  # Copy the content.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /^#define ASIO_VERSION /)
    {
      my $version = $version_major * 100000;
      $version += $version_minor * 100;
      $version += $version_sub_minor + 0;
      $line = "#define ASIO_VERSION " . $version;
      $line .= " // $version_major.$version_minor.$version_sub_minor";
    }
    print($output "$line\n");
  }

  # Close the files and move the temporary output into position.
  close($input);
  close($output);
  move($to, $from);
  unlink($to);
}

sub update_boost_asio_version_hpp
{
  # Open the files.
  my $from = "../boost/boost/asio/version.hpp";
  my $to = $from . ".tmp";
  open(my $input, "<$from") or die("Can't open $from for reading");
  open(my $output, ">$to") or die("Can't open $to for writing");

  # Copy the content.
  while (my $line = <$input>)
  {
    chomp($line);
    if ($line =~ /^#define BOOST_ASIO_VERSION /)
    {
      my $version = $version_major * 100000;
      $version += $version_minor * 100;
      $version += $version_sub_minor + 0;
      $line = "#define BOOST_ASIO_VERSION " . $version;
      $line .= " // $version_major.$version_minor.$version_sub_minor";
    }
    print($output "$line\n");
  }

  # Close the files and move the temporary output into position.
  close($input);
  close($output);
  move($to, $from);
  unlink($to);
}

sub build_asio_doc
{
  $ENV{BOOST_ROOT} = abs_path("../boost");
  system("rm -rf doc");
  my $bjam = abs_path(glob("../boost/tools/jam/src/bin.*/bjam"));
  chdir("src/doc");
  system("$bjam");
  chdir("../..");
  mkdir("doc");
  system("cp -vR src/doc/html/* doc");
}

sub make_asio_packages
{
  system("./autogen.sh");
  system("./configure");
  system("make dist");
}

sub create_manifests
{
  system("tar tfz asio-$version_major.$version_minor.$version_sub_minor.tar.gz"
         . " | sed -e 's/^[^\\/]*//' | sort > asio.manifest");
}

(scalar(@ARGV) == 1) or print_usage_and_exit();
determine_version($ARGV[0]);
update_configure_ac();
update_readme();
update_asio_version_hpp();
update_boost_asio_version_hpp();
build_asio_doc();
make_asio_packages();
create_manifests();
