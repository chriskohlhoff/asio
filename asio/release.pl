#!/usr/bin/perl -w

use strict;
use Cwd qw(abs_path getcwd);
use Date::Format;
use File::Path;
use File::Copy;

our $version_major;
our $version_minor;
our $version_sub_minor;
our $asio_name;
our $boost_asio_name;

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

    our $asio_name = "asio";
    $asio_name .= "-$version_major";
    $asio_name .= ".$version_minor";
    $asio_name .= ".$version_sub_minor";

    our $boost_asio_name = "boost_asio";
    $boost_asio_name .= "_$version_major";
    $boost_asio_name .= "_$version_minor";
    $boost_asio_name .= "_$version_sub_minor";
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
      $line = "Released " . strftime("%A, %d %B %Y", @time) . ".";
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
  my $bjam = abs_path(glob("../boost/bjam"));
  chdir("src/doc");
  system("$bjam clean");
  system("rm -rf html");
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

sub build_boost_asio_doc
{
  my $cwd = getcwd();
  my $bjam = abs_path(glob("../boost/bjam"));
  chdir("../boost/doc");
  system("$bjam clean");
  system("rm -rf html/boost_asio");
  system("$bjam asio");
  chdir($cwd);
}

our $boost_asio_readme = <<"EOF";
Copy the `boost', `doc' and `libs' directories into an existing boost 1.33.0,
1.33.1, 1.34, 1.34.1, 1.35 or 1.36 distribution.

Before using Boost.Asio, the Boost.System library needs to be built. This can
be done by running bjam in the libs/system/build directory. Consult the Boost
Getting Started page (http://www.boost.org/more/getting_started.html) for more
information on how to build the Boost libraries.
EOF

our $boost_system_jamfile = <<"EOF";
# Boost System Library Build Jamfile

# (C) Copyright Beman Dawes 2002, 2006

# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or www.boost.org/LICENSE_1_0.txt)

# See library home page at http://www.boost.org/libs/system

subproject libs/system/build ;

SOURCES = error_code ;

lib boost_system
     : ../src/$(SOURCES).cpp
     : # build requirements
      <define>BOOST_SYSTEM_STATIC_LINK
      <sysinclude>$(BOOST_AUX_ROOT) <sysinclude>$(BOOST_ROOT)
      # common-variant-tag ensures that the library will
      # be named according to the rules used by the install
      # and auto-link features:
      common-variant-tag 
     : debug release  # build variants
     ;

dll boost_system
     : ../src/$(SOURCES).cpp
     : # build requirements
       <define>BOOST_SYSTEM_DYN_LINK=1  # tell source we're building dll's
       <runtime-link>dynamic  # build only for dynamic runtimes
       <sysinclude>$(BOOST_AUX_ROOT) <sysinclude>$(BOOST_ROOT)
      # common-variant-tag ensures that the library will
      # be named according to the rules used by the install
      # and auto-link features:
      common-variant-tag 
     : debug release  # build variants
     ;

install system lib
     : <lib>boost_system <dll>boost_system
     ;

stage stage/lib : <lib>boost_system <dll>boost_system
    :
        # copy to a path rooted at BOOST_ROOT:
        <locate>$(BOOST_ROOT)
        # make sure the names of the libraries are correctly named:
        common-variant-tag
        # add this target to the "stage" and "all" psuedo-targets:
        <target>stage
        <target>all
    :
        debug release
    ;

# end
EOF

sub create_boost_asio_content
{
  # Create directory structure.
  system("rm -rf $boost_asio_name");
  mkdir("$boost_asio_name");
  mkdir("$boost_asio_name/doc");
  mkdir("$boost_asio_name/doc/html");
  mkdir("$boost_asio_name/boost");
  mkdir("$boost_asio_name/boost/config");
  mkdir("$boost_asio_name/libs");

  # Copy files.
  system("cp -vR ../boost/doc/html/boost_asio.html $boost_asio_name/doc/html");
  system("cp -vR ../boost/doc/html/boost_asio $boost_asio_name/doc/html");
  system("cp -vR ../boost/boost/asio.hpp $boost_asio_name/boost");
  system("cp -vR ../boost/boost/asio $boost_asio_name/boost");
  system("cp -vR ../boost/boost/cerrno.hpp $boost_asio_name/boost");
  system("cp -vR ../boost/boost/config/warning_disable.hpp $boost_asio_name/boost/config");
  system("cp -vR ../boost/boost/system $boost_asio_name/boost");
  system("cp -vR ../boost/libs/asio $boost_asio_name/libs");
  system("cp -vR ../boost/libs/system $boost_asio_name/libs");

  # Add dummy definitions of BOOST_SYMBOL* to boost/system/config.hpp.
  my $from = "$boost_asio_name/boost/system/config.hpp";
  my $to = "$boost_asio_name/boost/system/config.hpp.new";
  open(my $input, "<$from") or die("Can't open $from for reading");
  open(my $output, ">$to") or die("Can't open $to for writing");
  while (my $line = <$input>)
  {
    print($output $line);
    if ($line =~ /<boost\/config\.hpp>/)
    {
      print($output "\n// These #defines added by the separate Boost.Asio package.\n");
      print($output "#if !defined(BOOST_SYMBOL_IMPORT)\n");
      print($output "# if defined(BOOST_HAS_DECLSPEC)\n");
      print($output "#  define BOOST_SYMBOL_IMPORT __declspec(dllimport)\n");
      print($output "# else // defined(BOOST_HAS_DECLSPEC)\n");
      print($output "#  define BOOST_SYMBOL_IMPORT\n");
      print($output "# endif // defined(BOOST_HAS_DECLSPEC)\n");
      print($output "#endif // !defined(BOOST_SYMBOL_IMPORT)\n");
      print($output "#if !defined(BOOST_SYMBOL_EXPORT)\n");
      print($output "# if defined(BOOST_HAS_DECLSPEC)\n");
      print($output "#  define BOOST_SYMBOL_EXPORT __declspec(dllexport)\n");
      print($output "# else // defined(BOOST_HAS_DECLSPEC)\n");
      print($output "#  define BOOST_SYMBOL_EXPORT\n");
      print($output "# endif // defined(BOOST_HAS_DECLSPEC)\n");
      print($output "#endif // !defined(BOOST_SYMBOL_EXPORT)\n");
      print($output "#if !defined(BOOST_SYMBOL_VISIBLE)\n");
      print($output "# define BOOST_SYMBOL_VISIBLE\n");
      print($output "#endif // !defined(BOOST_SYMBOL_VISIBLE)\n\n");
    }
  }
  close($input);
  close($output);
  system("mv $to $from");

  # Create readme.
  $to = "$boost_asio_name/README.txt";
  open($output, ">$to") or die("Can't open $to for writing");
  print($output $boost_asio_readme);
  close($output);

  # Create Boost.System Jamfile.
  $to = "$boost_asio_name/libs/system/build/Jamfile";
  open($output, ">$to") or die("Can't open $to for writing");
  print($output $boost_system_jamfile);
  close($output);

  # Remove SVN files.
  system("find $boost_asio_name -name .svn -exec rm -rf {} \\;");
}

sub make_boost_asio_packages
{
  system("tar --format=ustar -chf - $boost_asio_name | gzip -c >$boost_asio_name.tar.gz");
  system("tar --format=ustar -chf - $boost_asio_name | bzip2 -9 -c >$boost_asio_name.tar.bz2");
  system("rm -f $boost_asio_name.zip");
  system("zip -rq $boost_asio_name.zip $boost_asio_name");
  system("rm -rf $boost_asio_name");
}

sub create_manifests
{
  system("tar tfz $asio_name.tar.gz | sed -e 's/^[^\\/]*//' | sort -df > asio.manifest");
  system("tar tfz $boost_asio_name.tar.gz | sed -e 's/^[^\\/]*//' | sort -df > boost_asio.manifest");
}

(scalar(@ARGV) == 1) or print_usage_and_exit();
determine_version($ARGV[0]);
update_configure_ac();
update_readme();
update_asio_version_hpp();
update_boost_asio_version_hpp();
build_asio_doc();
make_asio_packages();
build_boost_asio_doc();
create_boost_asio_content();
make_boost_asio_packages();
create_manifests();
