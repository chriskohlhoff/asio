AC_INIT(include/asio.hpp)
AM_MAINTAINER_MODE
AM_INIT_AUTOMAKE(asio, [1.0.0])

AC_CANONICAL_HOST
AM_PROG_CC_C_O
AC_PROG_CXX
AC_LANG(C++)
AC_PROG_RANLIB

AC_DEFINE(_REENTRANT, [1], [Define this])

AC_ARG_WITH(boost,
  AC_HELP_STRING([--with-boost=DIR],[location of boost distribution]),
[
  CPPFLAGS="$CPPFLAGS -I${withval}"
],
[
  BOOSTDIR="`pwd`/../boost_1_35_0"
  if test -d "${BOOSTDIR}"; then
    CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
  else
    BOOSTDIR="`pwd`/../boost_1_34_1"
    if test -d "${BOOSTDIR}"; then
      CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
    else
      BOOSTDIR="`pwd`/../boost_1_34_0"
      if test -d "${BOOSTDIR}"; then
        CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
      else
        BOOSTDIR="`pwd`/../boost_1_33_1"
        if test -d "${BOOSTDIR}"; then
          CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
        else
          BOOSTDIR="`pwd`/../boost_1_33_0"
          if test -d "${BOOSTDIR}"; then
            CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
          fi
        fi
      fi
    fi
  fi
])

AC_CHECK_HEADER([boost/noncopyable.hpp],,
[
  echo "Can't find boost headers. Please check the location of the boost"
  echo "distribution and rerun configure using the --with-boost=DIR option."
  exit 1
],[])

AC_ARG_WITH(openssl,
  AC_HELP_STRING([--with-openssl=DIR],[location of openssl]),
[
  CPPFLAGS="$CPPFLAGS -I${withval}"
],[])

AC_CHECK_HEADER([openssl/ssl.h],,
[
  OPENSSL_FOUND=no
],[])

if test x$OPENSSL_FOUND != xno; then
  LIBS="$LIBS -lssl -lcrypto"
fi

AM_CONDITIONAL(HAVE_OPENSSL,test x$OPENSSL_FOUND != xno)

WINDOWS=no
case $host in
  *-*-linux*)
    CXXFLAGS="$CXXFLAGS -pthread"
    LDFLAGS="$LDFLAGS -pthread"
    ;;
  *-*-solaris*)
    if test "$GXX" = yes; then
      CXXFLAGS="$CXXFLAGS"
    else
      # We'll assume Sun's CC.
      CXXFLAGS="$CXXFLAGS -mt"
    fi
    LIBS="$LIBS -lsocket -lnsl -lpthread"
    ;;
  *-*-mingw32*)
    CXXFLAGS="$CXXFLAGS -mthreads"
    LDFLAGS="$LDFLAGS -mthreads"
    LIBS="$LIBS -lws2_32 -lmswsock"
    WINDOWS=yes
    ;;
  *-pc-cygwin*)
    CXXFLAGS="$CXXFLAGS -D__USE_W32_SOCKETS -D_WIN32_WINNT=0x0501"
    LIBS="$LIBS -lws2_32 -lmswsock"
    WINDOWS=yes
    ;;
  *-apple-darwin*)
    CXXFLAGS="$CXXFLAGS"
    LDFLAGS="$LDFLAGS"
    ;;
  *-*-freebsd*)
    CXXFLAGS="$CXXFLAGS -pthread"
    LDFLAGS="$LDFLAGS -pthread"
    ;;
esac

if test "$GXX" = yes; then
  CXXFLAGS="$CXXFLAGS -ftemplate-depth-256"
fi

AM_CONDITIONAL(WINDOWS_TARGET,test x$WINDOWS != xno)

AC_OUTPUT([
  Makefile
  include/Makefile
  src/Makefile])
