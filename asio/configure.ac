AC_INIT(include/asio.hpp)
AM_MAINTAINER_MODE
AM_INIT_AUTOMAKE(asio, [0.1.13])

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
  BOOSTDIR="`pwd`/../boost_1_32_0"
  if test -d "${BOOSTDIR}"; then
    CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
  else
    BOOSTDIR="`pwd`/../boost_1_31_0"
    if test -d "${BOOSTDIR}"; then
      CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
    else
      BOOSTDIR="`pwd`/../boost-1.30.2"
      if test -d "${BOOSTDIR}"; then
        CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
      else
        BOOSTDIR="`pwd`/../boost_1_30_0"
        if test -d "${BOOSTDIR}"; then
          CPPFLAGS="$CPPFLAGS -I${BOOSTDIR}"
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
    LIBS="$LIBS -lws2_32"
    ;;
esac

if test "$GXX" = yes; then
  CXXFLAGS="$CXXFLAGS -ftemplate-depth-32"
fi

AC_CHECK_PROG(DOXYGEN_FOUND,doxygen,yes)
AM_CONDITIONAL(HAVE_DOXYGEN,test x$DOXYGEN_FOUND = xyes)
AC_CHECK_PROG(HAVE_DOT,dot,YES,NO)
AC_SUBST(HAVE_DOT)

AC_OUTPUT([
  Makefile
  include/Makefile
  src/Makefile
  src/doc/Makefile
  src/doc/asio.dox])
