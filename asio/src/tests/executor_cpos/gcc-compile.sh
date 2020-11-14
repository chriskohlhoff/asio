SUFFIX=$1
SOURCE=$2
ASIO_ROOT="../../../include"
CXXFLAGS="-std=c++2a -I$ASIO_ROOT -O1 -DASIO_NO_DEPRECATED -DEXECUTOR_HEADER=\"test_executor_$SUFFIX.hpp\""
g++-10 $CXXFLAGS -x c++-header -c pch.hpp
g++-10 $CXXFLAGS -include pch.hpp -c $SOURCE
if [ $? != 0 ]; then
  exit 1
fi
for i in {1..25}; do
  (time g++-10 $CXXFLAGS -include pch.hpp -c $SOURCE) 2>&1 | grep user | sed -e 's/^.*0m\(.*\)s$/\1/'
done
