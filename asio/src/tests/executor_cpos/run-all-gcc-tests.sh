echo "--- gcc, execute(x,f), members (status quo) ---"
./gcc-compile.sh props_member execute.cpp
echo ""

echo "--- gcc, execute(x,f), friends ---"
./gcc-compile.sh props_friend execute.cpp
echo ""

echo "--- gcc, execute(x,f), single member ---"
./gcc-compile.sh props_single_member execute.cpp
echo ""

echo "--- gcc, execute(x,f), single friend ---"
./gcc-compile.sh props_single_friend execute.cpp
echo ""

echo "--- gcc, custom execute(x,f), members (status quo) ---"
./gcc-compile.sh props_member execute-via-custom-cpo.cpp
echo ""

echo "--- gcc, custom execute(x,f), friends ---"
./gcc-compile.sh props_friend execute-via-custom-cpo.cpp
echo ""

echo "--- gcc, custom execute(x,f), single member ---"
./gcc-compile.sh props_single_member execute-via-custom-cpo.cpp
echo ""

echo "--- gcc, custom execute(x,f), single friend ---"
./gcc-compile.sh props_single_friend execute-via-custom-cpo.cpp
echo ""

echo "--- gcc, require(x,blocking.never), members (status quo) ---"
./gcc-compile.sh props_member require_blocking_never.cpp
echo ""

echo "--- gcc, require(x,blocking.never), friends ---"
./gcc-compile.sh props_friend require_blocking_never.cpp
echo ""

echo "--- gcc, require(x,blocking.never), single member ---"
./gcc-compile.sh props_single_member require_blocking_never.cpp
echo ""

echo "--- gcc, require(x,blocking.never), single friend ---"
./gcc-compile.sh props_single_friend require_blocking_never.cpp
echo ""

echo "--- gcc, require(x,relationship.fork), members (status quo) ---"
./gcc-compile.sh props_member require_relationship_fork.cpp
echo ""

echo "--- gcc, require(x,relationship.fork), friends ---"
./gcc-compile.sh props_friend require_relationship_fork.cpp
echo ""

echo "--- gcc, require(x,relationship.fork), single member ---"
./gcc-compile.sh props_single_member require_relationship_fork.cpp
echo ""

echo "--- gcc, require(x,relationship.fork), single friend ---"
./gcc-compile.sh props_single_friend require_relationship_fork.cpp
echo ""

echo "--- gcc, custom require(x,relationship.fork), members (status quo) ---"
./gcc-compile.sh props_member require_relationship_fork-via-custom-cpo.cpp
echo ""

echo "--- gcc, custom require(x,relationship.fork), friends ---"
./gcc-compile.sh props_friend require_relationship_fork-via-custom-cpo.cpp
echo ""

echo "--- gcc, custom require(x,relationship.fork), single member ---"
./gcc-compile.sh props_single_member require_relationship_fork-via-custom-cpo.cpp
echo ""

echo "--- gcc, custom require(x,relationship.fork), single friend ---"
./gcc-compile.sh props_single_friend require_relationship_fork-via-custom-cpo.cpp
echo ""

echo "--- gcc, prefer(x,blocking.possibly), members (status quo) ---"
./gcc-compile.sh props_member prefer_blocking_possibly.cpp
echo ""

echo "--- gcc, prefer(x,blocking.possibly), friends ---"
./gcc-compile.sh props_friend prefer_blocking_possibly.cpp
echo ""

echo "--- gcc, prefer(x,blocking.possibly), single member ---"
./gcc-compile.sh props_single_member prefer_blocking_possibly.cpp
echo ""

echo "--- gcc, prefer(x,blocking.possibly), single friend ---"
./gcc-compile.sh props_single_friend prefer_blocking_possibly.cpp
echo ""

echo "--- gcc, prefer(x,outstanding_work.tracked), members (status quo) ---"
./gcc-compile.sh props_member prefer_outstanding_work_tracked.cpp
echo ""

echo "--- gcc, prefer(x,outstanding_work.tracked), friends ---"
./gcc-compile.sh props_friend prefer_outstanding_work_tracked.cpp
echo ""

echo "--- gcc, prefer(x,outstanding_work.tracked), single member ---"
./gcc-compile.sh props_single_member prefer_outstanding_work_tracked.cpp
echo ""

echo "--- gcc, prefer(x,outstanding_work.tracked), single friend ---"
./gcc-compile.sh props_single_friend prefer_outstanding_work_tracked.cpp
echo ""
