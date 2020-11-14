echo "--- clang, execute(x,f), members (status quo) ---"
./clang-compile.sh props_member execute.cpp
echo ""

echo "--- clang, execute(x,f), friends ---"
./clang-compile.sh props_friend execute.cpp
echo ""

echo "--- clang, execute(x,f), single member ---"
./clang-compile.sh props_single_member execute.cpp
echo ""

echo "--- clang, execute(x,f), single friend ---"
./clang-compile.sh props_single_friend execute.cpp
echo ""

echo "--- clang, custom execute(x,f), members (status quo) ---"
./clang-compile.sh props_member execute-via-custom-cpo.cpp
echo ""

echo "--- clang, custom execute(x,f), friends ---"
./clang-compile.sh props_friend execute-via-custom-cpo.cpp
echo ""

echo "--- clang, custom execute(x,f), single member ---"
./clang-compile.sh props_single_member execute-via-custom-cpo.cpp
echo ""

echo "--- clang, custom execute(x,f), single friend ---"
./clang-compile.sh props_single_friend execute-via-custom-cpo.cpp
echo ""

echo "--- clang, require(x,blocking.never), members (status quo) ---"
./clang-compile.sh props_member require_blocking_never.cpp
echo ""

echo "--- clang, require(x,blocking.never), friends ---"
./clang-compile.sh props_friend require_blocking_never.cpp
echo ""

echo "--- clang, require(x,blocking.never), single member ---"
./clang-compile.sh props_single_member require_blocking_never.cpp
echo ""

echo "--- clang, require(x,blocking.never), single friend ---"
./clang-compile.sh props_single_friend require_blocking_never.cpp
echo ""

echo "--- clang, require(x,relationship.fork), members (status quo) ---"
./clang-compile.sh props_member require_relationship_fork.cpp
echo ""

echo "--- clang, require(x,relationship.fork), friends ---"
./clang-compile.sh props_friend require_relationship_fork.cpp
echo ""

echo "--- clang, require(x,relationship.fork), single member ---"
./clang-compile.sh props_single_member require_relationship_fork.cpp
echo ""

echo "--- clang, require(x,relationship.fork), single friend ---"
./clang-compile.sh props_single_friend require_relationship_fork.cpp
echo ""

echo "--- clang, custom require(x,relationship.fork), members (status quo) ---"
./clang-compile.sh props_member require_relationship_fork-via-custom-cpo.cpp
echo ""

echo "--- clang, custom require(x,relationship.fork), friends ---"
./clang-compile.sh props_friend require_relationship_fork-via-custom-cpo.cpp
echo ""

echo "--- clang, custom require(x,relationship.fork), single member ---"
./clang-compile.sh props_single_member require_relationship_fork-via-custom-cpo.cpp
echo ""

echo "--- clang, custom require(x,relationship.fork), single friend ---"
./clang-compile.sh props_single_friend require_relationship_fork-via-custom-cpo.cpp
echo ""

echo "--- clang, prefer(x,blocking.possibly), members (status quo) ---"
./clang-compile.sh props_member prefer_blocking_possibly.cpp
echo ""

echo "--- clang, prefer(x,blocking.possibly), friends ---"
./clang-compile.sh props_friend prefer_blocking_possibly.cpp
echo ""

echo "--- clang, prefer(x,blocking.possibly), single member ---"
./clang-compile.sh props_single_member prefer_blocking_possibly.cpp
echo ""

echo "--- clang, prefer(x,blocking.possibly), single friend ---"
./clang-compile.sh props_single_friend prefer_blocking_possibly.cpp
echo ""

echo "--- clang, prefer(x,outstanding_work.tracked), members (status quo) ---"
./clang-compile.sh props_member prefer_outstanding_work_tracked.cpp
echo ""

echo "--- clang, prefer(x,outstanding_work.tracked), friends ---"
./clang-compile.sh props_friend prefer_outstanding_work_tracked.cpp
echo ""

echo "--- clang, prefer(x,outstanding_work.tracked), single member ---"
./clang-compile.sh props_single_member prefer_outstanding_work_tracked.cpp
echo ""

echo "--- clang, prefer(x,outstanding_work.tracked), single friend ---"
./clang-compile.sh props_single_friend prefer_outstanding_work_tracked.cpp
echo ""
