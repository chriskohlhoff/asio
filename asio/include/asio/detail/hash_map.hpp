//
// hash_map.hpp
// ~~~~~~~~~~~~
//
// Copyright (c) 2003, 2004 Christopher M. Kohlhoff (chris@kohlhoff.com)
//
// Permission to use, copy, modify, distribute and sell this software and its
// documentation for any purpose is hereby granted without fee, provided that
// the above copyright notice appears in all copies and that both the copyright
// notice and this permission notice appear in supporting documentation. This
// software is provided "as is" without express or implied warranty, and with
// no claim as to its suitability for any purpose.
//

#ifndef ASIO_DETAIL_HASH_MAP_HPP
#define ASIO_DETAIL_HASH_MAP_HPP

#include "asio/detail/push_options.hpp"

#include "asio/detail/push_options.hpp"
#include <cassert>
#include <list>
#include <utility>
#include <boost/noncopyable.hpp>
#include "asio/detail/pop_options.hpp"

namespace asio {
namespace detail {

template <typename K>
unsigned int hash(const K& k)
{
  return (unsigned int)k;
}

template <typename K, typename V>
class hash_map
  : private boost::noncopyable
{
public:
  // The type of a value in the map.
  typedef std::pair<const K, V> value_type;

  // The type of a non-const iterator over the hash map.
  typedef typename std::list<value_type>::iterator iterator;

  // The type of a const iterator over the hash map.
  typedef typename std::list<value_type>::const_iterator const_iterator;

  // Constructor.
  hash_map()
  {
    // Initialise all buckets to empty.
    for (unsigned int i = 0; i < num_buckets; ++i)
      buckets_[i].first = buckets_[i].last = values_.end();
  }

  // Get an iterator for the beginning of the map.
  iterator begin()
  {
    return values_.begin();
  }

  // Get an iterator for the beginning of the map.
  const_iterator begin() const
  {
    return values_.begin();
  }

  // Get an iterator for the end of the map.
  iterator end()
  {
    return values_.end();
  }

  // Get an iterator for the end of the map.
  const_iterator end() const
  {
    return values_.end();
  }

  // Find an entry in the map.
  iterator find(const K& k)
  {
    unsigned int bucket = hash(k) % num_buckets;
    iterator it = buckets_[bucket].first;
    if (it == values_.end())
      return values_.end();
    iterator end = buckets_[bucket].last;
    ++end;
    while (it != end)
    {
      if (it->first == k)
        return it;
      ++it;
    }
    return values_.end();
  }

  // Find an entry in the map.
  const_iterator find(const K& k) const
  {
    unsigned int bucket = hash(k) % num_buckets;
    const_iterator it = buckets_[bucket].first;
    if (it == values_.end())
      return it;
    const_iterator end = buckets_[bucket].last;
    ++end;
    while (it != end)
    {
      if (it->first == k)
        return it;
      ++it;
    }
    return values_.end();
  }

  // Insert a new entry into the map.
  std::pair<iterator, bool> insert(const value_type& v)
  {
    unsigned int bucket = hash(v.first) % num_buckets;
    iterator it = buckets_[bucket].first;
    if (it == values_.end())
    {
      buckets_[bucket].first = buckets_[bucket].last =
        values_.insert(values_.end(), v);
      return std::pair<iterator, bool>(buckets_[bucket].last, true);
    }
    iterator end = buckets_[bucket].last;
    ++end;
    while (it != end)
    {
      if (it->first == v.first)
        return std::pair<iterator, bool>(it, false);
      ++it;
    }
    buckets_[bucket].last = values_.insert(end, v);
    return std::pair<iterator, bool>(buckets_[bucket].last, true);
  }

  // Erase an entry from the map.
  void erase(iterator it)
  {
    assert(it != values_.end());

    unsigned int bucket = hash(it->first) % num_buckets;
    bool is_first = (it == buckets_[bucket].first);
    bool is_last = (it == buckets_[bucket].last);
    if (is_first && is_last)
      buckets_[bucket].first = buckets_[bucket].last = values_.end();
    else if (is_first)
      ++buckets_[bucket].first;
    else if (is_last)
      --buckets_[bucket].last;

    values_.erase(it);
  }

private:
  // The list of all values in the hash map.
  std::list<value_type> values_;

  // The type for a bucket in the hash table.
  struct bucket_type
  {
    iterator first;
    iterator last;
  };

  // The number of buckets in the hash.
  enum { num_buckets = 1021 };

  // The buckets in the hash.
  bucket_type buckets_[num_buckets];
};

} // namespace detail
} // namespace asio

#include "asio/detail/pop_options.hpp"

#endif // ASIO_DETAIL_HASH_MAP_HPP
