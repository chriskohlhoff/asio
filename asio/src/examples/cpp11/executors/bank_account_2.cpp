#include <asio/package.hpp>
#include <asio/post.hpp>
#include <asio/thread_pool.hpp>
#include <iostream>

using asio::package;
using asio::post;
using asio::thread_pool;

// Traditional active object pattern.
// Member functions block until operation is finished.

class bank_account
{
  int balance_ = 0;
  mutable thread_pool pool_{1};

public:
  void deposit(int amount)
  {
    post(pool_,
      package([=]
        {
          balance_ += amount;
        })).get();
  }

  void withdraw(int amount)
  {
    post(pool_,
      package([=]
        {
          if (balance_ >= amount)
            balance_ -= amount;
        })).get();
  }

  int balance() const
  {
    return post(pool_,
      package([=]
        {
          return balance_;
        })).get();
  }
};

int main()
{
  bank_account acct;
  acct.deposit(20);
  acct.withdraw(10);
  std::cout << "balance = " << acct.balance() << "\n";
}
