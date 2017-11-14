#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <functional>

void Shard(int max_parallelism, int num_threads, int total,
           int cost_per_unit, std::function<void(int, int)> work);

#endif // #ifndef THREAD_POOL_H
