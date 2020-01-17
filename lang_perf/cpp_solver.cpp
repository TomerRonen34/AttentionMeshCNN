#include "cpp_solver.h"

const int NOT_PROCESSED = std::numeric_limits<int>::max();


CppSolver::CppSolver(vvi* _graph)
  : graph(_graph)
{

}

void CppSolver::run_apsp(char* scores, int cutoff) {
    auto n = graph->size();
    std::memset(scores, 255, n * n);

    unsigned long long offset = 0;
    for (int start = 0; start < n; start++) {
        scores[start + offset] = 1;
        deque<int> queue;
        queue.push_back(start);

        while (!queue.empty()) {
            int now = queue.front();
            queue.pop_front();
            char value = scores[now + offset];

            for (const auto& next : (*graph)[now]) {
                if (scores[next + offset] == (char)255) {
                    scores[next + offset] = value + 1;
                    if (cutoff == -1 || value < cutoff) {
                        queue.push_back(next);
                    }
                }
            }
        }
        offset += n;
    }
}

void CppSolver::run_bfs(int start, vi* _scores, int cutoff) {
  vi& scores = *_scores;
  auto n = graph->size();
  scores.assign(n, NOT_PROCESSED);
  scores[start] = 0;

  deque<int> queue;
  queue.push_back(start);
  while (!queue.empty()) {

    int now = queue.front();
    queue.pop_front();
    int value = scores[now];

    for (const auto& next : (*graph)[now]) {
      if (scores[next] == NOT_PROCESSED) {
        scores[next] = value + 1;
        if (cutoff == -1 || scores[next] < cutoff)
            queue.push_back(next);
      }
    }
  }
}

//void CppSolver::compute_hash() {
//  clock_t tbegin = clock();
//
//  auto num_cities = graph->size();
//  vvi distances(num_cities);
//  for (int i = 0; i < num_cities; ++i) {
//    vi scores;
//    run_bfs(i, &scores);
//    distances[i] = std::move(scores);
//  }
//
//  int total_score = 0;
//  for (int i = 0; i < num_cities; ++i) {
//    int current_score = 0;
//    for (const auto& d : distances[i]) {
//      if (d == NOT_PROCESSED) {
//        continue;
//      }
//      int d2 = d * d;
//      current_score ^= d2;
//    }
//    total_score += current_score;
//  }
//  cout << "graph_hash=" << total_score << endl;
//
//  clock_t tend = clock();
//  double elapsed_msecs = double(tend - tbegin) / CLOCKS_PER_SEC * 1000;
//  cout << "time=" << elapsed_msecs << endl;
//}
