from libcpp.vector cimport vector
import numpy as np

NOT_PROCESSED = -1

cdef extern from "cpp_solver.h":
    cdef cppclass CppSolver:
        #Methods
        CppSolver(vector[vector[int]]*) except +
        void run_bfs(int, vector[int]*, int)
        void compute_hash()


cdef class Solver:
    cdef vector[vector[int]] graph
    cdef int num_edges
    cdef CppSolver* cpp_solver

    def init(self, adj_lists):
        self.num_edges = len(adj_lists)

        cdef vector[vector[int]] graph
        graph.resize(self.num_edges)
        for s, neighs in enumerate(adj_lists):
            for t in neighs:
                s = int(s)
                t = int(t)

                graph[s].push_back(t)
                graph[t].push_back(s)

        self.graph = graph
        self.cpp_solver = new CppSolver(&self.graph)

    def all_pairs_shortest_path(self, cutoff=None):
        if cutoff is None:
            cutoff = -1
        cdef vector[vector[int]] distances
        distances.resize(self.num_edges)
        for v in range(self.num_edges):
            self.cpp_solver.run_bfs(v, &distances[v], cutoff)

        return distances
