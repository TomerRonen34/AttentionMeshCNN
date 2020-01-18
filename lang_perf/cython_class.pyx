from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
import numpy as np

NOT_PROCESSED = -1

cdef extern from "cpp_solver.h":
    cdef cppclass CppSolver:
        #Methods
        CppSolver(vector[vector[int]]*) except +
        void run_bfs(int, vector[int]*, int)
        void run_apsp(char*, int)
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

        n = self.num_edges
        cdef char* bla = <char *> malloc(n ** 3)

        self.cpp_solver.run_apsp(bla, cutoff)

        dists_flat = np.frombuffer(bla, dtype="uint8", count=n**2) - 1
        free(bla)
        return dists_flat
