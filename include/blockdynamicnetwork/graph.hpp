
#pragma once
#include <vector>
#include <cstddef>

namespace bdn {

class Graph {
public:
    explicit Graph(std::size_t n);

    std::size_t size() const noexcept;
    void add_edge(std::size_t u, std::size_t v);
    bool has_edge(std::size_t u, std::size_t v) const;
    const std::vector<std::size_t>& neighbors(std::size_t u) const;
    std::size_t number_of_edges() const noexcept;

private:
    std::vector<std::vector<std::size_t>> adjacency_;
    std::size_t m_ = 0;

    void check_node(std::size_t u) const;
};

}
