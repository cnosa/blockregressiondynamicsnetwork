#include "blockdynamicnetwork/graph.hpp"
#include <stdexcept>
#include <algorithm>

namespace bdn {

Graph::Graph(std::size_t n) : adjacency_(n) {}

std::size_t Graph::size() const noexcept {
    return adjacency_.size();
}

void Graph::check_node(std::size_t u) const {
    if (u >= size())
        throw std::out_of_range("Node index outside graph bounds.");
}

void Graph::add_edge(std::size_t u, std::size_t v) {
    check_node(u);
    check_node(v);

    if (u == v)
        throw std::invalid_argument("Self-loops are not supported.");

    if (has_edge(u, v))
        return;

    adjacency_[u].push_back(v);
    adjacency_[v].push_back(u);
    ++m_;
}

bool Graph::has_edge(std::size_t u, std::size_t v) const {
    check_node(u);
    check_node(v);

    const auto& row = adjacency_[u];

    return std::find(row.begin(), row.end(), v) != row.end();
}

const std::vector<std::size_t>& Graph::neighbors(std::size_t u) const {
    check_node(u);
    return adjacency_[u];
}

std::size_t Graph::number_of_edges() const noexcept {
    return m_;
}

}