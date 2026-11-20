#pragma once

#include <vector>
#include <cstddef>

namespace bdn {

class Partition {
public:
    explicit Partition(std::vector<int> labels);

    std::size_t size() const noexcept;
    int label(std::size_t node) const;

    const std::vector<int>& labels() const noexcept;

    int number_of_communities() const noexcept;
    std::vector<std::size_t> community_sizes() const;

    void canonicalize();

private:
    std::vector<int> z_;
};

}