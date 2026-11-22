#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "blockdynamicnetwork/graph.hpp"
#include "blockdynamicnetwork/partition.hpp"
#include "blockdynamicnetwork/sbm.hpp"
#include "blockdynamicnetwork/partition_prior.hpp"
#include "blockdynamicnetwork/posterior.hpp"
#include "blockdynamicnetwork/gibbs_sampler.hpp"
#include "blockdynamicnetwork/split_merge_sampler.hpp"
#include "blockdynamicnetwork/split_merge_results.hpp"
#include "blockdynamicnetwork/hybrid_sampler.hpp"
#include "blockdynamicnetwork/hybrid_sampler_results.hpp"
#include "blockdynamicnetwork/mcmc_diagnostics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(blockdynamicnetworks, m) {

    m.doc() = "Python bindings for blockdynamicnetworks";

    // ==================================================
    // Graph
    // ==================================================

    py::class_<bdn::Graph>(m, "Graph")
        .def(py::init<std::size_t>(), py::arg("n"))
        .def("size", &bdn::Graph::size)
        .def("add_edge", &bdn::Graph::add_edge,
             py::arg("u"), py::arg("v"))
        .def("has_edge", &bdn::Graph::has_edge,
             py::arg("u"), py::arg("v"))
        .def("neighbors",
             [](const bdn::Graph& graph, std::size_t u) {
                 return graph.neighbors(u);
             },
             py::arg("u"))
        .def("number_of_edges", &bdn::Graph::number_of_edges);


    // ==================================================
    // Partition
    // ==================================================

    py::class_<bdn::Partition>(m, "Partition")
        .def(py::init<std::vector<int>>(),
             py::arg("labels"))
        .def("size", &bdn::Partition::size)
        .def("label", &bdn::Partition::label,
             py::arg("node"))
        .def("labels",
             [](const bdn::Partition& partition) {
                 return partition.labels();
             })
        .def("number_of_communities",
             &bdn::Partition::number_of_communities)
        .def("community_sizes",
             &bdn::Partition::community_sizes)
        .def("canonicalize",
             &bdn::Partition::canonicalize);


    // ==================================================
    // PriorType
    // ==================================================

    py::enum_<bdn::PriorType>(m, "PriorType")
        .value("DP", bdn::PriorType::DP)
        .value("PY", bdn::PriorType::PY)
        .value("DM", bdn::PriorType::DM)
        .value("GN", bdn::PriorType::GN)
        .value("MFM", bdn::PriorType::MFM)
        .value("PCPP", bdn::PriorType::PCPP);


    // ==================================================
    // SBMModel
    // ==================================================

    py::class_<bdn::SBMModel>(m, "SBMModel")
        .def(py::init<const bdn::Graph&, const bdn::Partition&>(),
             py::arg("graph"),
             py::arg("partition"),
             py::keep_alive<1, 2>())
        .def("log_likelihood",
             &bdn::SBMModel::log_likelihood,
             py::arg("partition"),
             py::arg("theta"))
        .def("log_collapsed_likelihood",
             &bdn::SBMModel::log_collapsed_likelihood,
             py::arg("partition"),
             py::arg("a") = 1.0,
             py::arg("b") = 1.0);


    // ==================================================
    // PartitionPrior
    // ==================================================

    py::class_<bdn::PartitionPrior>(m, "PartitionPrior")

        .def(py::init<double>(),
             py::arg("alpha") = 1.0)

        .def(py::init<bdn::PriorType, double, double>(),
             py::arg("type"),
             py::arg("alpha"),
             py::arg("sigma"))

        .def(py::init<bdn::PriorType, double, std::size_t>(),
             py::arg("type"),
             py::arg("beta"),
             py::arg("H_total"))

        .def(py::init<bdn::PriorType, double>(),
             py::arg("type"),
             py::arg("gamma"))

        .def(py::init<bdn::PriorType, double, const std::vector<double>&>(),
             py::arg("type"),
             py::arg("rho"),
             py::arg("p_k"))

        .def(py::init<bdn::PriorType, double, double, double>(),
             py::arg("type"),
             py::arg("theta"),
             py::arg("omega"),
             py::arg("pcpp_tol"))

        .def("log_probability",
             &bdn::PartitionPrior::log_probability)

        .def("alpha", &bdn::PartitionPrior::alpha)
        .def("sigma", &bdn::PartitionPrior::sigma)
        .def("beta", &bdn::PartitionPrior::beta)
        .def("gamma", &bdn::PartitionPrior::gamma)
        .def("rho", &bdn::PartitionPrior::rho)
        .def("theta", &bdn::PartitionPrior::theta)
        .def("omega", &bdn::PartitionPrior::omega)
        .def("pcpp_tol", &bdn::PartitionPrior::pcpp_tol)
        .def("H_total", &bdn::PartitionPrior::H_total)
        .def("type", &bdn::PartitionPrior::type);


    // ==================================================
    // PosteriorModel
    // ==================================================

    py::class_<bdn::PosteriorModel>(m, "PosteriorModel")

        .def(py::init<
                 const bdn::Graph&,
                 const bdn::PartitionPrior&,
                 double,
                 double>(),
             py::arg("graph"),
             py::arg("prior"),
             py::arg("a") = 1.0,
             py::arg("b") = 1.0,
             py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>())

        .def("log_likelihood",
             &bdn::PosteriorModel::log_likelihood)

        .def("log_prior",
             &bdn::PosteriorModel::log_prior)

        .def("log_probability",
             &bdn::PosteriorModel::log_probability)

        .def("a", &bdn::PosteriorModel::a)
        .def("b", &bdn::PosteriorModel::b);


    // ==================================================
    // GibbsSampler
    // ==================================================

    py::class_<bdn::GibbsSampler>(m, "GibbsSampler")

        .def(py::init<
                 const bdn::PosteriorModel&,
                 std::size_t,
                 std::size_t,
                 std::size_t,
                 std::uint64_t>(),
             py::arg("posterior"),
             py::arg("iterations"),
             py::arg("burn_in") = 0,
             py::arg("thinning") = 1,
             py::arg("seed") = 123,
             py::keep_alive<1, 2>())

        .def("run",
             &bdn::GibbsSampler::run,
             py::arg("initial_partition"))

        .def("samples",
             &bdn::GibbsSampler::samples)

        .def("number_of_communities",
             &bdn::GibbsSampler::number_of_communities)

        .def("final_partition",
             &bdn::GibbsSampler::final_partition,
             py::return_value_policy::reference_internal)

        .def("iterations",
             &bdn::GibbsSampler::iterations);


    // ==================================================
    // SplitMergeResults
    // ==================================================

    py::class_<bdn::SplitMergeResults>(m, "SplitMergeResults")

        .def_readonly("samples",
                     &bdn::SplitMergeResults::samples)

        .def_readonly("k_trace",
                     &bdn::SplitMergeResults::k_trace)

        .def_readonly("log_posterior_trace",
                     &bdn::SplitMergeResults::log_posterior_trace)

        .def_readonly("split_attempt_trace",
                     &bdn::SplitMergeResults::split_attempt_trace)

        .def_readonly("merge_attempt_trace",
                     &bdn::SplitMergeResults::merge_attempt_trace)

        .def_readonly("split_acceptance_trace",
                     &bdn::SplitMergeResults::split_acceptance_trace)

        .def_readonly("merge_acceptance_trace",
                     &bdn::SplitMergeResults::merge_acceptance_trace)

        .def_readonly("split_acceptance_rate",
                     &bdn::SplitMergeResults::split_acceptance_rate)

        .def_readonly("merge_acceptance_rate",
                     &bdn::SplitMergeResults::merge_acceptance_rate)

        .def_readonly("ess_k",
                     &bdn::SplitMergeResults::ess_k)

        .def_readonly("ess_log_posterior",
                     &bdn::SplitMergeResults::ess_log_posterior)

        .def_readonly("iterations",
                     &bdn::SplitMergeResults::iterations)

        .def_readonly("burn_in",
                     &bdn::SplitMergeResults::burn_in)

        .def_readonly("thinning",
                     &bdn::SplitMergeResults::thinning)

        .def_readonly("saved_samples",
                     &bdn::SplitMergeResults::saved_samples)

        .def_readonly("final_partition",
                     &bdn::SplitMergeResults::final_partition);


    // ==================================================
    // SplitMergeSampler
    // ==================================================

    py::class_<bdn::SplitMergeSampler>(m, "SplitMergeSampler")

        .def(py::init<
          const bdn::PosteriorModel&,
          std::size_t,
          std::size_t,
          std::size_t,
          std::uint64_t,
          bool>(),
          py::arg("posterior"),
          py::arg("iterations"),
          py::arg("burn_in") = 0,
          py::arg("thinning") = 1,
          py::arg("seed") = 123,
          py::arg("verbose") = true,
          py::keep_alive<1, 2>())

        .def("run",
             &bdn::SplitMergeSampler::run,
             py::arg("initial_partition"))

        .def("samples",
             &bdn::SplitMergeSampler::samples)

        .def("number_of_communities",
             &bdn::SplitMergeSampler::number_of_communities)

        .def("final_partition",
             &bdn::SplitMergeSampler::final_partition,
             py::return_value_policy::reference_internal)

        .def("iterations",
             &bdn::SplitMergeSampler::iterations)

        .def("accepted_splits",
             &bdn::SplitMergeSampler::accepted_splits)

        .def("accepted_merges",
             &bdn::SplitMergeSampler::accepted_merges)

        .def("results",
             &bdn::SplitMergeSampler::results,
             py::return_value_policy::reference_internal);
     
     // ==================================================
     // HybridSamplerResults
     // ==================================================

     py::class_<bdn::HybridSamplerResults>(m, "HybridSamplerResults")

     .def_readonly("samples",
                    &bdn::HybridSamplerResults::samples)

     .def_readonly("k_trace",
                    &bdn::HybridSamplerResults::k_trace)

     .def_readonly("log_posterior_trace",
                    &bdn::HybridSamplerResults::log_posterior_trace)

     .def_readonly("sampler_selection_trace",
                    &bdn::HybridSamplerResults::sampler_selection_trace)

     .def_readonly("gibbs_iterations",
                    &bdn::HybridSamplerResults::gibbs_iterations)

     .def_readonly("split_merge_iterations",
                    &bdn::HybridSamplerResults::split_merge_iterations)

     .def_readonly("split_attempt_trace",
                    &bdn::HybridSamplerResults::split_attempt_trace)

     .def_readonly("merge_attempt_trace",
                    &bdn::HybridSamplerResults::merge_attempt_trace)

     .def_readonly("split_acceptance_trace",
                    &bdn::HybridSamplerResults::split_acceptance_trace)

     .def_readonly("merge_acceptance_trace",
                    &bdn::HybridSamplerResults::merge_acceptance_trace)

     .def_readonly("split_acceptance_rate",
                    &bdn::HybridSamplerResults::split_acceptance_rate)

     .def_readonly("merge_acceptance_rate",
                    &bdn::HybridSamplerResults::merge_acceptance_rate)

     .def_readonly("ess_k",
                    &bdn::HybridSamplerResults::ess_k)

     .def_readonly("ess_log_posterior",
                    &bdn::HybridSamplerResults::ess_log_posterior)

     .def_readonly("iterations",
                    &bdn::HybridSamplerResults::iterations)

     .def_readonly("burn_in",
                    &bdn::HybridSamplerResults::burn_in)

     .def_readonly("thinning",
                    &bdn::HybridSamplerResults::thinning)

     .def_readonly("saved_samples",
                    &bdn::HybridSamplerResults::saved_samples)

     .def_readonly("gibbs_probability",
                    &bdn::HybridSamplerResults::gibbs_probability)

     .def_readonly("final_partition",
                    &bdn::HybridSamplerResults::final_partition);

     // ==================================================
     // HybridSampler
     // ==================================================

     py::class_<bdn::HybridSampler>(m, "HybridSampler")

     .def(py::init<
          const bdn::PosteriorModel&,
          std::size_t,
          std::size_t,
          std::size_t,
          std::uint64_t,
          double>(),
          py::arg("posterior"),
          py::arg("iterations"),
          py::arg("burn_in") = 0,
          py::arg("thinning") = 1,
          py::arg("seed") = 123,
          py::arg("gibbs_probability") = 0.75,
          py::keep_alive<1, 2>())

     .def("run",
          &bdn::HybridSampler::run,
          py::arg("initial_partition"))

     .def("samples",
          &bdn::HybridSampler::samples)

     .def("number_of_communities",
          &bdn::HybridSampler::number_of_communities)

     .def("final_partition",
          &bdn::HybridSampler::final_partition,
          py::return_value_policy::reference_internal)

     .def("iterations",
          &bdn::HybridSampler::iterations)

     .def("gibbs_iterations",
          &bdn::HybridSampler::gibbs_iterations)

     .def("split_merge_iterations",
          &bdn::HybridSampler::split_merge_iterations)

     .def("gibbs_probability",
          &bdn::HybridSampler::gibbs_probability)

     .def("results",
          &bdn::HybridSampler::results,
          py::return_value_policy::reference_internal);

    // ==================================================
    // MCMC diagnostics
    // ==================================================

    m.def("effective_sample_size",
          &bdn::effective_sample_size,
          py::arg("trace"),
          py::arg("max_lag") = 1000);
}