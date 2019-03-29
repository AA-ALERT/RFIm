// Copyright 2019 Netherlands eScience Center and Netherlands Institute for Radio Astronomy (ASTRON)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <OpenCLTypes.hpp>
#include <Kernel.hpp>
#include <InitializeOpenCL.hpp>
#include <Observation.hpp>
#include <utils.hpp>
#include <Statistics.hpp>
#include <Timer.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <map>

#pragma once

namespace RFIm
{

/**
 ** @brief RFI specific kernel configuration.
 */
class RFImConfig : public isa::OpenCL::KernelConf
{
public:
    RFImConfig();
    ~RFImConfig();
    /**
     ** @brief Return true if in subbanding mode, false otherwise.
     */
    bool getSubbandDedispersion() const;
    /**
     ** @brief Return true if using a condition for replacement, false otherwise.
     */
    bool getConditionalReplacement() const;
    /**
     ** @brief Set the subbanding mode.
     */
    void setSubbandDedispersion(const bool subband);
    /**
     ** @brief Set the conditional replacement mode.
     */
    void setConditionalReplacement(const bool replacement);
    /**
     ** @brief Print the configuration.
     */
    std::string print() const;

private:
    bool subbandDedispersion;
    bool conditionalReplacement;
};

/**
 ** @brief The kernel type.
 */
enum RFImKernel
{
    TimeDomainSigmaCut,
    FrequencyDomainSigmaCut
};

/**
 ** @brief Ordering of input/output data.
 */
enum DataOrdering
{
    FrequencyTime,
    TimeFrequency
};

/**
 ** @brief Strategy for flagged data replacement.
 */
enum ReplacementStrategy
{
    ReplaceWithMean,
    ReplaceWithMedian
};

using RFImConfigurations = std::map<std::string, std::map<unsigned int, std::map<float, RFImConfig *> *> *>;

/**
 ** @brief Read one RFImConfig from a configuration file.
 **
 ** @param configurations Where to store all configurations.
 ** @param filename The file to read the configurations from.
 */
void readRFImConfig(RFImConfigurations & configurations, const std::string & filename);

/**
 ** @brief Read the
 */
void readSigmaSteps(const std::string &inputFilename, std::vector<float> &steps);

/**
 ** @brief Compute time domain sigma cut.
 ** Not optimized, just for testing purpose.
 **
 ** @param subbandDedispersion True if using subband dedispersion.
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param observation The observation object.
 ** @param time_series The input data.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 **
 ** @return The number of replaced samples.
 */
template<typename DataType>
std::uint64_t timeDomainSigmaCut(const bool subbandDedispersion, const DataOrdering & ordering, const ReplacementStrategy & replacement, const AstroData::Observation & observation, std::vector<DataType> & time_series, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Generates the OpenCL code for the time domain sigma cut.
 **
 ** @param config The kernel configuration.
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 **
 ** @return String containing the generated code.
 */
template<typename DataType>
std::string * getTimeDomainSigmaCutOpenCL(const RFImConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Generates the OpenCL code for the time domain sigma cut.
 ** This function generates specialized code for the case in which the input is FrequencyTime ordered and flagged samples are replaced with the mean.
 **
 ** @param config The kernel configuration.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 **
 ** @return String containing the generated code.
 */
template<typename DataType>
std::string * getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean(const RFImConfig & config, const std::string & dataTypeName, const AstroData::Observation & observation, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Test the OpenCL kernel by comparing results with C++ implementation.
 **
 ** @param printCode Enable generated code printing.
 ** @param printResults Enable results printing.
 ** @param config The kernel configuration.
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param time_series The input data.
 ** @param openCLRunTime The OpenCL run time objects.
 ** @param clDeviceID The ID of the OpenCL device to use.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 */
template<typename DataType>
void testTimeDomainSigmaCut(const bool printCode, const bool printResults, const RFImConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, isa::OpenCL::OpenCLRunTime & openCLRunTime, const unsigned int clDeviceID, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Tune the OpenCL kernel to find best performing configuration for a certain scenario.
 **
 ** @param subbandDedispersion True if using subband dedispersion.
 ** @param parameters Tuning parameters.
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param time_series The input data.
 ** @param clPlatformID The ID of the OpenCL platform to use.
 ** @param clDeviceID The ID of the OpenCL device to use.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 */
template<typename DataType>
void tuneTimeDomainSigmaCut(const bool subbandDedispersion, const isa::OpenCL::TuningParameters & parameters, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, const unsigned int clPlatformID, const unsigned int clDeviceID, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Compute frequency domain sigma cut.
 ** Not optimized, just for testing purpose.
 **
 ** @param subbandDedispersion True if using subband dedispersion.
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param observation The observation object.
 ** @param time_series The input data.
 ** @param nrBins The number of bins for the bandpass.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 **
 ** @return The number of replaced samples.
 */
template<typename DataType>
std::uint64_t frequencyDomainSigmaCut(const bool subbandDedispersion, const DataOrdering & ordering, const ReplacementStrategy & replacement, const AstroData::Observation & observation, std::vector<DataType> & time_series, const unsigned int nrBins, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Generates the OpenCL code for the frequency domain sigma cut.
 **
 ** @param config The kernel configuration.
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param nrBins The number of bins for the bandpass.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 **
 ** @return String containing the generated code.
 */
template<typename DataType>
std::string * getFrequencyDomainSigmaCutOpenCL(const RFImConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const unsigned int nrBins, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Generates the OpenCL code for the frequency domain sigma cut.
 ** This function generates specialized code for the case in which the input is FrequencyTime ordered and flagged samples are replaced with the mean.
 **
 ** @param config The kernel configuration.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param nrBins The number of bins for the bandpass.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 **
 ** @return String containing the generated code.
 */
template<typename DataType>
std::string * getFrequencyDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean(const RFImConfig & config, const std::string & dataTypeName, const AstroData::Observation & observation, const unsigned int nrBins, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Test the OpenCL kernel by comparing results with C++ implementation.
 **
 ** @param printCode Enable generated code printing.
 ** @param printResults Enable results printing.
 ** @param config The kernel configuration.
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param time_series The input data.
 ** @param openCLRunTime The OpenCL run time objects.
 ** @param clDeviceID The ID of the OpenCL device to use.
 ** @param nrBins The number of bins for the bandpass.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 */
template<typename DataType>
void testFrequencyDomainSigmaCut(const bool printCode, const bool printResults, const RFImConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, isa::OpenCL::OpenCLRunTime & openCLRunTime, const unsigned int clDeviceID, const unsigned int nrBins, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Tune the OpenCL kernel to find best performing configuration for a certain scenario.
 **
 ** @param subbandDedispersion True if using subband dedispersion.
 ** @param parameters Tuning parameters.
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param time_series The input data.
 ** @param clPlatformID The ID of the OpenCL platform to use.
 ** @param clDeviceID The ID of the OpenCL device to use.
 ** @param nrBins The number of bins for the bandpass.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 */
template<typename DataType>
void tuneFrequencyDomainSigmaCut(const bool subbandDedispersion, const isa::OpenCL::TuningParameters & parameters, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, const unsigned int clPlatformID, const unsigned int clDeviceID, const unsigned int nrBins, const float sigmaCut, const unsigned int padding);

} // RFIm

inline bool RFIm::RFImConfig::getSubbandDedispersion() const
{
    return subbandDedispersion;
}

inline bool RFIm::RFImConfig::getConditionalReplacement() const
{
    return conditionalReplacement;
}

inline void RFIm::RFImConfig::setSubbandDedispersion(const bool subband)
{
    subbandDedispersion = subband;
}

inline void RFIm::RFImConfig::setConditionalReplacement(const bool replacement)
{
    conditionalReplacement = replacement;
}

inline std::string RFIm::RFImConfig::print() const
{
    return std::to_string(subbandDedispersion) + " " + std::to_string(conditionalReplacement) + " " + isa::OpenCL::KernelConf::print();
}

template<typename DataType>
std::uint64_t RFIm::timeDomainSigmaCut(const bool subbandDedispersion, const DataOrdering & ordering, const ReplacementStrategy & replacement, const AstroData::Observation & observation, std::vector<DataType> & time_series, const float sigmaCut, const unsigned int padding)
{
    std::uint64_t replacedSamples = 0;
    for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ )
    {
        if ( ordering == FrequencyTime )
        {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ )
            {
                isa::utils::Statistics<DataType> statistics;
                for ( unsigned int sample_id = 0; sample_id < observation.getNrSamplesPerDispersedBatch(subbandDedispersion); sample_id++ )
                {
                    statistics.addElement(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + sample_id));
                }
                for ( unsigned int sample_id = 0; sample_id < observation.getNrSamplesPerDispersedBatch(subbandDedispersion); sample_id++ )
                {
                    DataType sample_value = time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + sample_id);
                    if ( sample_value > (statistics.getMean() + (sigmaCut * statistics.getStandardDeviation())) )
                    {
                        replacedSamples++;
                        if ( replacement == ReplaceWithMean )
                        {
                            time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + sample_id) = statistics.getMean();
                        }
                    }
                }
            }
        }
    }
    return replacedSamples;
}

template<typename DataType>
std::string * RFIm::getTimeDomainSigmaCutOpenCL(const RFImConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const float sigmaCut, const unsigned int padding)
{
    if ( (ordering == FrequencyTime) && (replacement == ReplaceWithMean) )
    {
        return getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean<DataType>(config, dataTypeName, observation, sigmaCut, padding);
    }
    return new std::string();
}

template<typename DataType>
std::string * RFIm::getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean(const RFImConfig & config, const std::string & dataTypeName, const AstroData::Observation & observation, const float sigmaCut, const unsigned int padding)
{
    std::string *code = new std::string();
    // Kernel template
    *code = "__kernel void timeDomainSigmaCut(__global " + dataTypeName + " * const restrict time_series) {\n"
    + config.getIntType() + " threshold = 0;\n"
    "float delta = 0.0f;\n"
    "float mean = 0.0f;\n"
    "float sigma_cut = 0.0f;\n"
    + dataTypeName + " sample_value;\n"
    "__local float reductionCOU[" + std::to_string(config.getNrThreadsD0()) + "];\n"
    "__local float reductionMEA[" + std::to_string(config.getNrThreadsD0()) + "];\n"
    "__local float reductionVAR[" + std::to_string(config.getNrThreadsD0()) + "];\n"
    "<%LOCAL_VARIABLES%>"
    "\n"
    "// Compute mean and standard deviation\n"
    "for ( " + config.getIntType() + " sample_id = get_local_id(0) + " + std::to_string(config.getNrThreadsD0() * config.getNrItemsD0()) + "; sample_id < " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion())) + "; sample_id += " + std::to_string(config.getNrThreadsD0() * config.getNrItemsD0()) + " ) "
    "{\n"
    "<%LOCAL_COMPUTE%>"
    "}\n"
    "// Local reduction (if necessary)\n"
    "<%LOCAL_REDUCE%>"
    "reductionCOU[get_local_id(0)] = counter_0;\n"
    "reductionMEA[get_local_id(0)] = mean_0;\n"
    "reductionVAR[get_local_id(0)] = variance_0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Global reduction\n"
    "threshold = " + std::to_string(config.getNrThreadsD0() / 2) + ";\n"
    "for (" + config.getIntType() + " sample_id = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( (sample_id < threshold)) {\n"
    "delta = reductionMEA[sample_id + threshold] - mean_0;\n"
    "counter_0 += reductionCOU[sample_id + threshold];\n"
    "mean_0 = ((reductionCOU[sample_id] * mean_0) + (reductionCOU[sample_id + threshold] * reductionMEA[sample_id + threshold])) / counter_0;\n"
    "variance_0 += reductionVAR[sample_id + threshold] + ((delta * delta) * ((reductionCOU[sample_id] * reductionCOU[sample_id + threshold]) / counter_0));\n"
    "reductionCOU[sample_id] = counter_0;\n"
    "reductionMEA[sample_id] = mean_0;\n"
    "reductionVAR[sample_id] = variance_0;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "mean = reductionMEA[0];\n"
    "sigma_cut = (" + std::to_string(sigmaCut) + " * native_sqrt(reductionVAR[0] * " + std::to_string(1.0f/(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion()) - 1)) + "f));\n"
    "// Replace samples over the sigma cut with mean\n"
    "for (" + config.getIntType() + " sample_id = get_local_id(0); sample_id < " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion())) + "; sample_id += " + std::to_string(config.getNrThreadsD0() * config.getNrItemsD0()) + " ) "
    "{\n"
    "<%REPLACE%>"
    "}\n"
    "}\n";
    // Declaration of per thread variables
    std::string localVariablesTemplate = "float counter_<%ITEM_NUMBER%> = 1.0f;\n"
    "float variance_<%ITEM_NUMBER%> = 0.0f;\n"
    "float mean_<%ITEM_NUMBER%> = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>];\n";
    // Local compute
    // Version without boundary checks
    std::string localComputeNoCheckTemplate = "sample_value = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>];\n"
    "counter_<%ITEM_NUMBER%> += 1.0f;\n"
    "delta = sample_value - mean_<%ITEM_NUMBER%>;\n"
    "mean_<%ITEM_NUMBER%> += delta / counter_<%ITEM_NUMBER%>;\n"
    "variance_<%ITEM_NUMBER%> += delta * (sample_value - mean_<%ITEM_NUMBER%>);\n";
    // Version with boundary checks
    std::string localComputeCheckTemplate = "if ( sample_id + <%ITEM_OFFSET%> < " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion())) + " ) {\n"
    "sample_value = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>];\n"
    "counter_<%ITEM_NUMBER%> += 1.0f;\n"
    "delta = sample_value - mean_<%ITEM_NUMBER%>;\n"
    "mean_<%ITEM_NUMBER%> += delta / counter_<%ITEM_NUMBER%>;\n"
    "variance_<%ITEM_NUMBER%> += delta * (sample_value - mean_<%ITEM_NUMBER%>);\n"
    "}\n";
    // In-thread reduction
    std::string localReduceTemplate = "delta = mean_<%ITEM_NUMBER%> - mean_0;\n"
    "counter_0 += counter_<%ITEM_NUMBER%>;\n"
    "mean_0 = (((counter_0 - counter_<%ITEM_NUMBER%>) * mean_0) + (counter_<%ITEM_NUMBER%> * mean_<%ITEM_NUMBER%>)) / counter_0;\n"
    "variance_0 += variance_<%ITEM_NUMBER%> + ((delta * delta) * (((counter_0 - counter_<%ITEM_NUMBER%>) * counter_<%ITEM_NUMBER%>) / counter_0));\n";
    std::string replaceConditionTemplate = "if ( time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>] > (mean + sigma_cut) ) {\n"
    "time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>] = mean;"
    "}\n";
    std::string replaceTemplate = "sample_value = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>];\n"
    "time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>] = (sample_value * (convert_" + dataTypeName + "(sample_value < (mean + sigma_cut)))) + (mean * (convert_" + dataTypeName + "(sample_value > mean + sigma_cut)));\n";
    // End of kernel template
    // Code generation
    std::string localVariables;
    std::string localCompute;
    std::string localReduce;
    std::string replace;
    for ( unsigned int item = 0; item < config.getNrItemsD0(); item++ )
    {
        std::string * temp;
        std::string itemString = std::to_string(item);
        std::string itemOffsetString = std::to_string(item * config.getNrThreadsD0());
        temp = isa::utils::replace(&localVariablesTemplate, "<%ITEM_NUMBER%>", itemString);
        if ( item == 0 )
        {
            temp = isa::utils::replace(temp, " + <%ITEM_OFFSET%>", std::string(), true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%ITEM_OFFSET%>", itemOffsetString, true);
        }
        localVariables.append(*temp);
        delete temp;
        if ( (observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion()) % (config.getNrThreadsD0() * config.getNrItemsD0())) == 0 )
        {
            temp = isa::utils::replace(&localComputeNoCheckTemplate, "<%ITEM_NUMBER%>", itemString);
        }
        else
        {
            temp = isa::utils::replace(&localComputeCheckTemplate, "<%ITEM_NUMBER%>", itemString);
        }
        if ( item == 0 )
        {
            temp = isa::utils::replace(temp, " + <%ITEM_OFFSET%>", std::string(), true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%ITEM_OFFSET%>", itemOffsetString, true);
        }
        localCompute.append(*temp);
        delete temp;
        if ( item > 0 )
        {
            temp = isa::utils::replace(&localReduceTemplate, "<%ITEM_NUMBER%>", itemString);
            localReduce.append(*temp);
            delete temp;
        }
        if ( config.getConditionalReplacement() )
        {
            temp = isa::utils::replace(&replaceConditionTemplate, "<%ITEM_NUMBER%>", itemString);
        }
        else
        {
            temp = isa::utils::replace(&replaceTemplate, "<%ITEM_NUMBER%>", itemString);
        }
        if ( item == 0 )
        {
            temp = isa::utils::replace(temp, " + <%ITEM_OFFSET%>", std::string(), true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%ITEM_OFFSET%>", itemOffsetString, true);
        }
        replace.append(*temp);
    }
    code = isa::utils::replace(code, "<%LOCAL_VARIABLES%>", localVariables, true);
    code = isa::utils::replace(code, "<%LOCAL_COMPUTE%>", localCompute, true);
    code = isa::utils::replace(code, "<%LOCAL_REDUCE%>", localReduce, true);
    code = isa::utils::replace(code, "<%REPLACE%>", replace, true);
    return code;
}

template<typename DataType>
void RFIm::testTimeDomainSigmaCut(const bool printCode, const bool printResults, const RFImConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, isa::OpenCL::OpenCLRunTime & openCLRunTime, const unsigned int clDeviceID, const float sigmaCut, const unsigned int padding)
{
    std::uint64_t wrongSamples = 0;
    std::uint64_t replacedSamples = 0;
    std::vector<DataType> test_time_series, control_time_series;
    test_time_series = time_series;
    control_time_series = time_series;
    cl::Buffer device_time_series;
    cl::Kernel * kernel = nullptr;
    // Generate OpenCL code
    std::string * code = getTimeDomainSigmaCutOpenCL<DataType>(config, ordering, replacement, dataTypeName, observation, sigmaCut, padding);
    if ( printCode )
    {
        std::cout << std::endl;
        std::cout << *code << std::endl;
        std::cout << std::endl;
        delete code;
        return;
    }
    // Execute control code
    replacedSamples = timeDomainSigmaCut(config.getSubbandDedispersion(), ordering, replacement, observation, control_time_series, sigmaCut, padding);
    // Execute OpenCL code
    try
    {
        device_time_series = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_WRITE, test_time_series.size() * sizeof(DataType), 0, 0);
    }
    catch ( const cl::Error & err )
    {
        std::cerr << "OpenCL device memory allocation error: " << std::to_string(err.err()) << "." << std::endl;
        throw err;
    }
    try
    {
        openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(device_time_series, CL_FALSE, 0, test_time_series.size() * sizeof(DataType), reinterpret_cast<void *>(test_time_series.data()));
    }
    catch( const cl::Error & err )
    {
        std::cerr <<  "OpenCL transfer H2D error: " << std::to_string(err.err()) << "." << std::endl;
    }
    try
    {
        kernel = isa::OpenCL::compile("timeDomainSigmaCut", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
    }
    catch ( const isa::OpenCL::OpenCLError & err )
    {
        std::cerr << err.what() << std::endl;
        delete code;
    }
    delete code;
    try
    {
        cl::NDRange global, local;
        global = cl::NDRange(config.getNrThreadsD0(), observation.getNrChannels(), observation.getNrBeams());
        local = cl::NDRange(config.getNrThreadsD0(), 1, 1);
        kernel->setArg(0, device_time_series);
        openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
        openCLRunTime.queues->at(clDeviceID)[0].enqueueReadBuffer(device_time_series, CL_TRUE, 0, test_time_series.size() * sizeof(DataType), reinterpret_cast<void *>(test_time_series.data()));
    }
    catch ( const cl::Error & err )
    {
        std::cerr << "OpenCL kernel execution error: " << std::to_string(err.err()) << "." << std::endl;
        delete kernel;
    }
    delete kernel;
    // Compare results
    for ( unsigned int beam  = 0; beam < observation.getNrBeams(); beam++ )
    {
        if ( ordering == FrequencyTime )
        {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ )
            {
                for ( unsigned int sample_id = 0; sample_id < observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion()); sample_id++ )
                {
                    if ( !isa::utils::same(test_time_series.at(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + sample_id)), control_time_series.at(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + sample_id))) )
                    {
                        wrongSamples++;
                    }
                    if ( printResults )
                    {
                        std::cout << static_cast<double>(test_time_series.at(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + sample_id))) << " " << static_cast<double>(control_time_series.at(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + sample_id))) << " ; ";
                    }
                }
                if ( printResults )
                {
                    std::cout << std::endl;
                }
            }
        }
    }
    // Print test results
    if ( wrongSamples > 0 )
    {
        std::cout << std::fixed;
        std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast<std::uint64_t>(observation.getNrBeams()) * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion()) << "%)." << std::endl;
    }
    else
    {
        std::cout << "TEST PASSED." << std::endl;
    }
}

template<typename DataType>
void RFIm::tuneTimeDomainSigmaCut(const bool subbandDedispersion, const isa::OpenCL::TuningParameters & parameters, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, const unsigned int clPlatformID, const unsigned int clDeviceID, const float sigmaCut, const unsigned int padding)
{
    bool initializeDevice = true;
    isa::utils::Timer timer;
    double bestTime = std::numeric_limits<double>::max();
    RFImConfig bestConfig;
    std::vector<RFImConfig> configurations;
    isa::OpenCL::OpenCLRunTime openCLRunTime;
    cl::Event clEvent;
    cl::Buffer device_time_series;
    // Generate valid configurations
    for ( unsigned int threads = parameters.getMinThreads(); threads <= parameters.getMaxThreads(); threads *= 2 )
    {
        for ( unsigned int items = 1; (items * 3) + 8 <= parameters.getMaxItems(); items++ )
        {
            if ( (threads * items) > observation.getNrSamplesPerDispersedBatch(subbandDedispersion) )
            {
                break;
            }
            RFImConfig baseConfig, tempConfig;
            baseConfig.setSubbandDedispersion(subbandDedispersion);
            baseConfig.setNrThreadsD0(threads);
            baseConfig.setNrItemsD0(items);
            // conditional = 0, int = 0
            tempConfig = baseConfig;
            tempConfig.setConditionalReplacement(false);
            tempConfig.setIntType(0);
            configurations.push_back(tempConfig);
            // conditional = 0, int = 1
            tempConfig = baseConfig;
            tempConfig.setConditionalReplacement(false);
            tempConfig.setIntType(1);
            configurations.push_back(tempConfig);
            // conditional = 1, int = 0
            tempConfig = baseConfig;
            tempConfig.setConditionalReplacement(true);
            tempConfig.setIntType(0);
            configurations.push_back(tempConfig);
            // conditional = 1, int = 1
            tempConfig = baseConfig;
            tempConfig.setConditionalReplacement(true);
            tempConfig.setIntType(1);
            configurations.push_back(tempConfig);
        }
    }
    // Test performance of each configuration
    std::cout << std::fixed;
    if ( !parameters.getBestMode() )
    {
        std::cout << std::endl << "# nrBeams nrChannels nrSamplesPerDispersedBatch sigma *configuration* time stdDeviation COV" << std::endl << std::endl;
    }
    for ( auto configuration = configurations.begin(); configuration != configurations.end(); ++configuration )
    {
        cl::Kernel * kernel = nullptr;
        // Generate kernel
        std::string * code = getTimeDomainSigmaCutOpenCL<DataType>(*configuration, ordering, replacement, dataTypeName, observation, sigmaCut, padding);
        if ( initializeDevice )
        {
            isa::OpenCL::initializeOpenCL(clPlatformID, 1, openCLRunTime);
            try
            {
                device_time_series = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_WRITE, time_series.size() * sizeof(DataType), 0, 0);
            }
            catch ( const cl::Error & err )
            {
                std::cerr << "OpenCL device memory allocation error: " << std::to_string(err.err()) << "." << std::endl;
                throw err;
            }
            initializeDevice = false;
        }
        try
        {
            kernel = isa::OpenCL::compile("timeDomainSigmaCut", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
        }
        catch ( const isa::OpenCL::OpenCLError & err )
        {
            std::cerr << err.what() << std::endl;
            delete code;
        }
        delete code;
        try
        {
            cl::NDRange global, local;
            global = cl::NDRange((*configuration).getNrThreadsD0(), observation.getNrChannels(), observation.getNrBeams());
            local = cl::NDRange((*configuration).getNrThreadsD0(), 1, 1);
            kernel->setArg(0, device_time_series);
            // Warm-up run
            openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(device_time_series, CL_FALSE, 0, time_series.size() * sizeof(DataType), reinterpret_cast<const void *>(time_series.data()), nullptr, &clEvent);
            clEvent.wait();
            openCLRunTime.queues->at(clDeviceID)[0].finish();
            openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, nullptr, &clEvent);
            clEvent.wait();
            // Tuning runs
            for ( unsigned int iteration = 0; iteration < parameters.getNrIterations(); iteration++ )
            {
                openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(device_time_series, CL_FALSE, 0, time_series.size() * sizeof(DataType), reinterpret_cast<const void *>(time_series.data()), nullptr, &clEvent);
                clEvent.wait();
                openCLRunTime.queues->at(clDeviceID)[0].finish();
                timer.start();
                openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, nullptr, &clEvent);
                clEvent.wait();
                timer.stop();
            }
        }
        catch ( const cl::Error & err )
        {
            std::cerr << "OpenCL kernel error during tuning: " << std::to_string(err.err()) << "." << std::endl;
            delete kernel;
            if (err.err() == -4 || err.err() == -61)
            {
                throw err;
            }
            initializeDevice = true;
            break;
        }
        delete kernel;
        if ( timer.getAverageTime() < bestTime )
        {
            bestTime = timer.getAverageTime();
            bestConfig = *configuration;
        }
        if ( !parameters.getBestMode() )
        {
            std::cout << observation.getNrBeams() << " " << observation.getNrChannels() << " ";
            std::cout << observation.getNrSamplesPerDispersedBatch(subbandDedispersion) << " ";
            std::cout.precision(2);
            std::cout << sigmaCut << " ";
            std::cout << (*configuration).print() << " ";
            std::cout.precision(6);
            std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation();
            std::cout  << std::endl;
        }
    }
    if ( parameters.getBestMode() )
    {
        std::cout.precision(2);
        std::cout << observation.getNrSamplesPerDispersedBatch(subbandDedispersion) << " " << sigmaCut << " ";
        std::cout << bestConfig.print() << std::endl;
    }
    else
    {
        std::cout << std::endl;
    }
}

template<typename DataType>
std::uint64_t RFIm::frequencyDomainSigmaCut(const bool subbandDedispersion, const DataOrdering & ordering, const ReplacementStrategy & replacement, const AstroData::Observation & observation, std::vector<DataType> & time_series, const unsigned int nrBins, const float sigmaCut, const unsigned int padding)
{
    std::uint64_t replacedSamples = 0;
    for ( unsigned int beam = 0; beam < observation.getNrBeams(); beam++ )
    {
        if ( ordering == FrequencyTime )
        {
            for ( unsigned int sample_id = 0; sample_id < observation.getNrSamplesPerDispersedBatch(subbandDedispersion); sample_id++ )
            {
                isa::utils::Statistics<DataType> statistics;
                isa::utils::Statistics<DataType> statistics_corrected;
                isa::utils::Statistics<DataType> * local_statistics = new isa::utils::Statistics<DataType> [(int) std::ceil(observation.getNrChannels()/nrBins)];

                int bin;
                for ( unsigned int channel = 0, bin=0; channel < observation.getNrChannels(); channel++ )
                {
                    if ( (channel != 0) && ((channel % nrBins) == 0) )
                        bin++;

                    local_statistics[bin].addElement(
                        time_series.at(
                                (beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) +
                                (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) +
                                sample_id
                        )
                    );
                }

                for ( unsigned int channel = 0, bin=0; channel < observation.getNrChannels(); channel++ )
                {
                    if ( (channel != 0) && ((channel % nrBins) == 0) )
                        bin++;

                    statistics.addElement(
                        time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + sample_id)
                    );

                    statistics_corrected.addElement(
                        time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + sample_id) - local_statistics[bin].getMean()
                    );
                }

                for ( unsigned int channel = 0, bin=0; channel < observation.getNrChannels(); channel++ )
                {
                    if ( (channel != 0) && ((channel % nrBins) == 0) )
                        bin++;

                    DataType sample_value = time_series.at(
                        (beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) +
                        (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) +
                        sample_id
                    );

                    if ( (sample_value-local_statistics[bin].getMean()) > (statistics_corrected.getMean() + (sigmaCut * statistics_corrected.getStandardDeviation())) )
                    {
                        replacedSamples++;
                        if ( replacement == ReplaceWithMean )
                        {
                            time_series.at(
                                (beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) +
                                (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) +
                                sample_id
                            ) = local_statistics[bin].getMean();
                        }
                    }
                }
            }
        }
    }
    return replacedSamples;
}

template<typename DataType>
std::string * RFIm::getFrequencyDomainSigmaCutOpenCL(const RFImConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const unsigned int nrBins, const float sigmaCut, const unsigned int padding)
{
     if ( (ordering == FrequencyTime) && (replacement == ReplaceWithMean) )
    {
        return getFrequencyDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean<DataType>(config, dataTypeName, observation, nrBins, sigmaCut, padding);
    }
    return new std::string();
}

template<typename DataType>
std::string * RFIm::getFrequencyDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean(const RFImConfig & config, const std::string & dataTypeName, const AstroData::Observation & observation, const unsigned int nrBins, const float sigmaCut, const unsigned int padding)
{
    unsigned int binSize = observation.getNrChannels() / nrBins;
    std::string *code = new std::string();
    // Kernel template
    *code = "__kernel void frequencyDomainSigmaCut(__global " + dataTypeName + " * const restrict time_series) {\n"
    "float delta = 0.0f;\n"
    "float sigma_cut = 0.0f;\n"
    + dataTypeName + " sample_value;\n"
    "<%LOCAL_VARIABLES%>"
    "<%BIN_VARIABLES%>"
    "\n"
    "// Stop unnecessary threads\n"
    "if ( ((get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)) >= " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion())) + " )\n"
    "{\n"
    "return;\n"
    "}\n"
    "// Compute mean for each bin\n"
    "<%COMPUTE_MEAN_BIN%>"
    "// Compute mean and standard deviation corrected for local bin\n"
    "<%LOCAL_COMPUTE%>"
    "// Local reduction (if necessary)\n"
    "<%LOCAL_REDUCE%>"
    "sigma_cut = (" + std::to_string(sigmaCut) + " * native_sqrt(variance_0 * " + std::to_string(1.0f/(observation.getNrChannels() - 1)) + "f));\n"
    "// Replace samples over the sigma cut with mean\n"
    "<%REPLACE%>"
    "}\n";
    // Declaration of per thread variables
    std::string localVariablesTemplate = "float counter_<%ITEM_NUMBER%> = 1.0f;\n"
    "float variance_<%ITEM_NUMBER%> = 0.0f;\n"
    "float mean_<%ITEM_NUMBER%> = 0.0f;\n";
    std::string binVariablesTemplate = "float mean_bin_<%BIN_NUMBER%> = 0.0f;\n";
    // Compute mean for a single bin
    std::string binComputeMeanTemplate = "mean_bin_<%BIN_NUMBER%> = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + ((<%BIN_NUMBER%> * " + std::to_string(binSize) + ") * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)];\n"
    "for ( " + config.getIntType() + " channel_id = (<%BIN_NUMBER%> * " + std::to_string(binSize) + ") + 1; channel_id < (<%BIN_NUMBER%> + 1) * " + std::to_string(binSize) "; channel_id++ )\n"
    "{\n"
    "mean_bin_<%BIN_NUMBER%> += time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (channel_id * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)];\n"
    "}\n"
    "mean_bin_<%BIN_NUMBER%> /= " + std::to_string(binSize) + ";\n";
    // Local compute
    // Version without boundary checks
    std::string localComputeNoCheckTemplate = "mean_<%ITEM_NUMBER%> = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (((<%BIN_NUMBER%> * " + std::to_string(binSize) + ") + <%ITEM_NUMBER%>) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)];\n"
    "for ( " + config.getIntType() + " channel_id = ((<%BIN_NUMBER%> * " + std::to_string(binSize) + ") + <%ITEM_NUMBER%>) + " + std::to_string(config.getNrItemsD1()) + "; channel_id < (<%BIN_NUMBER%> + 1) * " + std::to_string(binSize) "; channel_id += " + std::to_string(config.getNrItemsD1()) + " )\n"
    "{\n"
    "sample_value = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + ((channel_id + <%ITEM_NUMBER%>) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)] - mean_bin_<%BIN_NUMBER%>;\n"
    "counter_<%ITEM_NUMBER%> += 1.0f;\n"
    "delta = sample_value - mean_<%ITEM_NUMBER%>;\n"
    "mean_<%ITEM_NUMBER%> += delta / counter_<%ITEM_NUMBER%>;\n"
    "variance_<%ITEM_NUMBER%> += delta * (sample_value - mean_<%ITEM_NUMBER%>);\n"
    "}\n";
    // Version with boundary checks
    std::string localComputeCheckTemplate = "mean_<%ITEM_NUMBER%> = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (((<%BIN_NUMBER%> * " + std::to_string(binSize) + ") + <%ITEM_NUMBER%>) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)];\n"
    "for ( " + config.getIntType() + " channel_id = ((<%BIN_NUMBER%> * " + std::to_string(binSize) + ") + <%ITEM_NUMBER%>) + " + std::to_string(config.getNrItemsD1()) + "; channel_id < (<%BIN_NUMBER%> + 1) * " + std::to_string(binSize) "; channel_id += " + std::to_string(config.getNrItemsD1()) + " )\n"
    "{\n"
    "if ( channel_id + <%ITEM_NUMBER%> < " + std::to_string(binSize) + " ) {\n"
    "sample_value = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + ((channel_id + <%ITEM_NUMBER%>) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)] - mean_bin_<%BIN_NUMBER%>;\n"
    "counter_<%ITEM_NUMBER%> += 1.0f;\n"
    "delta = sample_value - mean_<%ITEM_NUMBER%>;\n"
    "mean_<%ITEM_NUMBER%> += delta / counter_<%ITEM_NUMBER%>;\n"
    "variance_<%ITEM_NUMBER%> += delta * (sample_value - mean_<%ITEM_NUMBER%>);\n"
    "}\n"
    "}\n";
    // In-thread reduction
    std::string localReduceTemplate = "delta = mean_<%ITEM_NUMBER%> - mean_0;\n"
    "counter_0 += counter_<%ITEM_NUMBER%>;\n"
    "mean_0 = (((counter_0 - counter_<%ITEM_NUMBER%>) * mean_0) + (counter_<%ITEM_NUMBER%> * mean_<%ITEM_NUMBER%>)) / counter_0;\n"
    "variance_0 += variance_<%ITEM_NUMBER%> + ((delta * delta) * (((counter_0 - counter_<%ITEM_NUMBER%>) * counter_<%ITEM_NUMBER%>) / counter_0));\n";
    // Replace with boundary checks
    std::string replaceConditionTemplate = "for ( " + config.getIntType() + " channel_id = <%BIN_NUMBER%> * " + std::to_string(binSize) + "; channel_id < (<%BIN_NUMBER%> + 1) * " + std::to_string(binSize) "; channel_id++ )\n"
    "{\n"
    "if ( time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + ((channel_id + <%ITEM_NUMBER%>) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)] - mean_bin_<%BIN_NUMBER%> > (mean_0 + sigma_cut) ) {\n"
    "time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + ((channel_id + <%ITEM_NUMBER%>) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)] = mean_bin_<%BIN_NUMBER%>;"
    "}\n"
    "}\n";
    // Replace without boundary checks
    std::string replaceTemplate = "for ( " + config.getIntType() + " channel_id = <%BIN_NUMBER%> * " + std::to_string(binSize) + "; channel_id < (<%BIN_NUMBER%> + 1) * " + std::to_string(binSize) "; channel_id++ )\n"
    "{\n"
    "sample_value = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + ((channel_id + <%ITEM_NUMBER%>) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)] - mean_bin_<%BIN_NUMBER%>;\n"
    "time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + ((channel_id + <%ITEM_NUMBER%>) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(config.getNrThreadsD0()) + ") + get_local_id(0)] = (sample_value * (convert_" + dataTypeName + "(sample_value < (mean_0 + sigma_cut)))) + (mean_bin_<%BIN_NUMBER%> * (convert_" + dataTypeName + "(sample_value > mean_0 + sigma_cut)));\n"
    "}\n";
    // End of kernel template
    // Code generation
    std::string localVariables;
    std::string binVariables;
    std::string binComputeMean;
    std::string localCompute;
    std::string localReduce;
    std::string replace;
    for ( unsigned int item = 0; item < config.getNrItemsD1(); item++ )
    {
        std::string * temp;
        std::string itemString = std::to_string(item);
        temp = isa::utils::replace(&localVariablesTemplate, "<%ITEM_NUMBER%>", itemString);
        localVariables.append(*temp);
        delete temp;
    }
    for ( unsigned int bin = 0; bin < nrBins; bin++ )
    {
      std::string * temp;
      std::string binString = std::to_string(bin);
      temp = isa::utils::replace(&binVariablesTemplate, "<%BIN_NUMBER%>", binString);
      binVariables.append(*temp);
      delete temp;
      temp = isa::utils::replace(&binComputeMeanTemplate, "<%BIN_NUMBER%>", binString);
      binComputeMean.append(*temp);
      delete temp;
      for ( unsigned int item = 0; item < config.getNrItemsD1(); item++ )
      {
          if ( (binSize % config.getNrItemsD1()) == 0 )
          {
              temp = isa::utils::replace(&localComputeNoCheckTemplate, "<%ITEM_NUMBER%>", itemString);
          }
          else
          {
              temp = isa::utils::replace(&localComputeCheckTemplate, "<%ITEM_NUMBER%>", itemString);
          }
          if ( item == 0 )
          {
              temp = isa::utils::replace(temp, " + 0", std::string(), true);
          }
          temp = isa::utils::replace(temp, "<%BIN_NUMBER%>", binString, true);
          localCompute.append(*temp);
          delete temp;
          if ( item > 0 )
          {
              temp = isa::utils::replace(&localReduceTemplate, "<%ITEM_NUMBER%>", itemString);
              localReduce.append(*temp);
              delete temp;
          }
          if ( config.getConditionalReplacement() )
          {
              temp = isa::utils::replace(&replaceConditionTemplate, "<%ITEM_NUMBER%>", itemString);
          }
          else
          {
              temp = isa::utils::replace(&replaceTemplate, "<%ITEM_NUMBER%>", itemString);
          }
          if ( item == 0 )
          {
              temp = isa::utils::replace(temp, " + 0", std::string(), true);
          }
          temp = isa::utils::replace(temp, "<%BIN_NUMBER%>", binString, true);
          replace.append(*temp);
      }
    }
    code = isa::utils::replace(code, "<%LOCAL_VARIABLES%>", localVariables, true);
    code = isa::utils::replace(code, "<%BIN_VARIABLES%>", binVariables, true);
    code = isa::utils::replace(code, "<%COMPUTE_MEAN_BIN%>", binComputeMean, true);
    code = isa::utils::replace(code, "<%LOCAL_COMPUTE%>", localCompute, true);
    code = isa::utils::replace(code, "<%LOCAL_REDUCE%>", localReduce, true);
    code = isa::utils::replace(code, "<%REPLACE%>", replace, true);
    return code;
}

template<typename DataType>
void RFIm::testFrequencyDomainSigmaCut(const bool printCode, const bool printResults, const RFImConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, isa::OpenCL::OpenCLRunTime & openCLRunTime, const unsigned int clDeviceID, const unsigned int nrBins, const float sigmaCut, const unsigned int padding)
{
    std::uint64_t wrongSamples = 0;
    std::uint64_t replacedSamples = 0;
    std::vector<DataType> test_time_series, control_time_series;
    test_time_series = time_series;
    control_time_series = time_series;
    cl::Buffer device_time_series;
    cl::Kernel * kernel = nullptr;
    // Generate OpenCL code
    std::string * code = getFrequencyDomainSigmaCutOpenCL<DataType>(config, ordering, replacement, dataTypeName, observation, nrBins, sigmaCut, padding);
    if ( printCode )
    {
        std::cout << std::endl;
        std::cout << *code << std::endl;
        std::cout << std::endl;
        delete code;
        return;
    }
    // Execute control code
    replacedSamples = frequencyDomainSigmaCut(config.getSubbandDedispersion(), ordering, replacement, observation, control_time_series, nrBins, igmaCut, padding);
    // Execute OpenCL code
    try
    {
        device_time_series = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_WRITE, test_time_series.size() * sizeof(DataType), 0, 0);
    }
    catch ( const cl::Error & err )
    {
        std::cerr << "OpenCL device memory allocation error: " << std::to_string(err.err()) << "." << std::endl;
        throw err;
    }
    try
    {
        openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(device_time_series, CL_FALSE, 0, test_time_series.size() * sizeof(DataType), reinterpret_cast<void *>(test_time_series.data()));
    }
    catch( const cl::Error & err )
    {
        std::cerr <<  "OpenCL transfer H2D error: " << std::to_string(err.err()) << "." << std::endl;
    }
    try
    {
        kernel = isa::OpenCL::compile("frequencyDomainSigmaCut", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
    }
    catch ( const isa::OpenCL::OpenCLError & err )
    {
        std::cerr << err.what() << std::endl;
        delete code;
    }
    delete code;
    try
    {
        cl::NDRange global, local;
        global = cl::NDRange(isa::utils::pad(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion()), config.getNrThreadsD0()), 1, observation.getNrBeams());
        local = cl::NDRange(config.getNrThreadsD0(), 1, 1);
        kernel->setArg(0, device_time_series);
        openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
        openCLRunTime.queues->at(clDeviceID)[0].enqueueReadBuffer(device_time_series, CL_TRUE, 0, test_time_series.size() * sizeof(DataType), reinterpret_cast<void *>(test_time_series.data()));
    }
    catch ( const cl::Error & err )
    {
        std::cerr << "OpenCL kernel execution error: " << std::to_string(err.err()) << "." << std::endl;
        delete kernel;
    }
    delete kernel;
    // Compare results
    for ( unsigned int beam  = 0; beam < observation.getNrBeams(); beam++ )
    {
        if ( ordering == FrequencyTime )
        {
            for ( unsigned int channel = 0; channel < observation.getNrChannels(); channel++ )
            {
                for ( unsigned int sample_id = 0; sample_id < observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion()); sample_id++ )
                {
                    if ( !isa::utils::same(test_time_series.at(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + sample_id)), control_time_series.at(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + sample_id))) )
                    {
                        wrongSamples++;
                    }
                    if ( printResults )
                    {
                        std::cout << static_cast<double>(test_time_series.at(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + sample_id))) << " " << static_cast<double>(control_time_series.at(time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + sample_id))) << " ; ";
                    }
                }
                if ( printResults )
                {
                    std::cout << std::endl;
                }
            }
        }
    }
    // Print test results
    if ( wrongSamples > 0 )
    {
        std::cout << std::fixed;
        std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast<std::uint64_t>(observation.getNrBeams()) * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion()) << "%)." << std::endl;
    }
    else
    {
        std::cout << "TEST PASSED." << std::endl;
    }
}

template<typename DataType>
void RFIm::tuneFrequencyDomainSigmaCut(const bool subbandDedispersion, const isa::OpenCL::TuningParameters & parameters, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, const unsigned int clPlatformID, const unsigned int clDeviceID, const unsigned int nrBins, const float sigmaCut, const unsigned int padding)
{
    bool initializeDevice = true;
    isa::utils::Timer timer;
    double bestTime = std::numeric_limits<double>::max();
    RFImConfig bestConfig;
    std::vector<RFImConfig> configurations;
    isa::OpenCL::OpenCLRunTime openCLRunTime;
    cl::Event clEvent;
    cl::Buffer device_time_series;
    // Generate valid configurations
    for ( unsigned int threads = parameters.getMinThreads(); threads <= parameters.getMaxThreads(); threads *= 2 )
    {
        for ( unsigned int items = 1; (items * 3) + 3 + nrBins <= parameters.getMaxItems(); items++ )
        {
            if ( threads > observation.getNrSamplesPerDispersedBatch(subbandDedispersion) )
            {
                break;
            }
            RFImConfig baseConfig, tempConfig;
            baseConfig.setSubbandDedispersion(subbandDedispersion);
            baseConfig.setNrThreadsD0(threads);
            baseConfig.setNrItemsD1(items);
            // conditional = 0, int = 0
            tempConfig = baseConfig;
            tempConfig.setConditionalReplacement(false);
            tempConfig.setIntType(0);
            configurations.push_back(tempConfig);
            // conditional = 0, int = 1
            tempConfig = baseConfig;
            tempConfig.setConditionalReplacement(false);
            tempConfig.setIntType(1);
            configurations.push_back(tempConfig);
            // conditional = 1, int = 0
            tempConfig = baseConfig;
            tempConfig.setConditionalReplacement(true);
            tempConfig.setIntType(0);
            configurations.push_back(tempConfig);
            // conditional = 1, int = 1
            tempConfig = baseConfig;
            tempConfig.setConditionalReplacement(true);
            tempConfig.setIntType(1);
            configurations.push_back(tempConfig);
        }
    }
    // Test performance of each configuration
    std::cout << std::fixed;
    if ( !parameters.getBestMode() )
    {
        std::cout << std::endl << "# nrBeams nrChannels nrSamplesPerDispersedBatch sigma *configuration* time stdDeviation COV" << std::endl << std::endl;
    }
    for ( auto configuration = configurations.begin(); configuration != configurations.end(); ++configuration )
    {
        cl::Kernel * kernel = nullptr;
        // Generate kernel
        std::string * code = getFrequencyDomainSigmaCutOpenCL<DataType>(*configuration, ordering, replacement, dataTypeName, observation, nrBins, sigmaCut, padding);
        if ( initializeDevice )
        {
            isa::OpenCL::initializeOpenCL(clPlatformID, 1, openCLRunTime);
            try
            {
                device_time_series = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_WRITE, time_series.size() * sizeof(DataType), 0, 0);
            }
            catch ( const cl::Error & err )
            {
                std::cerr << "OpenCL device memory allocation error: " << std::to_string(err.err()) << "." << std::endl;
                throw err;
            }
            initializeDevice = false;
        }
        try
        {
            kernel = isa::OpenCL::compile("frequencyDomainSigmaCut", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
        }
        catch ( const isa::OpenCL::OpenCLError & err )
        {
            std::cerr << err.what() << std::endl;
            delete code;
        }
        delete code;
        try
        {
            cl::NDRange global, local;
            global = cl::NDRange(isa::utils::pad(observation.getNrSamplesPerDispersedBatch(subbandDedispersion), (*configuration).getNrThreadsD0()), 1, observation.getNrBeams());
            local = cl::NDRange((*configuration).getNrThreadsD0(), 1, 1);
            kernel->setArg(0, device_time_series);
            // Warm-up run
            openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(device_time_series, CL_FALSE, 0, time_series.size() * sizeof(DataType), reinterpret_cast<const void *>(time_series.data()), nullptr, &clEvent);
            clEvent.wait();
            openCLRunTime.queues->at(clDeviceID)[0].finish();
            openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, nullptr, &clEvent);
            clEvent.wait();
            // Tuning runs
            for ( unsigned int iteration = 0; iteration < parameters.getNrIterations(); iteration++ )
            {
                openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(device_time_series, CL_FALSE, 0, time_series.size() * sizeof(DataType), reinterpret_cast<const void *>(time_series.data()), nullptr, &clEvent);
                clEvent.wait();
                openCLRunTime.queues->at(clDeviceID)[0].finish();
                timer.start();
                openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, nullptr, &clEvent);
                clEvent.wait();
                timer.stop();
            }
        }
        catch ( const cl::Error & err )
        {
            std::cerr << "OpenCL kernel error during tuning: " << std::to_string(err.err()) << "." << std::endl;
            delete kernel;
            if (err.err() == -4 || err.err() == -61)
            {
                throw err;
            }
            initializeDevice = true;
            break;
        }
        delete kernel;
        if ( timer.getAverageTime() < bestTime )
        {
            bestTime = timer.getAverageTime();
            bestConfig = *configuration;
        }
        if ( !parameters.getBestMode() )
        {
            std::cout << observation.getNrBeams() << " " << observation.getNrChannels() << " ";
            std::cout << observation.getNrSamplesPerDispersedBatch(subbandDedispersion) << " ";
            std::cout.precision(2);
            std::cout << sigmaCut << " ";
            std::cout << (*configuration).print() << " ";
            std::cout.precision(6);
            std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation();
            std::cout  << std::endl;
        }
    }
    if ( parameters.getBestMode() )
    {
        std::cout.precision(2);
        std::cout << observation.getNrSamplesPerDispersedBatch(subbandDedispersion) << " " << sigmaCut << " ";
        std::cout << bestConfig.print() << std::endl;
    }
    else
    {
        std::cout << std::endl;
    }
}
