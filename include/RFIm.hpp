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

#pragma once

namespace RFIm
{

/**
 ** @brief RFI specific kernel configuration.
 */
class RFIConfig : public isa::OpenCL::KernelConf
{
public:
    RFIConfig();
    ~RFIConfig();
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
    TimeDomainSigmaCut
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
 */
template<typename DataType>
void timeDomainSigmaCut(const bool subbandDedispersion, const DataOrdering & ordering, const ReplacementStrategy & replacement, const AstroData::Observation & observation, std::vector<DataType> & time_series, const float sigmaCut, const unsigned int padding);

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
std::string * getTimeDomainSigmaCutOpenCL(const RFIConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const float sigmaCut, const unsigned int padding);

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
std::string * getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean(const RFIConfig & config, const std::string & dataTypeName, const AstroData::Observation & observation, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Test the OpenCL kernel by comparing results with C++ implementation.
 **
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
void testTimeDomainSigmaCut(const bool printResults, const RFIConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, isa::OpenCL::OpenCLRunTime & openCLRunTime, const unsigned int clDeviceID, const float sigmaCut, const unsigned int padding);

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

} // RFIm

inline bool RFIm::RFIConfig::getSubbandDedispersion() const
{
    return subbandDedispersion;
}

inline bool RFIm::RFIConfig::getConditionalReplacement() const
{
    return conditionalReplacement;
}

inline void RFIm::RFIConfig::setSubbandDedispersion(const bool subband)
{
    subbandDedispersion = subband;
}

inline void RFIm::RFIConfig::setConditionalReplacement(const bool replacement)
{
    conditionalReplacement = replacement;
}

inline std::string RFIm::RFIConfig::print() const
{
    return std::to_string(subbandDedispersion) + " " + std::to_string(conditionalReplacement) + " " + isa::OpenCL::KernelConf::print();
}

template<typename DataType>
void RFIm::timeDomainSigmaCut(const bool subbandDedispersion, const DataOrdering & ordering, const ReplacementStrategy & replacement, const AstroData::Observation & observation, std::vector<DataType> & time_series, const float sigmaCut, const unsigned int padding)
{
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
                        if ( replacement == ReplaceWithMean )
                        {
                            time_series.at((beam * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + (channel * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding / sizeof(DataType))) + sample_id) = statistics.getMean();
                        }
                    }
                }
            }
        }
    }
}

template<typename DataType>
std::string * RFIm::getTimeDomainSigmaCutOpenCL(const RFIConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const float sigmaCut, const unsigned int padding)
{
    if ( (ordering == FrequencyTime) && (replacement == ReplaceWithMean) )
    {
        return getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean<DataType>(config, dataTypeName, observation, sigmaCut, padding);
    }
    return new std::string();
}

template<typename DataType>
std::string * RFIm::getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean(const RFIConfig & config, const std::string & dataTypeName, const AstroData::Observation & observation, const float sigmaCut, const unsigned int padding)
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
    std::string replaceTemplate = "sample_value = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>];"
    "time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>] = (sample_value * (convert_float(sample_value < (mean + sigma_cut)))) + (mean * (convert_float(sample_value > mean + sigma_cut)));";
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
void RFIm::testTimeDomainSigmaCut(const bool printResults, const RFIConfig & config, const DataOrdering & ordering, const ReplacementStrategy & replacement, const std::string & dataTypeName, const AstroData::Observation & observation, const std::vector<DataType> & time_series, isa::OpenCL::OpenCLRunTime & openCLRunTime, const unsigned int clDeviceID, const float sigmaCut, const unsigned int padding)
{
    std::uint64_t wrongSamples = 0;
    std::vector<DataType> test_time_series, control_time_series;
    test_time_series = time_series;
    control_time_series = time_series;
    cl::Buffer device_time_series;
    cl::Kernel * kernel = nullptr;
    // Execute control code
    timeDomainSigmaCut(config.getSubbandDedispersion(), ordering, replacement, observation, control_time_series, sigmaCut, padding);
    // Execute OpenCL code
    std::string * code = getTimeDomainSigmaCutOpenCL<DataType>(config, ordering, replacement, dataTypeName, observation, sigmaCut, padding);
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
    }
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
    }
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
    double bestBandwidth = 0.0;
    RFIConfig bestConfig;
    std::vector<RFIConfig> configurations;
    isa::OpenCL::OpenCLRunTime openCLRunTime;
    cl::Event clEvent;
    cl::Buffer device_time_series;
    // Generate valid configurations
    for ( unsigned int threads = parameters.getMinThreads(); threads < parameters.getMaxThreads(); threads *= 2 )
    {
        for ( unsigned int items = 1; (items * 3) + 8 < parameters.getMaxItems(); items++ )
        {
            if ( (threads * items) > observation.getNrSamplesPerDispersedBatch(subbandDedispersion) )
            {
                break;
            }
            RFIConfig baseConfig, tempConfig;
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
    std::cout << std::fixed << std::endl;
    if ( !parameters.getBestMode() )
    {
        std::cout << "# nrBeams nrChannels nrSamplesPerDispersedBatch *configuration* GB/s time stdDeviation COV" << std::endl << std::endl;
    }
    for ( auto configuration = configurations.begin(); configuration != configurations.end(); ++configuration )
    {
        // We know how many time we read the data (2), but not how many elements we write, so we only count the reads
        // Although this may mean that the computed GB/s metric is lower than reality, the ordering of configurations is not affected
        double bandwidth = isa::utils::giga(observation.getNrSynthesizedBeams() * static_cast<uint64_t>(observation.getNrChannels()) * observation.getNrSamplesPerDispersedBatch(subbandDedispersion) * sizeof(DataType) * 2.0);
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
        }
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
        if ( (bandwidth / timer.getAverageTime()) > bestBandwidth )
        {
            bestBandwidth = bandwidth;
            bestConfig = *configuration;
        }
        if ( !parameters.getBestMode() )
        {
            std::cout << observation.getNrSynthesizedBeams() << " " << observation.getNrChannels() << " ";
            std::cout  << observation.getNrSamplesPerDispersedBatch(subbandDedispersion) << " ";
            std::cout << (*configuration).print() << " ";
            std::cout << std::setprecision(3);
            std::cout << bandwidth / timer.getAverageTime() << " ";
            std::cout << std::setprecision(6);
            std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation();
            std::cout  << std::endl;
        }
    }
    if ( parameters.getBestMode() )
    {
        std::cout << observation.getNrChannels() << " " << observation.getNrSamplesPerDispersedBatch(subbandDedispersion) << " ";
        std::cout << bestConfig.print() << std::endl;
    }
    else
    {
        std::cout << std::endl;
    }
}