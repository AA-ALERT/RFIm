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

#include <Kernel.hpp>
#include <Observation.hpp>
#include <Statistics.hpp>

#include <string>
#include <vector>

#pragma once

namespace RFIm
{

/**
 ** @brief RFI specific configuration.
 */
class rfiConfig : public isa::OpenCL::KernelConf
{
public:
    rfiConfig();
    ~rfiConfig();
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
    void setSubbandDedispersion(bool subband);
    /**
     ** @brief Set the conditional replacement mode.
     */
    void setConditionalReplacement(bool replacement);
    /**
     ** @brief Print the configuration.
     */
    std::string print() const;

private:
    bool subbandDedispersion;
    bool conditionalReplacement;
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
 ** @param ordering The ordering of the data.
 ** @param replacement The replacement strategy for flagged samples.
 ** @param observation The observation object.
 ** @param time_series The input data.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 */
template<typename DataType>
void timeDomainSigmaCut(const bool subbandDedispersion, const DataOrdering &ordering, const ReplacementStrategy &replacement, const AstroData::Observation &observation, std::vector<DataType> &time_series, const float sigmaCut, const unsigned int padding);

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
 */
template<typename DataType>
std::string * getTimeDomainSigmaCutOpenCL(const rfiConfig &config, const DataOrdering &ordering, const ReplacementStrategy &replacement, const std::string &dataTypeName, const AstroData::Observation &observation, const float sigmaCut, const unsigned int padding);

/**
 ** @brief Generates the OpenCL code for the time domain sigma cut.
 ** This function generates specialized code for the case in which the input is FrequencyTime ordered and flagged samples are replaced with the mean.
 **
 ** @param config The kernel configuration.
 ** @param dataTypeName The name of the input data type.
 ** @param observation The observation object.
 ** @param sigmaCut The threshold value for the sigma cut.
 ** @param padding The padding, in bytes, necessary to align data to cache lines.
 */
template<typename DataType>
std::string * getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean(const rfiConfig &config, const std::string &dataTypeName, const AstroData::Observation &observation, const float sigmaCut, const unsigned int padding);

} // RFIm

// Implementations

template<typename DataType>
void RFIm::timeDomainSigmaCut(const bool subbandDedispersion, const DataOrdering &ordering, const ReplacementStrategy &replacement, const AstroData::Observation &observation, std::vector<DataType> &time_series, const float sigmaCut, const unsigned int padding)
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
std::string * RFIm::getTimeDomainSigmaCutOpenCL(const rfiConfig &config, const DataOrdering &ordering, const ReplacementStrategy &replacement, const std::string &dataTypeName, const AstroData::Observation &observation, const float sigmaCut, const unsigned int padding)
{
    if ( (ordering == FrequencyTime) && (replacement == ReplaceWithMean) )
    {
        return getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean<DataType>(config, dataTypeName, observation, sigmaCut, padding);
    }
    return new std::string();
}

template<typename DataType>
std::string * RFIm::getTimeDomainSigmaCutOpenCL_FrequencyTime_ReplaceWithMean(const rfiConfig &config, const std::string &dataTypeName, const AstroData::Observation &observation, const float sigmaCut, const unsigned int padding)
{
    std::string *code = new std::string();
    // Kernel template
    *code = "__kernel void timeDomainSigmaCut(__global const " + dataTypeName + " * const restrict time_series) {\n"
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
    "delta = value - mean_<%ITEM_NUMBER%>;\n"
    "mean_<%ITEM_NUMBER%> += delta / counter_<%ITEM_NUMBER%>;\n"
    "variance_<%ITEM_NUMBER%> += delta * (value - mean_<%ITEM_NUMBER%>);\n";
    // Version with boundary checks
    std::string localComputeCheckTemplate = "if ( sample_id + <%ITEM_OFFSET%> < " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion())) + " ) {\n"
    "value = time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>];\n"
    "counter_<%ITEM_NUMBER%> += 1.0f;\n"
    "delta = value - mean_<%ITEM_NUMBER%>;\n"
    "mean_<%ITEM_NUMBER%> += delta / counter_<%ITEM_NUMBER%>;\n"
    "variance_<%ITEM_NUMBER%> += delta * (value - mean_<%ITEM_NUMBER%>);\n"
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
    "time_series[(get_global_id(2) * " + std::to_string(observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + (get_global_id(1) * " + std::to_string(observation.getNrSamplesPerDispersedBatch(config.getSubbandDedispersion(), padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>] = (sample_value * (convert_float(sample_value < (mean + sigma_cut)))) + (mean * (convert_float(sample > mean + sigma_cut)));";
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
        delete temp;
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
