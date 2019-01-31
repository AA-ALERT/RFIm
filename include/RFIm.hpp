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

#include <string>

#pragma once

namespace RFIm
{

/**
 ** @brief RFI specific configuration.
 */
class rfiConfig : public isa::OpenCL::KernelConf
{
};

/**
 ** @brief Ordering of input data.
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
 ** @brief Generates the OpenCL code for the time domain sigma cut.
 ** @param rfiConfig The kernel configuration.
 ** @param ordering The ordering of the input.
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
 ** @param rfiConfig The kernel configuration.
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
std::string * RFIm::getTimeDomainSigmaCutOpenCL(const rfiConfig &config, const DataOrdering &ordering, const ReplacementStrategy &replacement, const std::string &dataTypeName, const AstroData::Observation &observation, const float sigmaCut, const unsigned int padding)
{
    if ( (ordering == DataOrdering::FrequencyTime) && (replacement == ReplacementStrategy::ReplaceWithMean) )
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
    // End of kernel template
    // Code generation
    return code;
}
