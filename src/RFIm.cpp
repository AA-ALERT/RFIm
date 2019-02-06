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

#include <RFIm.hpp>

namespace RFIm
{
RFIConfig::RFIConfig() : KernelConf(), subbandDedispersion(false), conditionalReplacement(false) {}

RFIConfig::~RFIConfig() {}

inline bool RFIConfig::getSubbandDedispersion() const
{
    return subbandDedispersion;
}

inline bool RFIConfig::getConditionalReplacement() const
{
    return conditionalReplacement;
}

inline void RFIConfig::setSubbandDedispersion(const bool subband)
{
    subbandDedispersion = subband;
}

inline void RFIConfig::setConditionalReplacement(const bool replacement)
{
    conditionalReplacement = replacement;
}

inline std::string RFIConfig::print() const
{
    return std::to_string(subbandDedispersion) + " " + std::to_string(conditionalReplacement) + " " + isa::OpenCL::KernelConf::print();
}

} // RFIm
