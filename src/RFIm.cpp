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
rfiConfig::rfiConfig() : KernelConf(), subbandDedispersion(false), conditionalReplacement(false) {}

rfiConfig::~rfiConfig() {}

bool rfiConfig::getSubbandDedispersion() const
{
    return subbandDedispersion;
}

bool rfiConfig::getConditionalReplacement() const
{
    return conditionalReplacement;
}

void rfiConfig::setSubbandDedispersion(bool subband)
{
    subbandDedispersion = subband;
}

void rfiConfig::setConditionalReplacement(bool replacement)
{
    conditionalReplacement = replacement;
}

std::string rfiConfig::print() const
{
    return std::to_string(subbandDedispersion) + " " + std::to_string(conditionalReplacement) + " " + isa::OpenCL::KernelConf::print();
}

} // RFIm
