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

#include <configuration.hpp>

#include <RFIm.hpp>
#include <ArgumentList.hpp>

#include <random>
#include <cmath>

void usage(const std::string & name);

int main(int argc, char * argv[])
{
    bool subbandDedispersion = false;
    unsigned int padding = 0;
    unsigned int clPlatformID = 0;
    unsigned int clDeviceID = 0;
    unsigned int nrBins=0;
    float sigma = 0.0f;
    RFIm::RFImKernel kernelType;
    RFIm::DataOrdering dataOrdering;
    RFIm::ReplacementStrategy replacementStrategy;
    isa::OpenCL::TuningParameters parameters;
    AstroData::Observation observation;
    try
    {
        isa::utils::ArgumentList arguments(argc, argv);
        // Type of kernel to test
        if ( arguments.getSwitch("-time_domain_sigma_cut") )
        {
            kernelType = RFIm::RFImKernel::TimeDomainSigmaCut;
        }
        else if ( arguments.getSwitch("-frequency_domain_sigma_cut") )
        {
            kernelType = RFIm::RFImKernel::FrequencyDomainSigmaCut;
        }
        else
        {
            throw std::exception();
        }
        // General command line arguments
        clPlatformID = arguments.getSwitchArgument<unsigned int>("-opencl_platform");
        clDeviceID = arguments.getSwitchArgument<unsigned int>("-opencl_device");
        padding = arguments.getSwitchArgument<unsigned int>("-padding");
        observation.setNrBeams(arguments.getSwitchArgument<unsigned int>("-beams"));
        observation.setFrequencyRange(1, arguments.getSwitchArgument<unsigned int>("-channels"), 0.0f, 0.0f);
        subbandDedispersion = arguments.getSwitch("-subbanding");
        if ( subbandDedispersion )
        {
            observation.setNrSamplesPerDispersedBatch(arguments.getSwitchArgument<unsigned int>("-samples"), true);
        }
        else
        {
            observation.setNrSamplesPerDispersedBatch(arguments.getSwitchArgument<unsigned int>("-samples"));
        }
        // Tuning parameters
        parameters.setBestMode(arguments.getSwitch("-best"));
        parameters.setNrIterations(arguments.getSwitchArgument<unsigned int>("-iterations"));
        parameters.setMinThreads(arguments.getSwitchArgument<unsigned int>("-min_threads"));
        parameters.setMaxThreads(arguments.getSwitchArgument<unsigned int>("-max_threads"));
        parameters.setMaxItems(arguments.getSwitchArgument<unsigned int>("-max_items"));
        // Kernel specific command line arguments
        if ( kernelType == RFIm::RFImKernel::TimeDomainSigmaCut )
        {
            if ( arguments.getSwitch("-frequency_time") )
            {
                dataOrdering = RFIm::DataOrdering::FrequencyTime;
            }
            else
            {
                throw std::exception();
            }
            if ( arguments.getSwitch("-replace_mean") )
            {
                replacementStrategy = RFIm::ReplacementStrategy::ReplaceWithMean;
            }
            else
            {
                throw std::exception();
            }
            sigma = arguments.getSwitchArgument<float>("-sigma");
        }
        else if ( kernelType == RFIm::RFImKernel::FrequencyDomainSigmaCut )
        {
            nrBins = arguments.getSwitchArgument<unsigned int>("-nr_bins");
            if ( arguments.getSwitch("-frequency_time") )
            {
                dataOrdering = RFIm::DataOrdering::FrequencyTime;
            }
            else
            {
                throw std::exception();
            }
            if ( arguments.getSwitch("-replace_mean") )
            {
                replacementStrategy = RFIm::ReplacementStrategy::ReplaceWithMean;
            }
            else
            {
                throw std::exception();
            }
            sigma = arguments.getSwitchArgument<float>("-sigma");
        }
    }
    catch ( const isa::utils::SwitchNotFound & err )
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    catch ( const std::exception & err )
    {
        usage(argv[0]);
        return 1;
    }
    // Generate test data
    std::random_device randomDevice;
    std::mt19937 randomGenerator(randomDevice());
    std::normal_distribution<double> distribution(42, sigma);
    std::vector<InputDataType> time_series;
    if ( kernelType == RFIm::RFImKernel::TimeDomainSigmaCut || kernelType == RFIm::RFImKernel::FrequencyDomainSigmaCut )
    {
        time_series.resize(observation.getNrBeams() * observation.getNrChannels() * observation.getNrSamplesPerDispersedBatch(subbandDedispersion, padding));
    }
    for ( auto sample = time_series.begin(); sample != time_series.end(); ++sample )
    {
        *sample = static_cast<InputDataType>(distribution(randomGenerator));
    }
    // Tuning
    if ( kernelType == RFIm::RFImKernel::TimeDomainSigmaCut )
    {
        RFIm::tuneTimeDomainSigmaCut(subbandDedispersion, parameters, dataOrdering, replacementStrategy, inputDataName, observation, time_series, clPlatformID, clDeviceID, sigma, padding);
    }
    else if ( kernelType == RFIm::RFImKernel::FrequencyDomainSigmaCut )
    {
        RFIm::tuneFrequencyDomainSigmaCut(subbandDedispersion, parameters, dataOrdering, replacementStrategy, inputDataName, observation, time_series, clPlatformID, clDeviceID, nrBins, sigma, padding);
    }
    return 0;
}

void usage(const std::string & name)
{
    std::cerr << std::endl;
    std::cerr << name;
    std::cerr << " -opencl_platform <int> -opencl_device <int> -padding <int>";
    std::cerr << " [-subbanding] -beams <int> -channels <int> -samples <int>";
    std::cerr << " [-time_domain_sigma_cut] [-frequency_domain_sigma_cut]";
    std::cerr << " [-best] -iterations <int> -min_threads <int> -max_threads <int> -max_items <int>";
    std::cerr << std::endl;
    std::cerr << "\tTime Domain Sigma Cut: -frequency_time -replace_mean -sigma <float>";
    std::cerr << std::endl;
    std::cerr << "\tFrequency Domain Sigma Cut: -frequency_time -replace_mean -nr_bins <int> -sigma <float>";
    std::cerr << std::endl;
    std::cerr << std::endl;
}
