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
#include <Platform.hpp>

#include <fstream>

RFIm::RFImConfig::RFImConfig() : KernelConf(), subbandDedispersion(false), conditionalReplacement(false) {}

RFIm::RFImConfig::~RFImConfig() {}

void RFIm::readRFImConfig(RFIm::RFImConfigurations & configurations, const std::string & filename)
{
    unsigned int nrSamples = 0;
    float sigma = 0.0f;
    std::string deviceName;
    std::string temp;
    RFIm::RFImConfig * parameters = 0;
    std::ifstream file;

    file.open(filename);
    if ( !file )
    {
        throw AstroData::FileError("Impossible to open \"" + filename + "\".");
    }
    while ( !file.eof() )
    {
        unsigned int splitPoint = 0;

        std::getline(file, temp);
        if ( !std::isalpha(temp[0]) )
        {
            continue;
        }
        parameters = new RFImConfig();

        splitPoint = temp.find(" ");
        deviceName = temp.substr(0, splitPoint);
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        nrSamples = isa::utils::castToType<std::string, unsigned int>(temp.substr(0, splitPoint));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        sigma = isa::utils::castToType<std::string, float>(temp.substr(0, splitPoint));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        parameters->setSubbandDedispersion(isa::utils::castToType<std::string, bool>(temp.substr(0, splitPoint)));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        parameters->setConditionalReplacement(isa::utils::castToType<std::string, bool>(temp.substr(0, splitPoint)));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        parameters->setNrThreadsD0(isa::utils::castToType<std::string, unsigned int>(temp.substr(0, splitPoint)));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        parameters->setNrThreadsD1(isa::utils::castToType<std::string, unsigned int>(temp.substr(0, splitPoint)));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        parameters->setNrThreadsD2(isa::utils::castToType<std::string, unsigned int>(temp.substr(0, splitPoint)));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        parameters->setNrItemsD0(isa::utils::castToType<std::string, unsigned int>(temp.substr(0, splitPoint)));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        parameters->setNrItemsD1(isa::utils::castToType<std::string, unsigned int>(temp.substr(0, splitPoint)));
        temp = temp.substr(splitPoint + 1);
        splitPoint = temp.find(" ");
        parameters->setNrItemsD2(isa::utils::castToType<std::string, unsigned int>(temp.substr(0, splitPoint)));
        temp = temp.substr(splitPoint + 1);
        parameters->setIntType(isa::utils::castToType<std::string, unsigned int>(temp));

        if (configurations.count(deviceName) == 0)
        {
            std::map<unsigned int, std::map<float, RFIm::RFImConfig *> *> * externalContainer = new std::map<unsigned int, std::map<float, RFIm::RFImConfig *> *>();
            std::map<float, RFIm::RFImConfig *> * internalContainer = new std::map<float, RFIm::RFImConfig *>();

            internalContainer->insert(std::make_pair(sigma, parameters));
            externalContainer->insert(std::make_pair(nrSamples, internalContainer));
            configurations.insert(std::make_pair(deviceName, externalContainer));
        }
        else if (configurations.at(deviceName)->count(nrSamples) == 0)
        {
            std::map<float, RFIm::RFImConfig *> * internalContainer = new std::map<float, RFIm::RFImConfig *>();

            internalContainer->insert(std::make_pair(sigma, parameters));
            configurations.at(deviceName)->insert(std::make_pair(nrSamples, internalContainer));
        }
        else
        {
            configurations.at(deviceName)->at(nrSamples)->insert(std::make_pair(sigma, parameters));
        }
    }
    file.close();
}

void RFIm::readTimeDomainSigmaCutSteps(const std::string &inputFilename, std::vector<float> &steps)
{
    std::ifstream input;
    input.open(inputFilename);
    if ( !input )
    {
        throw AstroData::FileError("Impossible to open \"" + inputFilename + "\".");
    }
    while ( !input.eof() )
    {
        float step = 0.0f;
        input >> step;
        steps.push_back(step);
    }
    input.close();
}