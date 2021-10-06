// SPDX-License-Identifier: MIT
// This file is part of `openmm-dlext`, see LICENSE.md

#include "internal/DLExtForceImpl.h"
#include "DLExtKernels.h"


using namespace DLExt;


ForceImpl::ForceImpl(const Force& owner)
  : owner(owner)
{ }

void ForceImpl::initialize(OpenMM::ContextImpl& context) { }

const OpenMM::Force& ForceImpl::getOwner() const
{
    return owner;
}

void ForceImpl::updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) { }

double ForceImpl::calcForcesAndEnergy(
    OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups
) {
    long long step_count;
    auto dtype = dlDeviceType(context.getPlatform());

    if (dtype == kDLCPU) {
        auto data = reinterpret_cast<ReferencePlatformData*>(context.getPlatformData());
        step_count = static_cast<long long>(data->stepCount);
    }
#ifdef OPENMM_BUILD_CUDA_LIB
    if (dtype == kDLGPU) {
        auto data = reinterpret_cast<CudaPlatformData*>(context.getPlatformData());
        step_count = static_cast<long long>(data->stepCount);
    }
#endif

    callback(step_count);
    return 0.0;
}

std::map<std::string, double> ForceImpl::getDefaultParameters()
{
    return std::map<std::string, double>();
}

std::vector<std::string> ForceImpl::getKernelNames()
{
    std::vector<std::string> names = { ForceKernel::Name() };
    return names;
}

void ForceImpl::setCallback(Function<void, long long>& f)
{
    callback = f;
}
