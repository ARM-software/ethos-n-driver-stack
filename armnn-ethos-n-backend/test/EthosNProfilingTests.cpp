//
// Copyright © 2019-2020 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "EthosNBackend.hpp"
#include "EthosNBackendId.hpp"
#include "EthosNBackendProfilingContext.hpp"
#include "EthosNTestUtils.hpp"

#include <backendsCommon/test/CommonTestUtils.hpp>
#include <boost/test/unit_test.hpp>

using namespace armnn;
using namespace armnn::profiling;
using namespace testing_utils;

BOOST_AUTO_TEST_SUITE(EthosNProfiling)

BOOST_AUTO_TEST_CASE(TestProfilingRegisterCounters)
{
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));
    IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;

    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    auto& profilingService = GetProfilingService(static_cast<Runtime*>(runtime.get()));

    const armnn::profiling::ICounterMappings& counterMap = profilingService.GetCounterMappings();
    BOOST_CHECK(counterMap.GetGlobalId(
        static_cast<uint16_t>(ethosn::driver_library::profiling::PollCounterName::DriverLibraryNumLiveBuffers),
        EthosNBackendId()));
    BOOST_CHECK(counterMap.GetGlobalId(
        static_cast<uint16_t>(ethosn::driver_library::profiling::PollCounterName::DriverLibraryNumLiveInferences),
        EthosNBackendId()));
    BOOST_CHECK(counterMap.GetGlobalId(
        static_cast<uint16_t>(ethosn::driver_library::profiling::PollCounterName::KernelDriverNumMailboxMessagesSent),
        EthosNBackendId()));
    BOOST_CHECK(counterMap.GetGlobalId(
        static_cast<uint16_t>(
            ethosn::driver_library::profiling::PollCounterName::KernelDriverNumMailboxMessagesReceived),
        EthosNBackendId()));

    options.m_ProfilingOptions.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options.m_ProfilingOptions, true);
}

BOOST_AUTO_TEST_CASE(TestEnableProfiling)
{
    auto backendObjPtr = CreateBackendObject(EthosNBackendId());
    BOOST_TEST((backendObjPtr != nullptr));
    IRuntime::CreationOptions options;
    options.m_ProfilingOptions.m_EnableProfiling = true;

    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    auto& profilingService = GetProfilingService(static_cast<Runtime*>(runtime.get()));
    armnn::EthosNBackendProfilingService ethosnProfilingService = armnn::EthosNBackendProfilingService::Instance();
    armnn::profiling::EthosNBackendProfilingContext* ethosnProfilingContext = ethosnProfilingService.GetContext();

    // FIXME :- Check the return type when IBackendProfilingContext::EnableProfiling returns bool
    ethosnProfilingContext->EnableProfiling(true);

    ethosnProfilingContext->ActivateCounters(
        100,
        { (static_cast<uint16_t>(ethosn::driver_library::profiling::PollCounterName::DriverLibraryNumLiveBuffers)),
          (static_cast<uint16_t>(ethosn::driver_library::profiling::PollCounterName::DriverLibraryNumLiveInferences)),
          (static_cast<uint16_t>(
              ethosn::driver_library::profiling::PollCounterName::KernelDriverNumMailboxMessagesSent)),
          (static_cast<uint16_t>(
              ethosn::driver_library::profiling::PollCounterName::KernelDriverNumMailboxMessagesReceived)) });

    std::vector<Timestamp> timestamps = ethosnProfilingContext->ReportCounterValues();

    // This is because ethosnProfilingContext->ActivateCounters() is invoked for one capture period
    BOOST_TEST((timestamps.size() == 1));

    for (auto ts : timestamps)
    {
        // We know that the timestamp does not increment while running the test.
        // The timestamp is expected to increment when profiling is done on a running inference.
        BOOST_TEST(ts.timestamp == 100);

        // EthosNBackendProfilingContext::RegisterCounters registers 4 counters
        BOOST_TEST((ts.counterValues.size() == 4));

        for (auto cv : ts.counterValues)
        {
            // We do not test the counter values as they are always zero while running the tests.
            // The counter values are expected to be greater than zero when profiling is done on a
            // running inference.
            BOOST_TEST(
                ((cv.counterId ==
                  (static_cast<uint16_t>(
                      ethosn::driver_library::profiling::PollCounterName::DriverLibraryNumLiveBuffers))) ||
                 (cv.counterId ==
                  (static_cast<uint16_t>(
                      ethosn::driver_library::profiling::PollCounterName::DriverLibraryNumLiveInferences))) ||
                 (cv.counterId ==
                  (static_cast<uint16_t>(
                      ethosn::driver_library::profiling::PollCounterName::KernelDriverNumMailboxMessagesSent))) ||
                 (cv.counterId ==
                  (static_cast<uint16_t>(
                      ethosn::driver_library::profiling::PollCounterName::KernelDriverNumMailboxMessagesReceived)))));
        }
    }
    options.m_ProfilingOptions.m_EnableProfiling = false;
    profilingService.ResetExternalProfilingOptions(options.m_ProfilingOptions, true);
}

BOOST_AUTO_TEST_SUITE_END()
