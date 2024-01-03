//
// Copyright © 2022 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

/// Returns the stripe id of the agent down the sequence that first needs
/// stripe x of the current agent based on the dependency info.
function GetFirstReaderStripeId(dep, x)
{
    if (x == 0)
    {
        return 0;
    }

    var outer = dep.outerRatio.other * Math.floor(x / dep.outerRatio.self);

    var inner =     (x % dep.outerRatio.self) - dep.boundary;
    inner          = dep.innerRatio.other * Math.floor(inner / dep.innerRatio.self);
    inner          = Math.min(Math.max(inner, 0), dep.outerRatio.other - 1);

    return outer + inner;
}

/// Returns the largest stripe id of the producer agent up the sequence that needs
/// to be completed before stripe x of the current agent can start.
function GetLargestNeededStripeId(dep, x)
{
    var outer = dep.outerRatio.other * Math.floor(x / dep.outerRatio.self);

    var inner = x % dep.outerRatio.self;
    inner          = dep.innerRatio.other * Math.floor(inner / dep.innerRatio.self);
    inner          = inner + dep.innerRatio.other - 1 + dep.boundary;
    inner          = Math.min(Math.max(inner, 0), dep.outerRatio.other - 1);

    return outer + inner;
}

/// Returns the stripe id of the agent down the sequence that last uses
/// stripe x of the current agent.
function GetLastReaderStripeId(dep, x)
{
    var outer = dep.outerRatio.other * Math.floor(x / dep.outerRatio.self);

    var inner = (x % dep.outerRatio.self) + dep.boundary;
    inner          = dep.innerRatio.other * Math.floor(inner / dep.innerRatio.self);
    inner          = inner + dep.innerRatio.other - 1;
    inner          = Math.min(Math.max(inner, 0), dep.outerRatio.other - 1);

    return outer + inner;
}

/// Returns the stripe id of the agent down the sequence that last uses
/// stripe (x - tileSize) of the current agent.
function GetLastReaderOfEvictedStripeId(dep, x, tileSize)
{
    console.assert(x >= tileSize);
    return GetLastReaderStripeId(dep, x - tileSize);
}
