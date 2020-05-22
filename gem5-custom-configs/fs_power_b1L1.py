# Copyright (c) 2017 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Authors: Andreas Sandberg
#          Stephan Diestelhorst

# This configuration file extends the example ARM big.LITTLE(tm)
# with example power models.

from __future__ import print_function
from __future__ import absolute_import

import argparse
import os

import m5
from m5.objects import MathExprPowerModel, PowerModel

import fs_bigLITTLE as bL


class BCpuA15PowerOn(MathExprPowerModel):
    dyn = "(((((system.bigCluster.cpus0.numCycles)/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 6.06992538845e-10) + " + \
 "(((((system.bigCluster.cpus0.dcache.overall_accesses::total)/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 2.32633723171e-10) + " + \
 "(((((system.bigCluster.cpus0.iew.iewExecutedInsts)/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000-(((system.bigCluster.cpus0.iq.FU_type_0::IntAlu+system.bigCluster.cpus0.iq.FU_type_0::IntMult+system.bigCluster.cpus0.iq.FU_type_0::IntDiv)/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 5.43933973638e-10) + " + \
 "(((((system.bigCluster.cpus0.dcache.WriteReq_misses::total)/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 4.79625288372e-08) + " + \
 "(((((system.bigCluster.l2.overall_accesses::bigCluster.cpus0.data)/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 5.72830963981e-09) + " + \
 "(((((system.bigCluster.cpus0.icache.ReadReq_accesses::total)/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 8.41332534886e-10) + " + \
 "(((((system.bigCluster.cpus0.iq.FU_type_0::IntAlu+system.bigCluster.cpus0.iq.FU_type_0::IntMult+system.bigCluster.cpus0.iq.FU_type_0::IntDiv)/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 2.44859350364e-10)"

    st = "((1) * -681.604059986) + " + \
 "(((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * 0.117551170367) + " + \
 "((system.voltage_domain.voltage) * 2277.16890778) + " + \
 "(((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage) * -0.491846201277) + " + \
 "((system.voltage_domain.voltage*system.voltage_domain.voltage) * -2528.1574686) + " + \
 "(((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 0.645456768269) + " + \
 "((system.voltage_domain.voltage) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 932.937276293) + " + \
 "(((1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * -0.271180478671)"


class LCpuA7PowerOn(MathExprPowerModel):
    dyn = "(((((system.littleCluster.cpus0.numCycles)/1)/sim_seconds)/(1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 1.84132144059e-10) + " + \
 "(((((system.littleCluster.cpus0.committedInsts)/1)/sim_seconds)/(1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 1.11008189839e-10) + " + \
 "(((((system.littleCluster.cpus0.dcache.overall_accesses::total)/1)/sim_seconds)/(1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 2.7252658216e-10) + " + \
 "(((((system.littleCluster.cpus0.dcache.overall_misses::total)/1)/sim_seconds)/(1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 2.4016441235e-09) + " + \
 "(((((system.mem_ctrls.bytes_read::littleCluster.cpus0.data)/1)/sim_seconds)/(1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * ((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * -2.44881613234e-09)"

    st = "((1) * 31.0366448991) + " + \
 "(((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * -0.0267126706228) + " + \
 "((system.voltage_domain.voltage) * -87.7978467067) + " + \
 "(((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage) * 0.0748426796784) + " + \
 "((system.voltage_domain.voltage*system.voltage_domain.voltage) * 82.5596011612) + " + \
 "(((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * -0.0696612748138) + " + \
 "((system.voltage_domain.voltage) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * -25.8616662356) + " + \
 "(((1/(system.littleCluster.clk_domain.clock/1000000000000))/1000000) * (system.voltage_domain.voltage) * (system.voltage_domain.voltage*system.voltage_domain.voltage) * 0.0216526889381)"


class A15PowerModel(PowerModel):
    pm = [
        BCpuA15PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]


class A7PowerModel(PowerModel):
    pm = [
        LCpuA7PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]



class CpuPowerOn(MathExprPowerModel):
    # 2A per IPC, 3pA per cache miss
    # and then convert to Watt
    dyn = "voltage * (2 * ipc + " \
            "3 * 0.000000001 * dcache.overall_misses / sim_seconds)"
    st = "4 * temp"

class CpuPowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"

class CpuPowerModel(PowerModel):
    pm = [
        CpuPowerOn(), # ON
        CpuPowerOff(), # CLK_GATED
        CpuPowerOff(), # SRAM_RETENTION
        CpuPowerOff(), # OFF
    ]

class L2PowerOn(MathExprPowerModel):
    # Example to report l2 Cache overall_accesses
    # The estimated power is converted to Watt and will vary based on the size of the cache
    dyn = "overall_accesses*0.000018000"
    st = "(voltage * 3)/10"

class L2PowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"

class L2PowerModel(PowerModel):
    # Choose a power model for every power state
    pm = [
        L2PowerOn(), # ON
        L2PowerOff(), # CLK_GATED
        L2PowerOff(), # SRAM_RETENTION
        L2PowerOff(), # OFF
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generic ARM big.LITTLE configuration with "\
        "example power models")
    bL.addOptions(parser)
    options = parser.parse_args()

    if options.cpu_type != "timing":
        m5.fatal("The power example script requires 'timing' CPUs.")

    root = bL.build(options)

    # Wire up some example power models to the CPUs
    for cpu in root.system.descendants():
        if not isinstance(cpu, m5.objects.BaseCPU):
            continue

        cpu.default_p_state = "ON"
        # cpu.power_model = CpuPowerModel()
        if cpu in root.system.bigCluster.cpus:
            cpu.power_model = A15PowerModel()
        elif cpu in root.system.littleCluster.cpus:
            cpu.power_model = A7PowerModel()

    # Example power model for the L2 Cache of the bigCluster
    for l2 in root.system.bigCluster.l2.descendants():
        if not isinstance(l2, m5.objects.Cache):
            continue

        l2.default_p_state = "ON"
        l2.power_model = L2PowerModel()

    bL.instantiate(options)

    print("*" * 70)
    print("WARNING: The power numbers generated by this script are "
        "examples. They are not representative of any particular "
        "implementation or process.")
    print("*" * 70)

    # Dumping stats periodically
    m5.stats.periodicStatDump(m5.ticks.fromSeconds(0.1E-3))
    bL.run()


if __name__ == "__m5_main__":
    main()
