# Copyright (c) 2016-2017, 2019 ARM Limited
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
# Authors: Thomas E. Hansen

# This configuration file extends the extended ARM big.LITTLE(tm) fs file
# by adding power models generated through "gemstone-applypower". It builds
# on the provided `fs_power.py` file.

from __future__ import print_function
from __future__ import absolute_import

import m5
from m5.objects import MathExprPowerModel, PowerModel


# simple example power model, taken from `fs_power.py`
class ExampleCpuPowerOn(MathExprPowerModel):
    dyn = "voltage * (2 * ipc + " \
            "3 * 0.000000001 * dcache.overall_misses / sim_seconds)"
    st = "4 * temp"


class BigCpuA15x1PowerOn(MathExprPowerModel):
#    dyn = "(((((system.bigCluster.cpus0.numCycles)/1)/sim_seconds)/" \
#          + "(1/(system.bigCluster.clk_domain.clock/1000000000000))" \
#          + "/1000000) * ((1/(system.bigCluster.clk_domain.clock" \
#          + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
#          + "*system.voltage_domain.voltage) * 6.06992538845e-10) + " \
#          + "(((((system.bigCluster.cpus0.dcache.overall_accesses::total)" \
#          + "/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
#          + "/1000000000000))/1000000) * ((1/(" \
#          + "system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
#          + " * (system.voltage_domain.voltage" \
#          + "*system.voltage_domain.voltage) * 2.32633723171e-10) + " \
#          + "(((((system.bigCluster.cpus0.iew.iewExecutedInsts)/1)" \
#          + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
#          + "/1000000000000))/1000000-(((" \
#          + "system.bigCluster.cpus0.iq.FU_type_0::IntAlu" \
#          + "+system.bigCluster.cpus0.iq.FU_type_0::IntMult" \
#          + "+system.bigCluster.cpus0.iq.FU_type_0::IntDiv)/1)/sim_seconds)" \
#          + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))" \
#          + "/1000000) * ((1/(system.bigCluster.clk_domain.clock" \
#          + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
#          + "*system.voltage_domain.voltage) * 5.43933973638e-10) + " \
#          + "(((((system.bigCluster.cpus0.dcache.WriteReq_misses::total)/1)" \
#          + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
#          + "/1000000000000))/1000000) * ((1/(" \
#          + "system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
#          + " * (system.voltage_domain.voltage" \
#          + "*system.voltage_domain.voltage) * 4.79625288372e-08) + " \
#          + "(((((" \
#          + "system.bigCluster.l2.overall_accesses::bigCluster.cpus0.data)" \
#          + "/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
#          + "/1000000000000))/1000000) * ((1/(" \
#          + "system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
#          + " * (system.voltage_domain.voltage*" \
#          + "system.voltage_domain.voltage) * 5.72830963981e-09) + " \
#          + "(((((system.bigCluster.cpus0.icache.ReadReq_accesses::total)" \
#          + "/1)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
#          + "/1000000000000))/1000000) * ((1/(" \
#          + "system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
#          + " * (system.voltage_domain.voltage" \
#          + "*system.voltage_domain.voltage) * 8.41332534886e-10) + " \
#          + "(((((system.bigCluster.cpus0.iq.FU_type_0::IntAlu+" \
#          + "system.bigCluster.cpus0.iq.FU_type_0::IntMult+" \
#          + "system.bigCluster.cpus0.iq.FU_type_0::IntDiv)/1)/sim_seconds)" \
#          + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))" \
#          + "/1000000) * ((1/(system.bigCluster.clk_domain.clock" \
#          + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
#          + "*system.voltage_domain.voltage) * 2.44859350364e-10)"
#
#    st = "((1) * -681.604059986) + " \
#       + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
#       + "/1000000) * 0.117551170367) + " \
#       + "((system.voltage_domain.voltage) * 2277.16890778) + " \
#       + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
#       + "/1000000) * (system.voltage_domain.voltage) * -0.491846201277)" \
#       + " + " \
#       + "((system.voltage_domain.voltage*system.voltage_domain.voltage)" \
#       + " * -2528.1574686) + " \
#       + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
#       + "/1000000) * (system.voltage_domain.voltage" \
#       + "*system.voltage_domain.voltage) * 0.645456768269) + " \
#       + "((system.voltage_domain.voltage) * (" \
#       + "system.voltage_domain.voltage*system.voltage_domain.voltage)" \
#       + " * 932.937276293) + " \
#       + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
#       + "/1000000) * (system.voltage_domain.voltage)" \
#       + " * (system.voltage_domain.voltage" \
#       + "*system.voltage_domain.voltage) * -0.271180478671)"

    dyn = "(((((numCycles)/1)/sim_seconds)/(1/(clock_period/1000000000000))" \
        + "/1000000) * ((1/(clock_period/1000000000000))/1000000) * (voltage" \
        + "*voltage) * 6.06992538845e-10) + " \
        + "(((((cpus.dcache.overall_accesses::total)/1)/sim_seconds)" \
        + "/(1/(clock_period/1000000000000))/1000000) * ((1/(clock_period/1000000000000))" \
        + "/1000000) * (voltage*voltage) * 2.32633723171e-10) + " \
        + "(((((iew.iewExecutedInsts)/1)/sim_seconds)/(1/(clock_period" \
        + "/1000000000000))/1000000-(((cpus.iq.FU_type_0::IntAlu" \
        + "+cpus.iq.FU_type_0::IntMult+cpus.iq.FU_type_0::IntDiv)/1)" \
        + "/sim_seconds)/(1/(clock_period/1000000000000))/1000000) * ((1/(clock_period" \
        + "/1000000000000))/1000000) * (voltage*voltage)" \
        + " * 5.43933973638e-10) + " \
        + "(((((cpus.dcache.WriteReq_misses::total)/1)/sim_seconds)" \
        + "/(1/(clock_period/1000000000000))/1000000) * ((1/(clock_period/1000000000000))" \
        + "/1000000) * (voltage*voltage) * 4.79625288372e-08) + " \
        + "(((((system.bigCluster.l2.overall_accesses::bigCluster.cpus.data)" \
        + "/1)/sim_seconds)/(1/(clock_period/1000000000000))/1000000) * ((1" \
        + "/(clock_period/1000000000000))/1000000) * (voltage*voltage)" \
        + " * 5.72830963981e-09) + " \
        + "(((((cpus.icache.ReadReq_accesses::total)/1)/sim_seconds)/" \
        + "(1/(clock_period/1000000000000))/1000000) * ((1/(clock_period/1000000000000))" \
        + "/1000000) * (voltage*voltage) * 8.41332534886e-10) + " \
        + "(((((cpus.iq.FU_type_0::IntAlu+cpus.iq.FU_type_0::IntMult" \
        + "+cpus.iq.FU_type_0::IntDiv)/1)/sim_seconds)/(1/(clock_period" \
        + "/1000000000000))/1000000) * ((1/(clock_period/1000000000000))/1000000)" \
        + " * (voltage*voltage) * 2.44859350364e-10)"

    st = "((1) * -681.604059986) + " \
       + "(((1/(clock_period/1000000000000))/1000000) * 0.117551170367) + " \
       + "((voltage) * 2277.16890778) + " \
       + "(((1/(clock_period/1000000000000))/1000000) * (voltage)" \
       + " * -0.491846201277) + " \
       + "((voltage*voltage) * -2528.1574686) + " \
       + "(((1/(clock_period/1000000000000))/1000000) * (voltage*voltage)" \
       + " * 0.645456768269) + " \
       + "((voltage) * (voltage*voltage) * 932.937276293) + " \
       + "(((1/(clock_period/1000000000000))/1000000) * (voltage)" \
       + " * (voltage*voltage) * -0.271180478671)"


class BigCpuA15x2PowerOn(MathExprPowerModel):
    dyn = "(((((system.bigCluster.cpus0.numCycles +" \
        + "system.bigCluster.cpus1.numCycles)/2)/sim_seconds)/(1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 6.06992538845e-10) + " \
        + "(((((system.bigCluster.cpus0.dcache.overall_accesses::total" \
        + " + system.bigCluster.cpus1.dcache.overall_accesses::total)/2)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1/(" \
        + "system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 2.32633723171e-10) + " \
        + "(((((system.bigCluster.cpus0.iew.iewExecutedInsts" \
        + "+system.bigCluster.cpus1.iew.iewExecutedInsts)/2)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000" \
        + "-(((system.bigCluster.cpus0.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus1.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntDiv)/2)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 5.43933973638e-10) + " \
        + "(((((system.bigCluster.cpus0.dcache.WriteReq_misses::total" \
        + " + system.bigCluster.cpus1.dcache.WriteReq_misses::total)/2)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1/(" \
        + "system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 4.79625288372e-08) + " \
        + "(((((system.bigCluster.l2.overall_accesses::bigCluster.cpus0.data" \
        + " + system.bigCluster.l2.overall_accesses::bigCluster.cpus1.data)" \
        + "/2)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1/(" \
        + "system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 5.72830963981e-09) + " \
        + "(((((system.bigCluster.cpus0.icache.ReadReq_accesses::total" \
        + " + system.bigCluster.cpus1.icache.ReadReq_accesses::total)/2)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1/(" \
        + "system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 8.41332534886e-10) + " \
        + "(((((system.bigCluster.cpus0.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus1.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntDiv)/2)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 2.44859350364e-10)"

    st = "((1) * -681.604059986) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * 0.117551170367) + " \
         + "((system.voltage_domain.voltage) * 2277.16890778) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage) * -0.491846201277)" \
         + " + " \
         + "((system.voltage_domain.voltage*system.voltage_domain.voltage)" \
         + " * -2528.1574686) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage" \
         + "*system.voltage_domain.voltage) * 0.645456768269) + " \
         + "((system.voltage_domain.voltage) * (" \
         + "system.voltage_domain.voltage*system.voltage_domain.voltage)" \
         + " * 932.937276293) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage)" \
         + " * (system.voltage_domain.voltage" \
         + "*system.voltage_domain.voltage) * -0.271180478671)"



class BigCpuA15x3PowerOn(MathExprPowerModel):
    dyn = "(((((system.bigCluster.cpus0.numCycles" \
        + " + system.bigCluster.cpus1.numCycles" \
        + " + system.bigCluster.cpus2.numCycles)/3)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 6.06992538845e-10) + " \
        + "(((((system.bigCluster.cpus0.dcache.overall_accesses::total" \
        + " + system.bigCluster.cpus1.dcache.overall_accesses::total" \
        + " + system.bigCluster.cpus2.dcache.overall_accesses::total)/3)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 2.32633723171e-10) + " \
        + "(((((system.bigCluster.cpus0.iew.iewExecutedInsts" \
        + "+system.bigCluster.cpus1.iew.iewExecutedInsts" \
        + "+system.bigCluster.cpus2.iew.iewExecutedInsts)/3)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000" \
        + "-(((system.bigCluster.cpus0.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus1.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus2.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus2.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus2.iq.FU_type_0::IntDiv)/3)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 5.43933973638e-10) + " \
        + "(((((system.bigCluster.cpus0.dcache.WriteReq_misses::total" \
        + " + system.bigCluster.cpus1.dcache.WriteReq_misses::total " \
        + " + system.bigCluster.cpus2.dcache.WriteReq_misses::total)/3)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 4.79625288372e-08) + " \
        + "(((((system.bigCluster.l2.overall_accesses::bigCluster.cpus0.data" \
        + " + system.bigCluster.l2.overall_accesses::bigCluster.cpus1.data" \
        + " + system.bigCluster.l2.overall_accesses::bigCluster.cpus2.data)" \
        + "/3)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 5.72830963981e-09) + "  \
        + "(((((system.bigCluster.cpus0.icache.ReadReq_accesses::total" \
        + " + system.bigCluster.cpus1.icache.ReadReq_accesses::total" \
        + " + system.bigCluster.cpus2.icache.ReadReq_accesses::total)/3)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 8.41332534886e-10) + " \
        + "(((((system.bigCluster.cpus0.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus1.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus2.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus2.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus2.iq.FU_type_0::IntDiv)/3)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 2.44859350364e-10)"

    st = "((1) * -681.604059986) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * 0.117551170367) + " \
         + "((system.voltage_domain.voltage) * 2277.16890778) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage) * -0.491846201277)" \
         + " + " \
         + "((system.voltage_domain.voltage*system.voltage_domain.voltage)" \
         + " * -2528.1574686) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage" \
         + "*system.voltage_domain.voltage) * 0.645456768269) + " \
         + "((system.voltage_domain.voltage) * (" \
         + "system.voltage_domain.voltage*system.voltage_domain.voltage)" \
         + " * 932.937276293) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage)" \
         + " * (system.voltage_domain.voltage" \
         + "*system.voltage_domain.voltage) * -0.271180478671)"


class BigCpuA15x4PowerOn(MathExprPowerModel):
    dyn = "(((((system.bigCluster.cpus0.numCycles" \
        + " + system.bigCluster.cpus1.numCycles" \
        + " + system.bigCluster.cpus2.numCycles" \
        + " + system.bigCluster.cpus3.numCycles)/4)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 6.06992538845e-10) + " \
        + "(((((system.bigCluster.cpus0.dcache.overall_accesses::total" \
        + " + system.bigCluster.cpus1.dcache.overall_accesses::total" \
        + " + system.bigCluster.cpus2.dcache.overall_accesses::total" \
        + " + system.bigCluster.cpus3.dcache.overall_accesses::total)/4)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 2.32633723171e-10) + " \
        + "(((((system.bigCluster.cpus0.iew.iewExecutedInsts" \
        + "+system.bigCluster.cpus1.iew.iewExecutedInsts" \
        + "+system.bigCluster.cpus2.iew.iewExecutedInsts" \
        + "+system.bigCluster.cpus3.iew.iewExecutedInsts)/4)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000" \
        + "-(((system.bigCluster.cpus0.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus1.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus2.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus2.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus2.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus3.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus3.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus3.iq.FU_type_0::IntDiv)/4)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 5.43933973638e-10) + " \
        + "(((((system.bigCluster.cpus0.dcache.WriteReq_misses::total" \
        + " + system.bigCluster.cpus1.dcache.WriteReq_misses::total" \
        + " + system.bigCluster.cpus2.dcache.WriteReq_misses::total" \
        + " + system.bigCluster.cpus3.dcache.WriteReq_misses::total)/4)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 4.79625288372e-08) + " \
        + "(((((system.bigCluster.l2.overall_accesses::bigCluster.cpus0.data" \
        + " + system.bigCluster.l2.overall_accesses::bigCluster.cpus1.data" \
        + " + system.bigCluster.l2.overall_accesses::bigCluster.cpus2.data" \
        + " + system.bigCluster.l2.overall_accesses::bigCluster.cpus3.data)" \
        + "/4)/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 5.72830963981e-09) + " \
        + "(((((system.bigCluster.cpus0.icache.ReadReq_accesses::total" \
        + " + system.bigCluster.cpus1.icache.ReadReq_accesses::total" \
        + " + system.bigCluster.cpus2.icache.ReadReq_accesses::total" \
        + " + system.bigCluster.cpus3.icache.ReadReq_accesses::total)/4)" \
        + "/sim_seconds)/(1/(system.bigCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 8.41332534886e-10) + " \
        + "(((((system.bigCluster.cpus0.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus0.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus1.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus1.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus2.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus2.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus2.iq.FU_type_0::IntDiv" \
        + " + system.bigCluster.cpus3.iq.FU_type_0::IntAlu" \
        + "+system.bigCluster.cpus3.iq.FU_type_0::IntMult" \
        + "+system.bigCluster.cpus3.iq.FU_type_0::IntDiv)/4)/sim_seconds)" \
        + "/(1/(system.bigCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * ((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 2.44859350364e-10)"

    st = "((1) * -681.604059986) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * 0.117551170367) + " \
         + "((system.voltage_domain.voltage) * 2277.16890778) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage) * -0.491846201277)" \
         + " + " \
         + "((system.voltage_domain.voltage*system.voltage_domain.voltage)" \
         + " * -2528.1574686) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage" \
         + "*system.voltage_domain.voltage) * 0.645456768269) + " \
         + "((system.voltage_domain.voltage) * (" \
         + "system.voltage_domain.voltage*system.voltage_domain.voltage)" \
         + " * 932.937276293) + " \
         + "(((1/(system.bigCluster.clk_domain.clock/1000000000000))" \
         + "/1000000) * (system.voltage_domain.voltage)" \
         + " * (system.voltage_domain.voltage" \
         + "*system.voltage_domain.voltage) * -0.271180478671)"


class LittleCpuA7x1PowerOn(MathExprPowerModel):
#    dyn = "(((((system.littleCluster.cpus.numCycles)/1)/sim_seconds)" \
#        + "/(1/(system.littleCluster.clk_domain.clock/1000000000000))" \
#        + "/1000000) * ((1/(system.littleCluster.clk_domain.clock" \
#        + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
#        + "*system.voltage_domain.voltage) * 1.84132144059e-10) + " \
#        + "(((((system.littleCluster.cpus.committedInsts)/1)/sim_seconds)" \
#        + "/(1/(system.littleCluster.clk_domain.clock/1000000000000))" \
#        + "/1000000) * ((1/(system.littleCluster.clk_domain.clock" \
#        + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
#        + "*system.voltage_domain.voltage) * 1.11008189839e-10) + " \
#        + "(((((system.littleCluster.cpus.dcache.overall_accesses::total)" \
#        + "/1)/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
#        + "/1000000000000))/1000000) * ((1/" \
#        + "(system.littleCluster.clk_domain.clock/1000000000000))" \
#        + "/1000000) * (system.voltage_domain.voltage" \
#        + "*system.voltage_domain.voltage) * 2.7252658216e-10) + " \
#        + "(((((system.littleCluster.cpus.dcache.overall_misses::total)" \
#        + "/1)/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
#        + "/1000000000000))/1000000) * ((1" \
#        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
#        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
#        + " * 2.4016441235e-09) + " \
#        + "(((((system.mem_ctrls.bytes_read::littleCluster.cpus.data)/1)" \
#        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
#        + "/1000000000000))/1000000) * ((1" \
#        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
#        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
#        + " * -2.44881613234e-09)"
#
#    st = "((1) * 31.0366448991) + " \
#       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
#       + "/1000000) * -0.0267126706228) + " \
#       + "((system.voltage_domain.voltage) * -87.7978467067) + " \
#       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
#       + "/1000000) * (system.voltage_domain.voltage) * 0.0748426796784) + " \
#       + "((system.voltage_domain.voltage*system.voltage_domain.voltage)" \
#       + " * 82.5596011612) + " \
#       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
#       + "/1000000) * (system.voltage_domain.voltage" \
#       + "*system.voltage_domain.voltage) * -0.0696612748138) + " \
#       + "((system.voltage_domain.voltage) * (system.voltage_domain.voltage" \
#       + "*system.voltage_domain.voltage) * -25.8616662356) + " \
#       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
#       + "/1000000) * (system.voltage_domain.voltage)" \
#       + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
#       + " * 0.0216526889381)"
    dyn = "(((((numCycles)/1)/sim_seconds)/(1/(clock_period/1000000000000))" \
        + "/1000000) * ((1/(clock_period/1000000000000))/1000000) * (voltage" \
        + "*voltage) * 1.84132144059e-10) + " \
        + "(((((committedInsts)/1)/sim_seconds)/(1/(clock_period" \
        + "/1000000000000))/1000000) * ((1/(clock_period/1000000000000))/1000000)" \
        + " * (voltage*voltage) * 1.11008189839e-10) + " \
        + "(((((cpus.dcache.overall_accesses::total)/1)/sim_seconds)" \
        + "/(1/(clock_period/1000000000000))/1000000) * ((1/(clock_period/1000000000000))" \
        + "/1000000) * (voltage*voltage) * 2.7252658216e-10) + " \
        + "(((((cpus.dcache.overall_misses::total)/1)/sim_seconds)/(1" \
        + "/(clock_period/1000000000000))/1000000) * ((1/(clock_period/1000000000000))" \
        + "/1000000) * (voltage*voltage) * 2.4016441235e-09) + " \
        + "(((((mem_ctrls.bytes_read::littleCluster.cpus.data)/1)" \
        + "/sim_seconds)/(1/(clock_period/1000000000000))/1000000) * ((1/(clock_period" \
        + "/1000000000000))/1000000) * (voltage*voltage) * -2.44881613234e-09)"

    st = "((1) * 31.0366448991) + " \
       + "(((1/(clock_period/1000000000000))/1000000) * -0.0267126706228) + " \
       + "((voltage) * -87.7978467067) + " \
       + "(((1/(clock_period/1000000000000))/1000000) * (voltage)" \
       + " * 0.0748426796784) + " \
       + "((voltage*voltage) * 82.5596011612) + " \
       + "(((1/(clock_period/1000000000000))/1000000) * (voltage*voltage)" \
       + " * -0.0696612748138) + " \
       + "((voltage) * (voltage*voltage) * -25.8616662356) + " \
       + "(((1/(clock_period/1000000000000))/1000000) * (voltage)" \
       + " * (voltage*voltage) * 0.0216526889381)"


class LittleCpuA7x2PowerOn(MathExprPowerModel):
    dyn = "(((((system.littleCluster.cpus0.numCycles" \
        + " + system.littleCluster.cpus1.numCycles)/2)/sim_seconds)" \
        + "/(1/(system.littleCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * ((1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 1.84132144059e-10) + " \
        + "(((((system.littleCluster.cpus0.committedInsts" \
        + " + system.littleCluster.cpus1.committedInsts)/2)/sim_seconds)" \
        + "/(1/(system.littleCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * ((1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 1.11008189839e-10) + " \
        + "(((((system.littleCluster.cpus0.dcache.overall_accesses::total" \
        + " + system.littleCluster.cpus1.dcache.overall_accesses::total)/2)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 2.7252658216e-10) + " \
        + "(((((system.littleCluster.cpus0.dcache.overall_misses::total" \
        + " + system.littleCluster.cpus1.dcache.overall_misses::total)/2)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 2.4016441235e-09) + " \
        + "(((((system.mem_ctrls.bytes_read::littleCluster.cpus0.data" \
        + " + system.mem_ctrls.bytes_read::littleCluster.cpus1.data)/2)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * -2.44881613234e-09)"

    st = "((1) * 31.0366448991) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * -0.0267126706228) + " \
       + "((system.voltage_domain.voltage) * -87.7978467067) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage) * 0.0748426796784) + " \
       + "((system.voltage_domain.voltage*system.voltage_domain.voltage)" \
       + " * 82.5596011612) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage" \
       + "*system.voltage_domain.voltage) * -0.0696612748138) + " \
       + "((system.voltage_domain.voltage) * (system.voltage_domain.voltage" \
       + "*system.voltage_domain.voltage) * -25.8616662356) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage)" \
       + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
       + " * 0.0216526889381)"


class LittleCpuA7x3PowerOn(MathExprPowerModel):
    dyn = "(((((system.littleCluster.cpus0.numCycles" \
        + " + system.littleCluster.cpus1.numCycles" \
        + " + system.littleCluster.cpus2.numCycles)/3)/sim_seconds)" \
        + "/(1/(system.littleCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * ((1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 1.84132144059e-10) + " \
        + "(((((system.littleCluster.cpus0.committedInsts" \
        + " + system.littleCluster.cpus1.committedInsts" \
        + " + system.littleCluster.cpus2.committedInsts)/3)/sim_seconds)" \
        + "/(1/(system.littleCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * ((1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 1.11008189839e-10) + " \
        + "(((((system.littleCluster.cpus0.dcache.overall_accesses::total" \
        + " + system.littleCluster.cpus1.dcache.overall_accesses::total" \
        + " + system.littleCluster.cpus2.dcache.overall_accesses::total)/3)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 2.7252658216e-10) + " \
        + "(((((system.littleCluster.cpus0.dcache.overall_misses::total" \
        + " + system.littleCluster.cpus1.dcache.overall_misses::total" \
        + " + system.littleCluster.cpus2.dcache.overall_misses::total)/3)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * 2.4016441235e-09) + " \
        + "(((((system.mem_ctrls.bytes_read::littleCluster.cpus0.data" \
        + " + system.mem_ctrls.bytes_read::littleCluster.cpus1.data" \
        + " + system.mem_ctrls.bytes_read::littleCluster.cpus2.data)/3)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
        + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + " * -2.44881613234e-09)"

    st = "((1) * 31.0366448991) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * -0.0267126706228) + " \
       + "((system.voltage_domain.voltage) * -87.7978467067) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage) * 0.0748426796784) + " \
       + "((system.voltage_domain.voltage*system.voltage_domain.voltage)" \
       + " * 82.5596011612) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage" \
       + "*system.voltage_domain.voltage) * -0.0696612748138) + " \
       + "((system.voltage_domain.voltage) * (system.voltage_domain.voltage" \
       + "*system.voltage_domain.voltage) * -25.8616662356) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage)" \
       + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
       + " * 0.0216526889381)"


class LittleCpuA7x4PowerOn(MathExprPowerModel):
    dyn = "(((((system.littleCluster.cpus0.numCycles" \
        + " + system.littleCluster.cpus1.numCycles" \
        + " + system.littleCluster.cpus2.numCycles" \
        + " + system.littleCluster.cpus3.numCycles)/4)/sim_seconds)" \
        + "/(1/(system.littleCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * ((1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 1.84132144059e-10) + " \
        + "(((((system.littleCluster.cpus0.committedInsts" \
        + " + system.littleCluster.cpus1.committedInsts" \
        + " + system.littleCluster.cpus2.committedInsts" \
        + " + system.littleCluster.cpus3.committedInsts)/4)/sim_seconds)" \
        + "/(1/(system.littleCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * ((1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * 1.11008189839e-10) + " \
        + "(((((system.littleCluster.cpus0.dcache.overall_accesses::total" \
        + " + system.littleCluster.cpus1.dcache.overall_accesses::total" \
        + " + system.littleCluster.cpus2.dcache.overall_accesses::total" \
        + " + system.littleCluster.cpus3.dcache.overall_accesses::total)/4)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
        + "* (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + "* 2.7252658216e-10) + " \
        + "(((((system.littleCluster.cpus0.dcache.overall_misses::total" \
        + " + system.littleCluster.cpus1.dcache.overall_misses::total" \
        + " + system.littleCluster.cpus2.dcache.overall_misses::total" \
        + " + system.littleCluster.cpus3.dcache.overall_misses::total)/4)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))/1000000)" \
        + "* (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
        + "* 2.4016441235e-09) + " \
        + "(((((system.mem_ctrls.bytes_read::littleCluster.cpus0.data" \
        + " + system.mem_ctrls.bytes_read::littleCluster.cpus1.data" \
        + " + system.mem_ctrls.bytes_read::littleCluster.cpus2.data" \
        + " + system.mem_ctrls.bytes_read::littleCluster.cpus3.data)/4)" \
        + "/sim_seconds)/(1/(system.littleCluster.clk_domain.clock" \
        + "/1000000000000))/1000000) * ((1" \
        + "/(system.littleCluster.clk_domain.clock/1000000000000))" \
        + "/1000000) * (system.voltage_domain.voltage" \
        + "*system.voltage_domain.voltage) * -2.44881613234e-09)"

    st = "((1) * 31.0366448991) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * -0.0267126706228) + " \
       + "((system.voltage_domain.voltage) * -87.7978467067) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage) * 0.0748426796784) + " \
       + "((system.voltage_domain.voltage*system.voltage_domain.voltage)" \
       + " * 82.5596011612) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage" \
       + "*system.voltage_domain.voltage) * -0.0696612748138) + " \
       + "((system.voltage_domain.voltage) * (system.voltage_domain.voltage" \
       + "*system.voltage_domain.voltage) * -25.8616662356) + " \
       + "(((1/(system.littleCluster.clk_domain.clock/1000000000000))" \
       + "/1000000) * (system.voltage_domain.voltage)" \
       + " * (system.voltage_domain.voltage*system.voltage_domain.voltage)" \
       + " * 0.0216526889381)"


# taken from `fs_power.py`
class CpuPowerOff(MathExprPowerModel):
    dyn = "0"
    st = "0"


# taken from `fs_power.py`
class ExamplePowerModel(PowerModel):
    pm = [
        ExampleCpuPowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]


class A15x1PowerModel(PowerModel):
    pm = [
        BigCpuA15x1PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]


class A15x2PowerModel(PowerModel):
    pm = [
        BigCpuA15x2PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]



class A15x3PowerModel(PowerModel):
    pm = [
        BigCpuA15x3PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]



class A15x4PowerModel(PowerModel):
    pm = [
        BigCpuA15x4PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]



class A7x1PowerModel(PowerModel):
    pm = [
        LittleCpuA7x1PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]



class A7x2PowerModel(PowerModel):
    pm = [
        LittleCpuA7x2PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]



class A7x3PowerModel(PowerModel):
    pm = [
        LittleCpuA7x3PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]



class A7x4PowerModel(PowerModel):
    pm = [
        LittleCpuA7x4PowerOn(),
        CpuPowerOff(),
        CpuPowerOff(),
        CpuPowerOff()
    ]


class L2PowerOn(MathExprPowerModel):
    # Example to report l2 Cache overall_accesses
    # The estimated power is converted to Watt and will vary based on the
    # size of the cache
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

