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
# Authors: Gabor Dozsa
#          Andreas Sandberg
#          Thomas E. Hansen

# This is an example configuration script for full system simulation of
# a generic ARM bigLITTLE system.
# Extensions have been made by Thomas E. Hansen to use DVFS as part of an
# integrated masters project at the University of St Andrews


from __future__ import print_function
from __future__ import absolute_import

import argparse
import os
import sys
import m5
import m5.util
from m5.objects import *

m5.util.addToPath("../../")

from common import FSConfig
from common import SysPaths
from common import ObjectList
from common import Options
from common.cores.arm import ex5_big, ex5_LITTLE

import devices
from devices import AtomicCluster, KvmCluster, FastmodelCluster

import fs_bL_power_models as bL_PMs


default_disk = 'aarch64-ubuntu-trusty-headless.img'
default_rcs = 'bootscript.rcS'

default_mem_size= "2GB"

def _to_ticks(value):
    """Helper function to convert a latency from string format to Ticks"""

    return m5.ticks.fromSeconds(m5.util.convert.anyToLatency(value))

def _using_pdes(root):
    """Determine if the simulator is using multiple parallel event queues"""

    for obj in root.descendants():
        if not m5.proxy.isproxy(obj.eventq_index) and \
               obj.eventq_index != root.eventq_index:
            return True

    return False


class DVFSBigCluster(devices.CpuCluster):
    def __init__(self, system, num_cpus, cpu_clock=None, cpu_voltage=None):
        # use the same CPU as BigCluster
        abstract_cpu = ObjectList.cpu_list.get("O3_ARM_v7a_3")
        # voltage values from Odroid N2 board /sys/class/regulator/regulator.2
        voltages = [  '0.981000V'
                    , '0.891000V'
                    , '0.861000V'
                    , '0.821000V'
                    , '0.791000V'
                    , '0.771000V'
                    , '0.771000V'
                    , '0.751000V'
                   ]
        # clock steps from Odroid N2 board using `cpupower`
        clock_steps = [  '1800MHz'
                       , '1700MHz'
                       , '1610MHz'
                       , '1510MHz'
                       , '1400MHz'
                       , '1200MHz'
                       , '1000MHz'
                       ,  '667MHz'
                      ]
        # since we're using custom DVFS values, the kwargs do nothing
        if cpu_clock is not None:
            print("Warn: Specifying CPU clock for the DVFS model does",
                  "nothing", file=sys.stderr)
        if cpu_voltage is not None:
            print("Warn: Specifying CPU voltage for the DVFS model does",
                  "nothing", file=sys.stderr)
        super(DVFSBigCluster, self).__init__(system, num_cpus,
                                             clock_steps, voltages,
                                             abstract_cpu,
                                             devices.L1I, devices.L1D,
                                             devices.WalkCache,
                                             devices.L2)


class DVFSLittleCluster(devices.CpuCluster):
    def __init__(self, system, num_cpus, cpu_clock=None, cpu_voltage=None):
        # use the same CPU as LittleCluster
        abstract_cpu = ObjectList.cpu_list.get("MinorCPU")
        # voltage values from Odroid N2 board /sys/class/regulator/regulator.1
        voltages = [  '0.981000V'
                    , '0.861000V'
                    , '0.831000V'
                    , '0.791000V'
                    , '0.761000V'
                    , '0.731000V'
                    , '0.731000V'
                    , '0.731000V'
                   ]
        # clock steps from Odroid N2 board using `cpupower`
        clock_steps = [  '1900MHz'
                       , '1700MHz'
                       , '1610MHz'
                       , '1510MHz'
                       , '1400MHz'
                       , '1200MHz'
                       , '1000MHz'
                       ,  '667MHz'
                      ]
        # since we're using custom DVFS values, the kwargs do nothing
        if cpu_clock is not None:
            print("Warn: Specifying CPU clock for the DVFS model does",
                  "nothing", file=sys.stderr)
        if cpu_voltage is not None:
            print("Warn: Specifying CPU voltage for the DVFS model does",
                  "nothing", file=sys.stderr)
        super(DVFSLittleCluster, self).__init__(system, num_cpus,
                                                clock_steps, voltages,
                                                abstract_cpu,
                                                devices.L1I, devices.L1D,
                                                devices.WalkCache,
                                                devices.L2)

class DVFSBigAtomicCluster(AtomicCluster):
    def __init__(self, system, num_cpus, cpu_clock=None, cpu_voltage=None):
        # the CPU will always be the "Atomic" one, but DVFS changes
        abstract_cpu = ObjectList.cpu_list.get("AtomicSimpleCPU")
        # voltage values from Odroid N2 board /sys/class/regulator/regulator.2
        voltages = [  '0.981000V'
                    , '0.891000V'
                    , '0.861000V'
                    , '0.821000V'
                    , '0.791000V'
                    , '0.771000V'
                    , '0.771000V'
                    , '0.751000V'
                   ]
        # clock steps from Odroid N2 board using `cpupower`
        clock_steps = [  '1800MHz'
                       , '1700MHz'
                       , '1610MHz'
                       , '1510MHz'
                       , '1400MHz'
                       , '1200MHz'
                       , '1000MHz'
                       ,  '667MHz'
                      ]
        # since we're using custom DVFS values, the kwargs do nothing
        if cpu_clock is not None:
            print("Warn: Specifying CPU clock for the DVFS model does",
                  "nothing", file=sys.stderr)
        if cpu_voltage is not None:
            print("Warn: Specifying CPU voltage for the DVFS model does",
                  "nothing", file=sys.stderr)
        super(DVFSBigAtomicCluster, self).__init__(system, num_cpus,
                                                   clock_steps,
                                                   voltages)


class DVFSLittleAtomicCluster(AtomicCluster):
    def __init__(self, system, num_cpus, cpu_clock=None, cpu_voltage=None):
        # the CPU will always be the "Atomic" one, but DVFS changes
        abstract_cpu = ObjectList.cpu_list.get("AtomicSimpleCPU")
        # voltage values from Odroid N2 board /sys/class/regulator/regulator.2
        voltages = [  '0.981000V'
                    , '0.861000V'
                    , '0.831000V'
                    , '0.791000V'
                    , '0.761000V'
                    , '0.731000V'
                    , '0.731000V'
                    , '0.731000V'
                   ]
        # clock steps from Odroid N2 board using `cpupower`
        clock_steps = [  '1900MHz'
                       , '1700MHz'
                       , '1610MHz'
                       , '1510MHz'
                       , '1400MHz'
                       , '1200MHz'
                       , '1000MHz'
                       ,  '667MHz'
                      ]
        # since we're using custom DVFS values, the kwargs do nothing
        if cpu_clock is not None:
            print("Warn: Specifying CPU clock for the DVFS model does",
                  "nothing", file=sys.stderr)
        if cpu_voltage is not None:
            print("Warn: Specifying CPU voltage for the DVFS model does",
                  "nothing", file=sys.stderr)
        super(DVFSLittleAtomicCluster, self).__init__(system, num_cpus,
                                                      clock_steps,
                                                      voltages)


class BigCluster(devices.CpuCluster):
    def __init__(self, system, num_cpus, cpu_clock,
                 cpu_voltage="1.0V"):
        cpu_config = [ ObjectList.cpu_list.get("O3_ARM_v7a_3"),
            devices.L1I, devices.L1D, devices.WalkCache, devices.L2 ]
        super(BigCluster, self).__init__(system, num_cpus, cpu_clock,
                                         cpu_voltage, *cpu_config)

class LittleCluster(devices.CpuCluster):
    def __init__(self, system, num_cpus, cpu_clock,
                 cpu_voltage="1.0V"):
        cpu_config = [ ObjectList.cpu_list.get("MinorCPU"), devices.L1I,
            devices.L1D, devices.WalkCache, devices.L2 ]
        super(LittleCluster, self).__init__(system, num_cpus, cpu_clock,
                                         cpu_voltage, *cpu_config)

class Ex5BigCluster(devices.CpuCluster):
    def __init__(self, system, num_cpus, cpu_clock,
                 cpu_voltage="1.0V"):
        cpu_config = [ ObjectList.cpu_list.get("ex5_big"), ex5_big.L1I,
            ex5_big.L1D, ex5_big.WalkCache, ex5_big.L2 ]
        super(Ex5BigCluster, self).__init__(system, num_cpus, cpu_clock,
                                         cpu_voltage, *cpu_config)

class Ex5LittleCluster(devices.CpuCluster):
    def __init__(self, system, num_cpus, cpu_clock,
                 cpu_voltage="1.0V"):
        cpu_config = [ ObjectList.cpu_list.get("ex5_LITTLE"),
            ex5_LITTLE.L1I, ex5_LITTLE.L1D, ex5_LITTLE.WalkCache,
            ex5_LITTLE.L2 ]
        super(Ex5LittleCluster, self).__init__(system, num_cpus, cpu_clock,
                                         cpu_voltage, *cpu_config)

def createSystem(caches, kernel, bootscript, machine_type="VExpress_GEM5",
                 disks=[],  mem_size=default_mem_size, bootloader=None):
    platform = ObjectList.platform_list.get(machine_type)
    m5.util.inform("Simulated platform: %s", platform.__name__)

    sys = devices.simpleSystem(LinuxArmSystem,
                               caches, mem_size, platform(),
                               kernel=SysPaths.binary(kernel),
                               readfile=bootscript)

    sys.mem_ctrls = [ SimpleMemory(range=r, port=sys.membus.master)
                      for r in sys.mem_ranges ]

    sys.connect()

    # Attach disk images
    if disks:
        def cow_disk(image_file):
            image = CowDiskImage()
            image.child.image_file = SysPaths.disk(image_file)
            return image

        sys.disk_images = [ cow_disk(f) for f in disks ]
        sys.pci_vio_block = [ PciVirtIO(vio=VirtIOBlock(image=img))
                              for img in sys.disk_images ]
        for dev in sys.pci_vio_block:
            sys.attach_pci(dev)

    sys.realview.setupBootLoader(sys, SysPaths.binary, bootloader)

    return sys

cpu_types = {
    "atomic" : (AtomicCluster, AtomicCluster),
    "timing" : (BigCluster, LittleCluster),
    "exynos" : (Ex5BigCluster, Ex5LittleCluster),
    "dvfs-timing" : (DVFSBigCluster, DVFSLittleCluster),
    "dvfs-atomic" : (DVFSBigAtomicCluster, DVFSLittleAtomicCluster),
}

# Only add the KVM CPU if it has been compiled into gem5
if devices.have_kvm:
    cpu_types["kvm"] = (KvmCluster, KvmCluster)

# Only add the FastModel CPU if it has been compiled into gem5
if devices.have_fastmodel:
    cpu_types["fastmodel"] = (FastmodelCluster, FastmodelCluster)

def addOptions(parser):
    parser.add_argument("--restore-from", type=str, default=None,
                        help="Restore from checkpoint")
    parser.add_argument("--dtb", type=str, default=None,
                        help="DTB file to load")
    parser.add_argument("--kernel", type=str, required=True,
                        help="Linux kernel")
    parser.add_argument("--root", type=str, default="/dev/vda1",
                        help="Specify the kernel CLI root= argument")
    parser.add_argument("--machine-type", type=str,
                        choices=ObjectList.platform_list.get_names(),
                        default="VExpress_GEM5",
                        help="Hardware platform class")
    parser.add_argument("--disk", action="append", type=str, default=[],
                        help="Disks to instantiate")
    parser.add_argument("--bootscript", type=str, default=default_rcs,
                        help="Linux bootscript")
    parser.add_argument("--cpu-type", type=str, choices=cpu_types.keys(),
                        default="timing",
                        help="CPU simulation mode. Default: %(default)s")
    parser.add_argument("--kernel-init", type=str, default="/sbin/init",
                        help="Override init")
    parser.add_argument("--big-cpus", type=int, default=1,
                        help="Number of big CPUs to instantiate")
    parser.add_argument("--little-cpus", type=int, default=1,
                        help="Number of little CPUs to instantiate")
    parser.add_argument("--caches", action="store_true", default=False,
                        help="Instantiate caches")
    parser.add_argument("--last-cache-level", type=int, default=2,
                        help="Last level of caches (e.g. 3 for L3)")
    parser.add_argument("--big-cpu-clock", type=str, default="2GHz",
                        help="Big CPU clock frequency")
    parser.add_argument("--little-cpu-clock", type=str, default="1GHz",
                        help="Little CPU clock frequency")
    parser.add_argument("--sim-quantum", type=str, default="1ms",
                        help="Simulation quantum for parallel simulation. " \
                        "Default: %(default)s")
    parser.add_argument("--mem-size", type=str, default=default_mem_size,
                        help="System memory size")
    parser.add_argument("--kernel-cmd", type=str, default=None,
                        help="Custom Linux kernel command")
    parser.add_argument("--bootloader", action="append",
                        help="executable file that runs before the --kernel")
    parser.add_argument("-P", "--param", action="append", default=[],
        help="Set a SimObject parameter relative to the root node. "
             "An extended Python multi range slicing syntax can be used "
             "for arrays. For example: "
             "'system.cpu[0,1,3:8:2].max_insts_all_threads = 42' "
             "sets max_insts_all_threads for cpus 0, 1, 3, 5 and 7 "
             "Direct parameters of the root object are not accessible, "
             "only parameters of its children.")
    parser.add_argument("--vio-9p", action="store_true",
                        help=Options.vio_9p_help)
    parser.add_argument("--power-models", action="store_true", default=False,
                        help="Add power models to the simulated system. Only "
                             "works with 'timing' CPU-models.")
    parser.add_argument("--stat-freq", type=float, default=1.0,
                        help="How often (in seconds) to dump stats. Accepts "
                             "scientific notation, e.g. 1.0E-3")
    parser.add_argument("--example-power", action="store_true", default=False,
                        help="Use the very basic, example power models "
                             "provided in `fs_power.py`.")
    parser.add_argument("--pmus", action="store_true", default=False,
                        help="Enable PMU events for the simulated system."
                             " Which events are counted can be controlled"
                             " through the --pmu-events flag.")
#    parser.add_argument("--pmu-events", default="all",
#                        help="What PMU events to enable in the system. The"
#                             " default is 'all', which enables all the PMU"
#                             " events. Alternatively, a single event number"
#                             " or a list of events (e.g."
#                             " --pmu-events=0x01,0x02,0x03) may be passed."
#                             " NOTE: nothing will happen if the --pmus flag"
#                             " is not used along with this flag.")
    return parser


def _apply_pm(cpus, power_model):
    for cpu in cpus:
        cpu.default_p_state = "ON"
        cpu.power_model = power_model()


def build(options):
    m5.ticks.fixGlobalFrequency()

    kernel_cmd = [
        "earlyprintk=pl011,0x1c090000",
        "console=ttyAMA0",
        "lpj=19988480",
        "norandmaps",
        "loglevel=8",
        "mem=%s" % options.mem_size,
        "root=%s" % options.root,
        "rw",
        "init=%s" % options.kernel_init,
        "vmalloc=768MB",
    ]

    root = Root(full_system=True)

    disks = [default_disk] if len(options.disk) == 0 else options.disk
    system = createSystem(options.caches,
                          options.kernel,
                          options.bootscript,
                          options.machine_type,
                          disks=disks,
                          mem_size=options.mem_size,
                          bootloader=options.bootloader)

    root.system = system
    if options.kernel_cmd:
        system.boot_osflags = options.kernel_cmd
    else:
        system.boot_osflags = " ".join(kernel_cmd)

    if options.big_cpus + options.little_cpus == 0:
        m5.util.panic("Empty CPU clusters")

    big_model, little_model = cpu_types[options.cpu_type]

    all_cpus = []
    # big cluster
    if options.big_cpus > 0:
        system.bigCluster = big_model(system, options.big_cpus,
                                      options.big_cpu_clock)
        system.mem_mode = system.bigCluster.memoryMode()
        all_cpus += system.bigCluster.cpus

    # little cluster
    if options.little_cpus > 0:
        system.littleCluster = little_model(system, options.little_cpus,
                                            options.little_cpu_clock)
        system.mem_mode = system.littleCluster.memoryMode()
        all_cpus += system.littleCluster.cpus

    # Figure out the memory mode
    if options.big_cpus > 0 and options.little_cpus > 0 and \
       system.bigCluster.memoryMode() != system.littleCluster.memoryMode():
        m5.util.panic("Memory mode missmatch among CPU clusters")


    # create caches
    system.addCaches(options.caches, options.last_cache_level)
    if not options.caches:
        if options.big_cpus > 0 and system.bigCluster.requireCaches():
            m5.util.panic("Big CPU model requires caches")
        if options.little_cpus > 0 and system.littleCluster.requireCaches():
            m5.util.panic("Little CPU model requires caches")

    # Create a KVM VM and do KVM-specific configuration
    if issubclass(big_model, KvmCluster):
        _build_kvm(system, all_cpus)

    # Linux device tree
    if options.dtb is not None:
        system.dtb_filename = SysPaths.binary(options.dtb)
    else:
        system.generateDtb(m5.options.outdir, 'system.dtb')

    if devices.have_fastmodel and issubclass(big_model, FastmodelCluster):
        from m5 import arm_fast_model as fm, systemc as sc
        # setup FastModels for simulation
        fm.setup_simulation("cortexa76")
        # setup SystemC
        root.systemc_kernel = m5.objects.SystemC_Kernel()
        m5.tlm.tlm_global_quantum_instance().set(
            sc.sc_time(10000.0 / 100000000.0, sc.sc_time.SC_SEC))

    if options.vio_9p:
        FSConfig.attach_9p(system.realview, system.iobus)

    # teh6: add dvfs handler to system
    if 'dvfs' in options.cpu_type:
        system.dvfs_handler.domains = [  system.bigCluster.clk_domain
                                       , system.littleCluster.clk_domain
                                      ]
        system.dvfs_handler.enable = True
        #print(vars(system.dvfs_handler.enable))
        #print(vars(system.dvfs_handler))
        #print(system.dvfs_handler.domains)
        #for dom in system.dvfs_handler.domains:
        #    print(dir(dom))

    # teh6: add power models
    if options.power_models:
        if options.example_power:
            m5.fatal("Both power models cannot be enabled at the same time.")
        if options.cpu_type != "dvfs-timing":
            m5.fatal("The power models require the 'dvfs-timing' CPU-type.")
        if options.big_cpus > 4 or options.little_cpus > 4:
            m5.fatal("The power models only work for up to 4 big and 4"\
                     + " LITTLE CPUs.")
        # big cluster
        if options.big_cpus == 1:
            _apply_pm(root.system.bigCluster.cpus, bL_PMs.A15x1PowerModel)
        elif options.big_cpus == 2:
            _apply_pm(root.system.bigCluster.cpus, bL_PMs.A15x2PowerModel)
        elif options.big_cpus == 3:
            _apply_pm(root.system.bigCluster.cpus, bL_PMs.A15x3PowerModel)
        else:
            _apply_pm(root.system.bigCluster.cpus, bL_PMs.A15x4PowerModel)

        # little cluster
        if options.little_cpus == 1:
            _apply_pm(root.system.littleCluster.cpus, bL_PMs.A7x1PowerModel)
        elif options.little_cpus == 2:
            _apply_pm(root.system.littleCluster.cpus, bL_PMs.A7x2PowerModel)
        elif options.little_cpus == 3:
            _apply_pm(root.system.littleCluster.cpus, bL_PMs.A7x3PowerModel)
        else:
            _apply_pm(root.system.littleCluster.cpus, bL_PMs.A7x4PowerModel)

        # big L2 cache (as per the `fs_power.py` file)
        for l2 in root.system.bigCluster.l2.descendants():
            if not isinstance(l2, m5.objects.Cache):
                continue
            l2.default_p_state = "ON"
            l2.power_model = bL_PMs.L2PowerModel()
    elif options.example_power:
        _apply_pm(root.system.bigCluster.cpus, bL_PMs.ExamplePowerModel)
        _apply_pm(root.system.littleCluster.cpus, bL_PMs.ExamplePowerModel)

    # teh6: add pmus
#    if not options.pmus and options.pmu_events != "all":
#        print("Warn: PMU events set, but PMUs not enabled. Did you mean to",
#              "pass in --pmus as well?")
    if options.pmus:
        root.system.bigCluster\
            .addPMUs(ints=[i for i in range(options.big_cpus)])
        root.system.littleCluster\
            .addPMUs(ints=[i for i in range(options.big_cpus,
                                            options.big_cpus
                                            + options.little_cpus)])

#        pmu_events = []
#        if options.pmu_events == 'all':
#            pass    #TODO
#        elif ',' in options.pmu_events:
#            pmu_events = list(map(lambda h: int(h, 16),
#                                  options.pmu_events.split(',')))
#        else:
#            pmu_events = [int(options.pmu_events, 16)]
#        print(pmu_events)
#        root.system.bigCluster\
#            .addPMUs(ints=[i for i in range(options.big_cpus)],
#                     events=pmu_events)
#        root.system.littleCluster\
#            .addPMUs(ints=[i for i in range(options.big_cpus,
#                                            options.big_cpus)],
#                     events=pmu_events)

    return root


def _build_kvm(system, cpus):
    system.kvm_vm = KvmVM()

    # Assign KVM CPUs to their own event queues / threads. This
    # has to be done after creating caches and other child objects
    # since these mustn't inherit the CPU event queue.
    if len(cpus) > 1:
        device_eq = 0
        first_cpu_eq = 1
        for idx, cpu in enumerate(cpus):
            # Child objects usually inherit the parent's event
            # queue. Override that and use the same event queue for
            # all devices.
            for obj in cpu.descendants():
                obj.eventq_index = device_eq
            cpu.eventq_index = first_cpu_eq + idx



def instantiate(options, checkpoint_dir=None):
    # Setup the simulation quantum if we are running in PDES-mode
    # (e.g., when using KVM)
    root = Root.getInstance()
    if root and _using_pdes(root):
        m5.util.inform("Running in PDES mode with a %s simulation quantum.",
                       options.sim_quantum)
        root.sim_quantum = _to_ticks(options.sim_quantum)

    # Get and load from the chkpt or simpoint checkpoint
    if options.restore_from:
        if checkpoint_dir and not os.path.isabs(options.restore_from):
            cpt = os.path.join(checkpoint_dir, options.restore_from)
        else:
            cpt = options.restore_from

        m5.util.inform("Restoring from checkpoint %s", cpt)
        m5.instantiate(cpt)
    else:
        m5.instantiate()


def run(checkpoint_dir=m5.options.outdir):
    # start simulation (and drop checkpoints when requested)
    while True:
        event = m5.simulate()
        exit_msg = event.getCause()
        if exit_msg == "checkpoint":
            print("Dropping checkpoint at tick %d" % m5.curTick())
            cpt_dir = os.path.join(checkpoint_dir, "cpt.%d" % m5.curTick())
            m5.checkpoint(cpt_dir)
            print("Checkpoint done.")
        else:
            print(exit_msg, " @ ", m5.curTick())
            break

    sys.exit(event.getCode())


def main():
    parser = argparse.ArgumentParser(
        description="Generic ARM big.LITTLE configuration")
    addOptions(parser)
    options = parser.parse_args()
    root = build(options)
    root.apply_config(options.param)
    instantiate(options)
    if options.example_power:
        print("*" * 70)
        print("WARNING: The power numbers generated by this script are "
              "examples. They are not representative of any particular "
              "implementation or process.")
        print("*" * 70)
    print("A" * 80)
#    print(root.system.bigCluster.getStatGroups())
#    print(dir(root.system.bigCluster.cpus[0]))
#    print(root.system.bigCluster.cpus[0]._parent)
#    print("root.system.bigCluster.getStatGroups():",
#          root.system.bigCluster.getStatGroups())
#    print("root.system.bigCluster.getStats():",
#          root.system.bigCluster.getStats())
#    print("root.system.bigCluster.cpus[0].getStats() details:")
#    for thing in root.system.bigCluster.cpus[0].getStats():
#        print('\t', dir(thing))
#        print("\t\tname:", thing.name, "\n\t\tdesc:", thing.desc)
    m5.stats.periodicStatDump(m5.ticks.fromSeconds(options.stat_freq))
    run()


if __name__ == "__m5_main__":
    main()
