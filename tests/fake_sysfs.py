"""Build a fake /sys, /proc and /run tree for cputune tests.

The trees come from the files in tests/machines/. Each file is the output of
the old core_ranks.py on a real machine. This module adds the files that the
driver of that machine has: boost switches, rank files, base frequency, EPP.

The fake tree has static files. A real kernel changes some files when you
write to others, for example cpuinfo_max_freq when boost goes off. The fake
tree does not.

Command line: python3 fake_sysfs.py DIR MACHINE    (MACHINE: a file name without .txt)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

MACHINES = Path(__file__).parent / "machines"
BOOT_ID = "5e0a4c1f-0d6b-4b52-9f39-2f1c8f0e7a11"
LAPTOPS = ("intel-i9-13950hx", "amd-ryzen-ai7-350")  # these have a power supply in sysfs

# Base frequency in MHz from the vendor data sheets. Intel hybrid: P-cores and E-cores.
BASE_MHZ = {
    "13th Gen Intel(R) Core(TM) i9-13950HX": {5500: 2200, 5300: 2200, 4000: 1600},
    "AMD EPYC 9354P 32-Core Processor": 3250,
    "AMD Ryzen 5 5500GT with Radeon Graphics": 3600,
    "AMD Ryzen 9 7950X 16-Core Processor": 4501,
    "AMD Ryzen AI 7 350 w/ Radeon 860M": 2000,
}


@dataclass
class Machine:
    model: str
    governor: str
    governors: list[str]
    turbo: str
    driver: str
    kernel: str
    cores: list[tuple[int, list[int], int, int]] = field(default_factory=list)  # rank, cpus, min, max

    @property
    def cpus(self) -> list[int]:
        return sorted(c for core in self.cores for c in core[1])


def names() -> list[str]:
    return sorted(p.stem for p in MACHINES.glob("*.txt"))


def parse(name: str) -> Machine:
    text = (MACHINES / f"{name}.txt").read_text()
    head = dict(re.findall(r"^([A-Za-z ]+):\s+(.*)$", text, re.M))
    gov = re.fullmatch(r"(\S+) \((.*)\)", head["Governor"])
    m = Machine(
        model=head["CPU Model"],
        governor=gov[1],
        governors=[g.strip() for g in gov[2].split(",")],
        turbo=head["Turbo"],
        driver=head["Driver"],
        kernel=head["Kernel"].split()[0],
    )
    for row in re.findall(r"^\s*(\d+) \|\s*([\d, ]+) \|\s*(\d+) \|\s*(\d+)$", text, re.M):
        cpus = [int(c) for c in row[1].split(",")]
        m.cores.append((int(row[0]), cpus, int(row[2]), int(row[3])))
    return m


def w(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n")


def base_mhz(m: Machine, max_mhz: int) -> int | None:
    base = BASE_MHZ.get(m.model)
    return base.get(max_mhz) if isinstance(base, dict) else base


def build(root: Path, name: str) -> Machine:
    m = parse(name)
    cpu_base = root / "sys/devices/system/cpu"
    for rank, cpus, min_mhz, max_mhz in m.cores:
        base = base_mhz(m, max_mhz)
        for n in cpus:
            d = cpu_base / f"cpu{n}"
            f = d / "cpufreq"
            w(d / "topology/thread_siblings_list", ",".join(map(str, cpus)))
            w(f / "cpuinfo_min_freq", min_mhz * 1000)
            w(f / "cpuinfo_max_freq", max_mhz * 1000)
            w(f / "scaling_min_freq", min_mhz * 1000)
            w(f / "scaling_max_freq", max_mhz * 1000)
            w(f / "scaling_cur_freq", min(max_mhz, 3100) * 1000)
            w(f / "scaling_governor", m.governor)
            w(f / "scaling_available_governors", " ".join(m.governors))
            w(f / "scaling_driver", m.driver)
            if m.driver == "intel_pstate":
                w(d / "thermal_throttle/core_throttle_count", 0)
                w(d / "thermal_throttle/package_throttle_count", 0)
                w(d / "topology/itmt_score", rank)
                w(f / "base_frequency", base * 1000)
            if m.driver.startswith("amd-pstate") or m.driver == "acpi-cpufreq":
                w(d / "acpi_cppc/highest_perf", rank)
                w(d / "acpi_cppc/nominal_freq", base)
            if m.driver in ("intel_pstate", "amd-pstate-epp"):
                w(f / "energy_performance_preference", "balance_performance")
            if m.driver.startswith("amd-pstate"):
                w(f / "boost", 1)  # per-core switch, as on kernel 6.11 and newer
                # lowest_nonlinear_perf 85, nominal_perf 127, as on the Ryzen 9 7950X
                w(f / "amd_pstate_lowest_nonlinear_freq", base * 1000 * 85 // 127)
            if m.driver == "acpi-cpufreq":
                w(f / "scaling_available_frequencies", f"{base * 1000} 2300000 {min_mhz * 1000}")
                # As on the EPYC 9354P with kernel 7.1: the kernel keeps scaling_max_freq at the
                # highest P-state, also after a write of cpuinfo_max_freq. Boost goes above it.
                w(f / "scaling_max_freq", base * 1000)
                w(f / "bios_limit", base * 1000)
    if m.driver == "intel_pstate":
        w(cpu_base / "intel_pstate/no_turbo", 0)
    elif m.turbo != "unknown":
        w(cpu_base / "cpufreq/boost", 1)

    # Entries that are not CPUs with cpufreq, as on a real system
    (cpu_base / f"cpu{max(m.cpus) + 1}").mkdir()  # offline CPU
    (cpu_base / "cpuidle").mkdir()
    key = "Model" if "Raspberry" in m.model else "model name"
    w(root / "proc/cpuinfo", f"processor\t: 0\n{key}\t: {m.model}\n")
    w(root / "proc/sys/kernel/osrelease", m.kernel)
    w(root / "proc/sys/kernel/version", "#1 SMP PREEMPT_DYNAMIC")
    w(root / "proc/sys/kernel/random/boot_id", BOOT_ID)
    w(root / "proc/sys/kernel/randomize_va_space", 2)
    # All CPUs idle. The file does not change, so cputune measures no load.
    w(root / "proc/stat", "cpu  0 0 0 0 0 0 0 0 0 0\n"
      + "".join(f"cpu{n} 1000 0 500 90000 100 0 20 0 0 0\n" for n in m.cpus))
    if name in LAPTOPS:
        w(root / "sys/class/power_supply/AC/type", "Mains")
        w(root / "sys/class/power_supply/AC/online", 1)
        w(root / "sys/class/power_supply/BAT0/type", "Battery")
    (root / "run").mkdir(exist_ok=True)
    return m


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in names():
        sys.exit(f"usage: {sys.argv[0]} DIR MACHINE\nMACHINE: {', '.join(names())}")
    build(Path(sys.argv[1]), sys.argv[2])
