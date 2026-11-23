"""Tests for cputune. Run: python3 -m unittest discover -s tests

Each test builds a fake sysfs tree from a recorded machine (tests/machines/)
and runs the real script against it with CPUTUNE_ROOT. Then cputune does not
call sudo, and it does not put load on the real CPUs.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import json
import io
import os
import pwd
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fake_sysfs  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "cputune"
# Fake trees in memory when possible: a test builds hundreds of small files, 8 times faster there.
# Docker mounts /dev/shm noexec: a test that runs files of its tree sets tmp = None (disk).
TMP = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None


def fake_sysfs_cpus(text: str) -> list[int]:
    cpus = []
    for part in text.strip().split(","):
        a, _, b = part.partition("-")
        cpus.extend(range(int(a), int(b or a) + 1))
    return cpus


class Case(unittest.TestCase):
    machine = "intel-i9-13950hx"
    tmp = TMP

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="cputune-test-", dir=self.tmp))
        self.addCleanup(shutil.rmtree, self.root)
        fake_sysfs.build(self.root, self.machine)
        self.cpu = self.root / "sys/devices/system/cpu"
        self.state = self.root / "run/cputune.json"

    def run_tool(self, *args: str, columns: int = 100, script: Path = SCRIPT, **env_vars: str):
        env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "COLUMNS")}
        env.update(CPUTUNE_ROOT=str(self.root), COLUMNS=str(columns), **env_vars)
        return subprocess.run(
            [sys.executable, str(script), *args], env=env, capture_output=True, text=True, timeout=60
        )

    def ok(self, *args: str) -> subprocess.CompletedProcess:
        result = self.run_tool(*args)
        self.assertEqual(result.returncode, 0, f"{args}: {result.stderr}")
        return result

    def fails(self, *args: str, code: int = 1) -> subprocess.CompletedProcess:
        result = self.run_tool(*args)
        self.assertEqual(result.returncode, code, f"{args}: {result.stdout} {result.stderr}")
        return result

    def json(self, *args: str) -> dict:
        return json.loads(self.ok(*args, "--json").stdout)

    def f(self, cpu: int, name: str) -> str:
        return (self.cpu / f"cpu{cpu}/cpufreq" / name).read_text().strip()

    def set_f(self, cpu: int, name: str, value: object) -> None:
        (self.cpu / f"cpu{cpu}/cpufreq" / name).write_text(f"{value}\n")

    def snapshot(self) -> dict[str, str]:
        files = {}
        for path in sorted((self.root / "sys").rglob("*")):
            if path.is_file():
                files[str(path.relative_to(self.root))] = path.read_text()
        return files


class ShowTest(Case):
    def test_all_machines(self) -> None:
        for name in fake_sysfs.names():
            with self.subTest(machine=name):
                shutil.rmtree(self.root)
                machine = fake_sysfs.build(self.root, name)
                data = self.json("show")
                self.assertEqual(data["system"]["cpu_model"], machine.model)
                self.assertEqual(data["system"]["driver"], machine.driver)
                self.assertEqual(len(data["cores"]), len(machine.cores))
                self.assertEqual(sorted(c["cpus"] for c in data["cores"]), sorted(c[1] for c in machine.cores))
                ranks = [c["rank"] for c in data["cores"]]
                self.assertEqual(ranks, sorted(ranks, reverse=True))
                self.assertNotIn(0, data["default_lock_cpus"])
                self.assertIsNone(data["changed_by_cputune"])
                for color in ("always", "never"):
                    text = self.ok("show", "--color", color).stdout
                    self.assertIn(machine.model, text)
                    self.assertEqual("\033[" in text, color == "always")

    def test_default_lock_core(self) -> None:
        # Best rank without CPU 0. On a tie, the lowest CPU IDs.
        expected = {
            "intel-i9-13950hx": [8, 9],
            "amd-ryzen9-7950x": [1, 17],
            "amd-epyc-9354p": [1, 33],
            "amd-ryzen5-5500gt": [2, 8],  # 0,6 has the same rank but holds CPU 0
            "rpi4": [1],
        }
        for name, cpus in expected.items():
            with self.subTest(machine=name):
                shutil.rmtree(self.root)
                fake_sysfs.build(self.root, name)
                self.assertEqual(self.json()["default_lock_cpus"], cpus)

    def test_base_frequency_sources(self) -> None:
        cores = {tuple(c["cpus"]): c for c in self.json("show")["cores"]}
        self.assertEqual(cores[(8, 9)]["base_mhz"], 2200)  # Intel base_frequency, P-core
        self.assertEqual(cores[(16,)]["base_mhz"], 1600)  # E-core

    def test_show_starts_without_slow_imports(self) -> None:
        # Show and watch run often and next to benchmarks: they must not load what only run,
        # lock and install need. -X importtime lists each module that Python loads.
        def loaded(*args: str) -> set[str]:
            env = dict(os.environ, CPUTUNE_ROOT=str(self.root))
            result = subprocess.run([sys.executable, "-X", "importtime", *args], env=env, capture_output=True,
                                    text=True, timeout=60)
            return {line.rsplit("|", 1)[-1].strip() for line in result.stderr.splitlines()}

        # What Python loads anyway for the same top-level imports and an argparse parser (3.9 loads
        # threading, 3.14 dataclasses): not the fault of cputune.
        base = loaded("-c", "import argparse, json, math, os, pwd, re, shutil, sys, textwrap, time, pathlib\n"
                            "p = argparse.ArgumentParser(); p.add_argument('--x'); p.parse_args([])")
        extra = loaded(str(SCRIPT)) - base
        for module in ("subprocess", "threading", "signal", "tempfile", "shlex", "dataclasses", "typing"):
            self.assertNotIn(module, extra)

    def test_one_rule_for_the_current_frequency(self) -> None:
        with mock.patch.dict(os.environ, CPUTUNE_ROOT=str(self.root)):
            m = load_module()
        avg = self.cpu / "cpu8/cpufreq/cpuinfo_avg_freq"  # scaling_cur_freq is 3100000 in the fake tree
        self.assertEqual((m.cpu_khz(8), m.cpu_khz(8, busy_only=True)), (3100000, 3100000))  # old kernel
        avg.write_text("4200000\n")
        self.assertEqual((m.cpu_khz(8), m.cpu_khz(8, busy_only=True)), (4200000, 4200000))
        avg.write_text("")  # a sleeping CPU: the file has no value
        self.assertEqual((m.cpu_khz(8), m.cpu_khz(8, busy_only=True)), (3100000, None))
        reader = m.FdReader()
        self.assertEqual(m.cpu_khz(8, reader.int), 3100000)
        # watch shows the sleeping core with scaling_cur_freq, not as "no value"
        avg.unlink()
        data = json.loads(self.ok("-w", "--count", "1", "--json").stdout)
        self.assertEqual(next(c for c in data["cores"] if c["cpus"] == [8, 9])["now_mhz"], 3100)

    def test_fd_reader_opens_a_missing_file_once(self) -> None:
        with mock.patch.dict(os.environ, CPUTUNE_ROOT=str(self.root)):
            m = load_module()
        reader, missing = m.FdReader(), self.cpu / "cpu8/cpufreq/cpuinfo_avg_freq"
        with mock.patch("os.open", side_effect=FileNotFoundError) as opened:
            self.assertIsNone(reader.int(missing))
            self.assertIsNone(reader.int(missing))
        self.assertEqual(opened.call_count, 1)  # watch reads it in every frame: no open() each time
        reader.close()

    def test_measure_skips_failed_reads(self) -> None:
        with mock.patch.dict(os.environ, CPUTUNE_ROOT=str(self.root)):
            m = load_module()
        # cpuinfo_avg_freq fails, and scaling_cur_freq fails once in 5 samples: the average is 4000, not 3200.
        cur = iter([4000000, None, 4000000, 4000000, 4000000])
        read_int = lambda path: next(cur) if path.name == "scaling_cur_freq" else None  # noqa: E731
        with mock.patch.object(m, "read_int", read_int):
            self.assertEqual(m.measure([8]), {8: 4000})
        with mock.patch.object(m, "read_int", lambda path: None):
            self.assertEqual(m.measure([8]), {8: 0})

    def legend(self, color: str, columns: int = 100) -> tuple[str, list[str]]:
        """The text above the legend of show, and the lines of the legend."""
        lines = self.run_tool("show", "--color", color, columns=columns).stdout.splitlines()
        start = next(i for i, line in enumerate(lines) if "min to " in line)
        end = next(i for i, line in enumerate(lines) if "all values in MHz" in line)
        return "\n".join(lines[:start]), lines[start:end + 1]

    def test_compact_header(self) -> None:
        # The kernel is on the line of the CPU model, right-aligned to the table; the GHz labels are in
        # the header row of the table, not on a line of their own.
        lines = self.run_tool("show", "--color", "never", columns=120).stdout.splitlines()
        kernel = (self.root / "proc/sys/kernel/osrelease").read_text().strip()
        self.assertTrue(lines[0].startswith("13th Gen Intel"), lines[0])
        self.assertTrue(lines[0].endswith(f"kernel {kernel}"), lines[0])
        self.assertEqual(lines[2], "")
        head = lines[3]
        self.assertTrue(head.startswith("rank  cpus") and "0 GHz" in head, head)
        self.assertEqual(head.index("0 GHz"), head.index("max") + 5)  # where the chart starts
        table = lines[3:lines.index("", 3)]
        self.assertEqual(len(lines[0]), max(map(len, table)))
        narrow = self.run_tool("show", "--color", "never", columns=60).stdout.splitlines()
        self.assertEqual(narrow[2], f"kernel {kernel}")  # does not fit next to the CPU model

    def test_legend_fits_the_terminal(self) -> None:
        for columns in (40, 60, 70, 200):  # 70: the first three marks fill exactly 70 columns
            with self.subTest(columns=columns):
                legend = self.legend("never", columns)[1]
                self.assertTrue(all(len(line) < columns for line in legend), legend)
                self.assertEqual(len(legend) == 1, columns == 200)
                self.assertTrue(any("★ best core: lock and run use it" in line for line in legend))  # not split

    def test_legend_marks_have_the_chart_colors(self) -> None:
        # ━ and ╍ look alike: only the color tells base range and boost apart, also in the legend.
        chart, legend = self.legend("always")
        legend = "\n".join(legend)
        colors = {}
        for mark in ("━", "╍"):
            found = re.search(rf"\033\[([0-9;]+)m{mark}", legend)
            self.assertIsNotNone(found, f"{mark} has no color in the legend: {legend!r}")
            colors[mark] = found[1]
            self.assertIn(f"\033[{found[1]}m{mark}", chart)  # the same color as in the chart
        self.assertNotEqual(colors["━"], colors["╍"])

    def test_now_and_widths(self) -> None:
        for columns in (40, 80, 250):
            result = self.run_tool("show", "--now", "--color", "always", columns=columns)
            self.assertEqual(result.returncode, 0, result.stderr)
        data = self.json("show", "--now")
        self.assertEqual({c["now_mhz"] for c in data["cores"]}, {3100})

    def test_without_cpufreq(self) -> None:
        for path in self.cpu.glob("cpu*/cpufreq"):
            shutil.rmtree(path)
        self.assertIn("no cpufreq", self.ok("show").stdout)
        self.assertIsNone(self.json("show")["default_lock_cpus"])
        self.assertIn("no CPU has cpufreq", self.fails("lock").stderr)

    def test_without_topology(self) -> None:
        for path in self.cpu.glob("cpu*/topology"):
            shutil.rmtree(path)
        data = self.json("show")
        self.assertEqual(len(data["cores"]), 32)  # one core for each CPU with cpufreq
        for path in self.cpu.glob("cpu*/cpufreq"):
            shutil.rmtree(path)
        self.assertIn("No CPU topology found", self.ok("show").stdout)
        self.assertEqual(self.json("show")["cores"], [])

    def test_default_command_and_errors(self) -> None:
        self.assertEqual(self.json()["command"], "show")
        self.assertIn("rank", self.ok("--color", "never").stdout)
        self.fails("lokc", code=2)
        self.fails("lock", "fast", code=2)
        self.assertIn("only for", self.fails("show", "--", "ls", code=2).stderr)

    def test_cpu_columns_are_aligned(self) -> None:
        shutil.rmtree(self.root)
        fake_sysfs.build(self.root, "amd-ryzen9-7950x")
        rows = [line for line in self.ok("show", "--color", "never").stdout.splitlines() if "425" in line]
        self.assertIn(" 236   1 17 ★   425", rows[0])
        self.assertIn(" 201  13 29     425", rows[8])
        self.assertEqual(len({row.index("425") for row in rows}), 1)
        self.assertNotIn(",", "".join(rows))

    def test_watch_reads_changes_and_stays_away(self) -> None:
        self.ok("lock")  # cpu 8,9
        env = {k: v for k, v in os.environ.items() if k != "NO_COLOR"} | {"CPUTUNE_ROOT": str(self.root)}
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT), "watch", "--json", "--interval", "0.2", "--count", "30"],
            env=env, stdout=subprocess.PIPE, text=True,
        )
        self.addCleanup(proc.wait)
        self.addCleanup(proc.stdout.close)
        self.addCleanup(proc.kill)
        first = json.loads(proc.stdout.readline())
        self.assertEqual(next(c for c in first["cores"] if c["cpus"] == [2, 3])["now_mhz"], 3100)
        status = Path(f"/proc/{proc.pid}/status").read_text()
        allowed = next(line for line in status.splitlines() if line.startswith("Cpus_allowed_list"))
        nice = int(Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()[16])
        self.assertEqual(nice, 19)
        if os.cpu_count() >= 10:
            self.assertNotIn(8, fake_sysfs_cpus(allowed.split(":")[1]))
            self.assertNotIn(9, fake_sysfs_cpus(allowed.split(":")[1]))
        # Files that change between updates: the open file descriptors see the new values.
        self.set_f(2, "scaling_cur_freq", 1234000)
        self.set_f(3, "scaling_cur_freq", 1234000)  # now_mhz is the highest value of the siblings
        self.set_f(2, "scaling_max_freq", 2000000)
        for _ in range(8):  # 1.6 s: before the rescan after 5 s
            core = next(c for c in json.loads(proc.stdout.readline())["cores"] if c["cpus"] == [2, 3])
            if core["now_mhz"] == 1234 and core["max_mhz"] == 2000:
                break
        else:
            self.fail("watch did not show the new values")

    def test_watch(self) -> None:
        for args in (("-w",), ("show", "--watch"), ("--interval", "0.1", "-w"), ("watch",)):  # watch: hidden
            with self.subTest(args=args):
                lines = self.ok(*args, "--count", "2", "--json").stdout.splitlines()
                self.assertEqual(len(lines), 2)
                self.assertEqual(json.loads(lines[1])["command"], "watch")
        self.assertIn("Ctrl+C", self.ok("-w", "--count", "1").stdout)

    def test_watch_stops_on_q(self) -> None:
        # In a terminal, q stops the live view before the next update, and the terminal mode is restored.
        import pty
        import select
        import termios

        main, child = pty.openpty()
        self.addCleanup(os.close, main)
        self.addCleanup(os.close, child)
        before = termios.tcgetattr(child)
        env = dict(os.environ, CPUTUNE_ROOT=str(self.root))
        proc = subprocess.Popen([sys.executable, str(SCRIPT), "-w", "--interval", "60"], env=env,
                                stdin=child, stdout=child, stderr=child)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        shown = b""
        while b"q or Ctrl+C to stop" not in shown:
            if not select.select([main], [], [], 10)[0]:
                self.fail(f"watch did not show the first update: {shown!r}")
            shown += os.read(main, 65536)
        os.write(main, b"q")
        self.assertEqual(proc.wait(timeout=5), 0)
        self.assertEqual(termios.tcgetattr(child), before)

    def test_notes_of_the_machine(self) -> None:
        fake_sysfs.w(self.root / "proc/11/comm", "tuned")
        fake_sysfs.w(self.root / "proc/12/comm", "tuned-ppd")
        (self.root / "sys/class/power_supply/AC/online").write_text("0\n")
        text = self.ok("--color", "never").stdout
        self.assertIn("★ best core: lock and run use it", text)
        notes = text[text.index("(all values in MHz)"):].splitlines()[2:]
        self.assertTrue(notes[0].startswith("! The computer runs on battery."), notes)  # problems first
        self.assertIn("· tuned and tuned-ppd run: power services of the system, not started by cputune.", text)
        self.assertIn("  To stop them until the next reboot: sudo systemctl stop tuned tuned-ppd", text)
        self.assertTrue(all(len(line) <= 100 for line in notes), notes)  # wrapped to COLUMNS
        data = self.json()
        self.assertEqual({i["name"]: i["level"] for i in data["checks"]},
                         {"service": "note", "battery": "warn", "aslr": "note"})

    def test_quiet_desktop_has_no_notes(self) -> None:
        shutil.rmtree(self.root)
        fake_sysfs.build(self.root, "amd-ryzen9-7950x")
        (self.root / "proc/sys/kernel/randomize_va_space").write_text("0\n")
        text = self.ok("--color", "never").stdout
        self.assertTrue(text.rstrip().endswith("(all values in MHz)"), text[-200:])

    def test_help_groups_the_main_commands(self) -> None:
        text = self.ok("--help").stdout
        for group in ("look at the machine:", "run a benchmark:", "change settings by hand:", "install and update:"):
            self.assertIn(f"\n{group}\n", text)
        for hidden in ("watch", "check", "update", "sudoers"):  # still work, for scripts
            self.assertNotIn(f"cputune {hidden}", text)
        self.assertIn("cputune install --no-password", text)
        self.assertTrue(self.ok("install", "--help").stdout.startswith("usage: cputune install "))

    def test_help_colors_follow_argparse(self) -> None:
        # Python 3.14 colors its help (FORCE_COLOR: also without a terminal). HELP takes the same colors.
        text = self.run_tool("--help", FORCE_COLOR="1").stdout  # run_tool removes NO_COLOR
        if sys.version_info < (3, 14):
            self.assertNotIn("\033[", text)  # argparse has no colors: HELP neither
            return
        heading, prog, extra = "\033[1;34m", "\033[1;35m", "\033[35m"
        self.assertIn(f"{heading}run a benchmark:\033[0m", text)
        self.assertIn(f"{heading}options:\033[0m", text)  # the same color from argparse itself
        self.assertIn(f"  {prog}cputune\033[0m {extra}run\033[0m \033[36m--strict\033[0m -- ./b   run ./b", text)
        self.assertIn(f"  {prog}cputune\033[0m {extra}lock\033[0m 3GHz \033[32m-c\033[0m 2        lock CPU 2", text)
        self.assertIn("  \033[1;33mFREQ\033[0m ", text)
        self.assertIn("  run executes COMMAND as your user", text)  # plain text stays plain
        self.assertNotIn("\033[", self.run_tool("--help", FORCE_COLOR="1", NO_COLOR="1").stdout)

    def test_help_names_all_options(self) -> None:
        text = self.ok("--help").stdout
        for command in ("show", "run", "lock", "limit", "reset", "install"):
            options = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", self.ok(command, "--help").stdout))
            for option in options:
                self.assertIn(option, text, command)

    def test_lock_default_intel(self) -> None:
        before = self.snapshot()
        data = self.json("lock")
        self.assertEqual(data["cpus"], [8, 9])
        self.assertEqual(data["boost_off"], "global")
        self.assertEqual(data["taskset"], "taskset -c 8,9")
        self.assertEqual((self.cpu / "intel_pstate/no_turbo").read_text().strip(), "1")
        for cpu in (8, 9):
            self.assertEqual(self.f(cpu, "scaling_governor"), "performance")
            self.assertEqual(self.f(cpu, "scaling_min_freq"), "2200000")
            self.assertEqual(self.f(cpu, "scaling_max_freq"), "2200000")
        # Other cores do not change.
        after = self.snapshot()
        changed = {k for k in after if after[k] != before[k]}
        self.assertTrue(all("cpu8/" in k or "cpu9/" in k or "no_turbo" in k for k in changed), changed)
        state = json.loads(self.state.read_text())
        self.assertEqual(state["cpus"]["8"]["max"], 5500000)
        self.assertEqual(state["cpus"]["8"]["governor"], "powersave")
        self.assertEqual(state["boost"], {str(self.cpu / "intel_pstate/no_turbo"): "0"})
        shown = self.json("show")
        self.assertEqual(shown["changed_by_cputune"]["cpus"], [8, 9])
        self.assertTrue(next(c for c in shown["cores"] if c["cpus"] == [8, 9])["locked"])
        self.assertIn("cputune reset", self.ok("show").stdout)

    def test_second_lock_keeps_originals(self) -> None:
        self.ok("lock")
        self.ok("lock", "min", "--cpu", "8")
        self.ok("lock", "--cpu", "2")
        self.assertEqual(self.f(8, "scaling_max_freq"), "800000")
        state = json.loads(self.state.read_text())
        self.assertEqual(state["cpus"]["8"], {"governor": "powersave", "epp": "balance_performance",
                                              "min": 800000, "max": 5500000})
        self.assertEqual(state["cpus"]["2"]["max"], 5300000)
        self.assertEqual(state["boost"][str(self.cpu / "intel_pstate/no_turbo")], "0")

    def test_frequency_values(self) -> None:
        for value, khz in (("1.5GHz", "1500000"), ("1200MHz", "1200000"), ("2000", "2000000"),
                           ("min", "800000"), ("max", "5500000"), ("base", "2200000")):
            with self.subTest(value=value):
                self.ok("lock", value, "--cpu", "8")
                self.assertEqual(self.f(8, "scaling_min_freq"), khz)
                self.assertEqual(self.f(8, "scaling_max_freq"), khz)

    def test_bad_frequency_changes_nothing(self) -> None:
        before = self.snapshot()
        self.assertIn("outside 800-5500", self.fails("lock", "9999").stderr)
        self.assertIn("above the base frequency 2200", self.fails("lock", "3000").stderr)
        self.assertIn("Did you mean 2GHz?", self.fails("lock", "2").stderr)
        self.assertIn('bad frequency "2.4". Did you mean 2.4GHz?', self.fails("lock", "2.4", code=2).stderr)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.state.exists())

    def test_taskset_names_the_requested_cpus(self) -> None:
        data = self.json("lock", "-c", "2")
        self.assertEqual((data["cpus"], data["pin_cpus"], data["taskset"]), ([2, 3], [2], "taskset -c 2"))
        data = self.json("lock")
        self.assertEqual((data["cpus"], data["pin_cpus"], data["taskset"]), ([8, 9], [8, 9], "taskset -c 8,9"))
        self.assertIn("taskset -c 2 ./bench", self.ok("lock", "--cpu", "2").stdout)

    def test_cpu_lists_include_siblings(self) -> None:
        self.assertEqual(self.json("lock", "--cpu", "0")["cpus"], [0, 1])
        self.assertEqual(self.json("lock", "--cpu", "16")["cpus"], [16])
        self.assertEqual(self.json("lock", "--cpu", "3,16-17")["cpus"], [2, 3, 16, 17])
        self.assertIn("bad CPU list", self.fails("lock", "--cpu", "2-x").stderr)
        self.assertIn("bad CPU list", self.fails("lock", "--cpu", "5-3").stderr)
        self.assertIn("cpu99 has no cpufreq", self.fails("lock", "--cpu", "99").stderr)
        self.assertIn("cpu32 has no cpufreq", self.fails("lock", "--cpu", "32").stderr)  # offline

    def test_exclusive(self) -> None:
        self.ok("lock")
        self.assertIn("cputune reset", self.fails("lock", "--exclusive", "--cpu", "2").stderr)
        self.assertNotIn("2", json.loads(self.state.read_text())["cpus"])

    def test_json_error(self) -> None:
        result = self.fails("--json", "lock", "--cpu", "99")
        data = json.loads(result.stdout)
        self.assertFalse(data["ok"])
        self.assertEqual(data["command"], "lock")
        self.assertIn("cpu99", data["error"])

    def test_failed_read_leaves_no_state_file(self) -> None:
        (self.cpu / "cpu9/cpufreq/scaling_max_freq").unlink()
        self.assertIn("cannot read", self.fails("lock", "--cpu", "8").stderr)
        self.assertFalse(self.state.exists())
        self.assertEqual((self.cpu / "intel_pstate/no_turbo").read_text().strip(), "0")

    def test_measured_mismatch_warns(self) -> None:
        # The fake tree reports 3100 MHz: faster than the lock, so boost is still on.
        data = self.json("lock")
        self.assertFalse(data["cores"][0]["ok"])
        self.assertIn("measured 3100 MHz under load, faster than the lock 2200 MHz", data["warnings"][0])
        self.set_f(8, "scaling_cur_freq", 2200000)
        self.assertTrue(self.json("lock")["cores"][0]["ok"])

    def test_below_the_lock_is_stable(self) -> None:
        # The CPU firmware can hold a lock a little below the target (Ryzen 9 7950X: 2729 MHz for 3000).
        self.set_f(8, "scaling_cur_freq", 2000000)
        data = self.json("lock")
        self.assertTrue(data["cores"][0]["ok"])
        self.assertNotIn("warnings", data)
        text = self.ok("lock").stdout
        self.assertIn("measured under load: 2000 MHz ✓ (9% below the lock: the CPU firmware sets this", text)
        self.set_f(8, "scaling_cur_freq", 2180000)  # within 3%: no note
        self.assertNotIn("below the lock", self.ok("lock").stdout)


class LockAmdTest(Case):
    machine = "amd-ryzen9-7950x"

    def test_per_core_boost(self) -> None:
        data = self.json("lock")
        self.assertEqual(data["cpus"], [1, 17])
        self.assertEqual(data["boost_off"], "core")
        self.assertEqual(data["cores"][0]["target_mhz"], 4501)  # acpi_cppc/nominal_freq
        for cpu in range(32):
            self.assertEqual(self.f(cpu, "boost"), "0" if cpu in (1, 17) else "1", cpu)
        self.assertEqual((self.cpu / "cpufreq/boost").read_text().strip(), "1")
        self.assertEqual(self.json("show")["system"]["boost_off_cpus"], [1, 17])

    def test_min_is_the_lowest_frequency_the_cpu_follows(self) -> None:
        # amd-pstate: below lowest_nonlinear_freq the CPU does not follow (425 MHz requested, 2170 measured).
        data = self.json("lock", "min", "--cpu", "2")
        self.assertEqual(data["cores"][0]["target_mhz"], 3012)
        self.assertEqual((self.f(2, "scaling_min_freq"), self.f(2, "scaling_max_freq")), ("3012480", "3012480"))
        self.ok("reset")
        self.ok("limit", "--max", "min")  # limit keeps the hardware minimum
        self.assertEqual(self.f(2, "scaling_max_freq"), "425000")


class LockAcpiTest(Case):
    machine = "amd-epyc-9354p"

    def test_global_boost_and_base(self) -> None:
        data = self.json("lock")
        self.assertEqual(data["cpus"], [1, 33])
        self.assertEqual(data["boost_off"], "global")
        self.assertEqual((self.cpu / "cpufreq/boost").read_text().strip(), "0")
        self.assertEqual(self.f(1, "scaling_max_freq"), "3250000")

    def test_base_from_available_frequencies(self) -> None:
        for path in self.cpu.glob("cpu*/acpi_cppc/nominal_freq"):
            path.unlink()
        self.set_f(2, "scaling_available_frequencies", "3100000 2200000 1500000")
        self.set_f(34, "scaling_available_frequencies", "3100000 2200000 1500000")
        self.assertEqual(self.json("lock", "--cpu", "2")["cores"][0]["target_mhz"], 3100)

    def test_top_pstate_is_no_limit(self) -> None:
        # Kernel 7.1 keeps scaling_max_freq at the highest P-state (3250, as in the fake tree), below
        # cpuinfo_max_freq (3800), and the cores still boost to 3800 MHz: the full range, not a limit.
        self.assertEqual(self.f(1, "scaling_max_freq"), "3250000")
        out = self.ok("show").stdout
        rows = [line for line in out.splitlines() if line.startswith(" 255")]
        self.assertEqual(len(rows), 32)
        self.assertTrue(all("╍" in row and "─" not in row for row in rows), rows[0])  # boost, no dark part
        self.assertNotIn("set MHz", out)
        self.assertNotIn("outside the set range", out)
        self.ok("limit", "--max", "2400")  # a real limit still shows
        out = self.ok("show").stdout
        self.assertIn("1500-2400", out)
        self.assertIn("outside the set range", out)


class LockRaspberryTest(Case):
    machine = "rpi4"

    def test_no_boost_switch(self) -> None:
        data = self.json("lock")
        self.assertEqual(data["cpus"], [1])
        self.assertEqual(data["boost_off"], "none")
        self.assertIn("no boost switch", data["warnings"][0])
        self.assertEqual(self.f(1, "scaling_max_freq"), "1500000")  # no base: highest frequency


class LimitTest(Case):
    def test_limit_clamps(self) -> None:
        data = self.json("limit", "--max", "0", "--cpu", "3")
        self.assertEqual(data["cpus"], [2, 3])
        self.assertEqual((self.f(3, "scaling_min_freq"), self.f(3, "scaling_max_freq")), ("800000", "800000"))
        self.ok("limit", "--min", "99999")
        self.assertEqual((self.f(0, "scaling_min_freq"), self.f(0, "scaling_max_freq")), ("5300000", "5300000"))
        self.assertEqual((self.f(16, "scaling_min_freq"), self.f(16, "scaling_max_freq")), ("4000000", "4000000"))

    def test_max_wins_over_min(self) -> None:
        self.ok("limit", "--min", "3000", "--max", "2000")
        self.assertEqual((self.f(5, "scaling_min_freq"), self.f(5, "scaling_max_freq")), ("2000000", "2000000"))

    def test_words_and_boost_untouched(self) -> None:
        self.ok("limit", "--max", "base")
        self.assertEqual(self.f(0, "scaling_max_freq"), "2200000")
        self.assertEqual(self.f(16, "scaling_max_freq"), "1600000")
        self.assertEqual(self.f(0, "scaling_governor"), "powersave")
        self.assertEqual((self.cpu / "intel_pstate/no_turbo").read_text().strip(), "0")
        self.assertIn("set MHz", self.ok("show").stdout)

    def test_needs_a_value(self) -> None:
        self.assertIn("give --min, --max", self.fails("limit").stderr)


class ResetTest(Case):
    def test_restores_everything(self) -> None:
        before = self.snapshot()
        self.ok("lock")
        self.ok("lock", "1GHz", "--cpu", "2")
        self.ok("limit", "--max", "1200")
        self.set_f(8, "energy_performance_preference", "performance")  # as the driver does
        self.assertNotEqual(self.snapshot(), before)
        data = self.json("reset")
        self.assertTrue(data["restored"])
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.state.exists())
        self.assertIsNone(self.json("show")["changed_by_cputune"])

    def test_restores_per_core_boost(self) -> None:
        shutil.rmtree(self.root)
        fake_sysfs.build(self.root, "amd-ryzen9-7950x")
        before = self.snapshot()
        self.ok("lock", "--cpu", "3")
        self.ok("reset")
        self.assertEqual(self.snapshot(), before)

    def test_without_state_opens_full_range(self) -> None:
        self.set_f(4, "scaling_max_freq", 1000000)
        self.set_f(4, "scaling_min_freq", 1000000)
        data = self.json("reset")
        self.assertFalse(data["restored"])
        self.assertEqual((self.f(4, "scaling_min_freq"), self.f(4, "scaling_max_freq")), ("800000", "5300000"))

    def test_skips_paths_that_are_not_boost_switches(self) -> None:
        self.ok("lock")
        state = json.loads(self.state.read_text())
        target = self.root / "proc/sys/kernel/osrelease"
        state["boost"][str(target)] = "hacked"
        self.state.write_text(json.dumps(state))
        data = self.json("reset")
        self.assertNotEqual(target.read_text().strip(), "hacked")
        self.assertIn("not a boost switch", data["warnings"][0])
        self.assertEqual((self.cpu / "intel_pstate/no_turbo").read_text().strip(), "0")

    def test_failed_write_keeps_state(self) -> None:
        self.ok("lock")
        governor = self.cpu / "cpu8/cpufreq/scaling_governor"
        governor.unlink()
        governor.mkdir()  # every write to it fails now
        self.assertIn("stays, so you can try again", self.fails("reset").stderr)
        self.assertTrue(self.state.exists())
        self.assertEqual(self.f(9, "scaling_governor"), "powersave")  # the other CPUs are restored

    def test_damaged_state(self) -> None:
        self.state.write_text("{not json")
        self.assertIn("sudo rm /run/cputune.json", self.fails("reset").stderr)
        # Valid JSON, but not a state: an error message, not a traceback.
        for text in ("[]", '{"cpus": {"x": {}}, "boost": {}}', '{"cpus": {"8": {"min": "1"}}, "boost": {}}'):
            with self.subTest(state=text):
                self.state.write_text(text)
                result = self.fails("reset")
                self.assertIn("/run/cputune.json is damaged", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertIn("cputune changed cpu ?", self.ok("show").stdout)

    def test_state_from_an_earlier_boot_is_ignored(self) -> None:
        self.ok("lock")
        self.assertEqual(json.loads(self.state.read_text())["boot_id"], fake_sysfs.BOOT_ID)
        self.ok("reset")
        # A reboot reset all settings. The saved values of the earlier boot must not come back.
        stale = {"boot_id": "an-earlier-boot", "cpus": {"8": {"governor": "performance", "epp": None,
                                                              "min": 1000000, "max": 1000000}}, "boost": {}}
        self.state.write_text(json.dumps(stale))
        self.assertIsNone(self.json("show")["changed_by_cputune"])
        data = self.json("lock", "--exclusive")  # not refused: nothing is changed in this boot
        state = json.loads(self.state.read_text())
        self.assertEqual((state["boot_id"], state["cpus"]["8"]["max"]), (fake_sysfs.BOOT_ID, 5500000))
        self.assertEqual(data["cpus"], [8, 9])
        self.ok("reset")
        self.assertEqual(self.f(8, "scaling_max_freq"), "5500000")
        self.state.write_text(json.dumps(stale))
        self.assertFalse(self.json("reset")["restored"])
        self.assertFalse(self.state.exists())  # root deletes the stale file
        self.assertEqual(self.f(8, "scaling_max_freq"), "5500000")

    def test_state_without_boot_id_is_used(self) -> None:
        # cputune 2.2.0 and older wrote no boot ID: such a file is from this boot.
        self.ok("lock")
        state = json.loads(self.state.read_text())
        del state["boot_id"]
        self.state.write_text(json.dumps(state))
        self.assertEqual(self.json("show")["changed_by_cputune"]["cpus"], [8, 9])
        self.assertTrue(self.json("reset")["restored"])


class RunTest(Case):
    def test_runs_pinned_and_resets(self) -> None:
        before = self.snapshot()
        code = "import os; print(os.environ['CPUTUNE_CPUS'])"
        result = self.ok("run", "--cpu", "4", "--", sys.executable, "-c", code)
        self.assertEqual(result.stdout, "4\n")  # only the named CPU; cputune messages go to stderr
        self.assertIn("locked cpu 4,5 at 2200 MHz", result.stderr)  # the whole core
        self.assertIn("running on cpu 4:", result.stderr)
        self.assertIn("reset done", result.stderr)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.state.exists())

    def test_pins_to_the_named_cpus(self) -> None:
        code = "import os; print(os.environ['CPUTUNE_CPUS'])"
        self.assertEqual(self.ok("run", "--", sys.executable, "-c", code).stdout, "8,9\n")  # default: whole core
        self.assertEqual(self.ok("run", "-c", "2", "--", sys.executable, "-c", code).stdout, "2\n")
        self.assertEqual(self.ok("run", "-c", "2,16", "--", sys.executable, "-c", code).stdout, "2,16\n")
        self.assertEqual(self.ok("run", "-c", "all", "--", sys.executable, "-c", code).stdout, "0-31\n")
        if len(os.sched_getaffinity(0)) > 3:  # the real pinning, where the test machine has these CPUs
            code = "import os; print(sorted(os.sched_getaffinity(0)))"
            self.assertEqual(self.ok("run", "-c", "3", "--", sys.executable, "-c", code).stdout, "[3]\n")

    def test_exit_code_and_json(self) -> None:
        result = self.run_tool("run", "1500", "--json", "--", sys.executable, "-c", "raise SystemExit(7)")
        self.assertEqual(result.returncode, 7)
        data = json.loads(result.stderr[result.stderr.index("{"):])
        self.assertEqual(data["exit_code"], 7)
        self.assertEqual(data["cores"][0]["target_mhz"], 1500)
        self.assertFalse(self.state.exists())

    def test_refuses_when_already_changed(self) -> None:
        self.ok("lock")
        marker = self.root / "ran"
        result = self.fails("run", "--", sys.executable, "-c", f"open({str(marker)!r}, 'w')")
        self.assertIn("cputune reset", result.stderr)
        self.assertFalse(marker.exists())
        self.assertTrue(self.state.exists())  # the earlier lock stays

    def test_bad_lock_runs_nothing(self) -> None:
        marker = self.root / "ran"
        self.fails("run", "9999", "--", sys.executable, "-c", f"open({str(marker)!r}, 'w')")
        self.assertFalse(marker.exists())
        self.assertFalse(self.state.exists())

    def test_lock_that_fails_halfway_says_how_to_undo(self) -> None:
        governor = self.cpu / "cpu8/cpufreq/scaling_governor"
        governor.unlink()
        governor.mkdir()  # lock fails after boost is off
        result = self.fails("lock")
        self.assertIn("scaling_governor", result.stderr)
        self.assertIn("Settings are partly changed. Undo with: cputune reset", result.stderr)

    def test_interrupt_during_the_lock_resets(self) -> None:
        with mock.patch.dict(os.environ, CPUTUNE_ROOT=str(self.root)):
            m = load_module()
        for existed in (False, True):
            with self.subTest(existed=existed):
                calls = []
                self.state.unlink(missing_ok=True)
                if existed:  # an earlier lock: run must not undo it
                    self.state.write_text(json.dumps({"cpus": {}, "boost": {}}))

                def run_root(argv, capture=False, calls=calls):
                    calls.append(argv[0])
                    if argv[0] == "lock":  # lock saved its state, then Ctrl+C
                        self.state.write_text(json.dumps({"cpus": {}, "boost": {}}))
                        raise KeyboardInterrupt
                    return subprocess.CompletedProcess(argv, 0, "{}")

                args = types.SimpleNamespace(command=["true"], cpu=None, freq=None, strict=False, json=False)
                with mock.patch.multiple(m, run_root=run_root, run_checks=lambda *a: []), \
                        mock.patch("signal.signal"), self.assertRaises(KeyboardInterrupt):
                    m.cmd_run(args)
                self.assertEqual(calls, ["lock"] if existed else ["lock", "reset"])

    def test_cannot_measure(self) -> None:
        for cpu in (8, 9):
            (self.cpu / f"cpu{cpu}/cpufreq/scaling_cur_freq").unlink()
        data = self.run_json("--", "true")
        self.assertEqual(data["warnings"], ["cpu8: cannot measure the frequency under load"])  # from lock
        self.assertFalse(data["stable"])
        error = self.fails("run", "--strict", "--", "true").stderr
        self.assertIn("the lock is not verified: cpu 8,9: cannot measure the frequency under load.", error)
        self.assertNotIn("faster", error)

    def test_lock_that_fails_halfway_is_reset(self) -> None:
        governor = self.cpu / "cpu8/cpufreq/scaling_governor"
        governor.unlink()
        governor.mkdir()  # lock fails after boost is off
        self.assertIn("lock failed", self.fails("run", "--", "true").stderr)
        self.assertEqual((self.cpu / "intel_pstate/no_turbo").read_text().strip(), "0")

    def test_needs_a_command(self) -> None:
        self.assertIn("give the command after --", self.fails("run").stderr)
        self.assertIn("cannot run", self.fails("run", "--", "/does/not/exist").stderr)
        self.assertFalse(self.state.exists())


    def run_json(self, *args: str, code: int = 0) -> dict:
        result = self.run_tool("run", "--json", *args)
        self.assertEqual(result.returncode, code, result.stderr)
        self.assertNotIn("cputune: locked cpu", result.stderr)  # --json: no text for people
        return json.loads(result.stderr[result.stderr.index("{"):])  # stdout belongs to the command

    def test_measures_while_the_command_runs(self) -> None:
        for cpu in (8, 9):
            self.set_f(cpu, "scaling_cur_freq", 2200000)  # the frequency of the lock
        data = self.run_json("--", sys.executable, "-c", "pass")
        self.assertTrue(data["stable"])
        self.assertNotIn("warnings", data)
        core = data["cores"][0]
        self.assertEqual((core["run_avg_mhz"], core["run_min_mhz"], core["throttled"]), (2200, 2200, 0))
        result = self.ok("run", "--", "true")
        self.assertIn("cpu 8,9 ran at 2200 MHz (lowest 2200 MHz), not throttled", result.stderr)
        self.assertNotIn("may vary", result.stderr)

    def test_finds_problems_while_the_command_runs(self) -> None:
        for cpu in (8, 9):
            self.set_f(cpu, "scaling_cur_freq", 2200000)
        # The command plays the kernel and another program: the CPU gets too hot, and boost goes on.
        code = (f"from pathlib import Path; d = Path({str(self.cpu)!r})\n"
                "(d / 'cpu8/thermal_throttle/core_throttle_count').write_text('3')\n"
                "for c in (8, 9): (d / f'cpu{c}/cpufreq/scaling_cur_freq').write_text('1500000')\n"
                "(d / 'intel_pstate/no_turbo').write_text('0')\n")
        data = self.run_json("--", sys.executable, "-c", code)
        self.assertFalse(data["stable"])
        core = data["cores"][0]
        self.assertEqual((core["run_min_mhz"], core["throttled"]), (1500, 3))
        warnings = "\n".join(data["warnings"])
        self.assertIn("cpu 8,9 was too hot and ran slower 3 times", warnings)
        self.assertIn("but at 2200 MHz after the lock (too hot, or at its power limit)", warnings)
        self.assertIn(". Check with: cputune\n", warnings + "\n")  # not the hidden "cputune check"
        self.assertIn(f"another program changed the lock while the command ran: {self.cpu}/intel_pstate/no_turbo",
                      warnings)
        self.assertFalse(self.state.exists())  # reset

    def test_warns_about_what_check_finds(self) -> None:
        fake_sysfs.w(self.root / "proc/4321/comm", "power-profiles-")  # a note only: run sees real changes
        for cpu in (8, 9):
            self.set_f(cpu, "scaling_cur_freq", 2200000)
        self.assertTrue(self.run_json("--", "true")["stable"])
        (self.root / "sys/class/power_supply/AC/online").write_text("0\n")
        data = self.run_json("--", "true")
        self.assertFalse(data["stable"])
        self.assertEqual(data["warnings"], ["the computer runs on battery. Then the CPU often has lower power "
                                            "limits. Connect the power supply."])

    def test_compares_with_the_measured_lock(self) -> None:
        for cpu in (8, 9):
            self.set_f(cpu, "scaling_cur_freq", 2000000)  # the firmware holds the lock below 2200 MHz
        data = self.run_json("--", "true")
        self.assertTrue(data["stable"])
        self.assertEqual(data["cores"][0]["run_avg_mhz"], 2000)
        cur = self.cpu / "cpu8/cpufreq/scaling_cur_freq"
        code = f"from pathlib import Path; Path({str(cur)!r}).write_text('1800000')"
        data = self.run_json("--", sys.executable, "-c", code)
        self.assertFalse(data["stable"])
        self.assertIn("ran at 1900 MHz on average while the command ran, but at 2000 MHz after the lock",
                      data["warnings"][0])

    def test_strict(self) -> None:
        before = self.snapshot()
        marker = self.root / "ran"
        command = ("--", sys.executable, "-c", f"open({str(marker)!r}, 'w')")
        result = self.fails("run", "--strict", *command)  # the fake tree measures 3100 MHz, the lock is 2200
        self.assertIn("the lock is not verified: cpu 8,9: measured 3100 MHz under load", result.stderr)
        self.assertIn("With --strict, the command does not run.", result.stderr)
        self.assertFalse(marker.exists())
        self.assertFalse(self.state.exists())
        self.assertEqual(self.snapshot(), before)
        for cpu in (8, 9):
            self.set_f(cpu, "scaling_cur_freq", 2200000)
        self.ok("run", "--strict", *command)
        self.assertTrue(marker.exists())


class CheckTest(Case):
    def levels(self, data: dict) -> dict[str, str]:
        return {item["name"]: item["level"] for item in data["checks"]}

    def test_quiet_machine(self) -> None:
        data = self.json("check")
        self.assertEqual(data["cpus"], [8, 9])  # the core that run uses
        self.assertTrue(data["clean"])
        self.assertEqual(self.levels(data), {"service": "ok", "battery": "ok", "aslr": "note"})
        text = self.ok("check").stdout
        self.assertIn("What can make results vary on cpu 8,9 (the core that lock and run use):", text)
        self.assertIn("Turn it off for one command: setarch -R COMMAND", text)
        self.assertIn("Nothing found that makes results vary.", text)

    def test_finds_problems(self) -> None:
        for pid, comm in ((10, "power-profiles-"), (11, "tuned"), (12, "bash")):
            fake_sysfs.w(self.root / f"proc/{pid}/comm", comm)
        (self.root / "sys/class/power_supply/AC/online").write_text("0\n")
        self.ok("lock", "--cpu", "2")
        before = self.snapshot()
        data = self.json("check", "--cpu", "0")
        self.assertEqual(self.snapshot(), before)  # check changes nothing
        self.assertEqual(data["cpus"], [0, 1])
        self.assertFalse(data["clean"])
        warnings = [item["fix"] for item in data["checks"] if item["level"] == "warn"]
        self.assertEqual(warnings, ["cputune reset", "Connect the power supply."])
        notes = {i["name"]: i for i in data["checks"] if i["level"] == "note"}
        self.assertEqual(list(notes), ["service", "aslr", "cpu 0"])
        self.assertIn("power-profiles-daemon and tuned run: power services of the system, not started by "
                      "cputune.", notes["service"]["text"])
        self.assertEqual(notes["service"]["fix"],
                         "To stop them until the next reboot: sudo systemctl stop power-profiles-daemon tuned")
        self.assertIn("2 problems can make results vary.", self.ok("check", "--cpu", "0").stdout)

    def test_desktop_without_aslr(self) -> None:
        shutil.rmtree(self.root)
        fake_sysfs.build(self.root, "amd-ryzen9-7950x")
        (self.root / "proc/sys/kernel/randomize_va_space").write_text("0\n")
        data = self.json("check")
        self.assertEqual(self.levels(data), {"service": "ok", "aslr": "ok"})  # no power supply in sysfs
        self.assertTrue(data["clean"])

    def test_load(self) -> None:
        with mock.patch.dict(os.environ, CPUTUNE_ROOT=str(self.root)):
            m = load_module()
        # CPU 8 is 50% busy, the machine 58%.
        times = iter([{0: (0, 100), 2: (0, 100), 8: (0, 100), 9: (0, 100)},
                      {0: (90, 200), 2: (90, 200), 8: (50, 200), 9: (1, 200)}])
        with mock.patch.object(m, "cpu_times", lambda: next(times)), mock.patch("os.sched_setaffinity"):
            items = {item["name"]: item for item in m.run_checks([8, 9])}
        self.assertEqual(items["load"]["level"], "warn")
        self.assertIn("cpu 8 50% busy", items["load"]["text"])
        self.assertEqual(items["machine load"]["level"], "warn")
        self.assertIn("The machine is 58% busy", items["machine load"]["text"])
        times = iter([{8: (0, 100), 9: (0, 100)}, {8: (5, 200), 9: (0, 200)}])
        with mock.patch.object(m, "cpu_times", lambda: next(times)), mock.patch("os.sched_setaffinity"):
            items = {item["name"]: item for item in m.run_checks([8, 9])}
        self.assertEqual(items["load"], {"name": "load", "level": "ok", "text": "No other load on cpu 8,9.",
                                         "fix": None})
        self.assertNotIn("machine load", items)

    def test_reads_the_real_proc_stat(self) -> None:
        m = load_module()
        if Path("/proc/stat").exists():
            self.assertTrue(m.cpu_times())

    def test_bad_cpu(self) -> None:
        self.assertIn("cpu99 has no cpufreq", self.fails("check", "--cpu", "99").stderr)


def with_version(version: str) -> bytes:
    """The cputune script with another __version__."""
    source = SCRIPT.read_bytes()
    current = re.search(rb'^__version__ = "[^"]+"', source, re.M)[0]
    return source.replace(current, f'__version__ = "{version}"'.encode(), 1)


class InstallTest(Case):
    def setUp(self) -> None:
        super().setUp()
        self.remote = self.root / "remote/cputune"  # stands in for GitHub
        self.remote.parent.mkdir()
        self.url = self.remote.as_uri()
        self.copy = self.root / "bin/cputune"  # an installed copy, not in a git checkout
        self.copy.parent.mkdir()
        self.copy.write_bytes(SCRIPT.read_bytes())

    def install(self, *args: str) -> dict:
        result = self.run_tool("install", "--json", *args, CPUTUNE_SOURCE_URL=self.url)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def update(self, script: Path | None = None) -> subprocess.CompletedProcess:
        return self.run_tool("update", "--json", script=script or self.copy, CPUTUNE_SOURCE_URL=self.url)

    def test_install_takes_the_newer_version(self) -> None:
        target = self.root / "home/.local/bin/cputune"
        self.remote.write_bytes(with_version("0.0.1"))
        self.assertEqual(self.install()["source"], "this file")
        self.assertEqual(target.read_bytes(), SCRIPT.read_bytes())
        self.assertEqual(target.stat().st_mode & 0o777, 0o755)
        self.remote.write_bytes(with_version("99.0.0"))
        data = self.install()
        self.assertEqual((data["source"], data["version"]), ("GitHub", "99.0.0"))
        self.assertEqual(target.read_bytes(), with_version("99.0.0"))
        self.assertEqual(self.install("--no-check")["source"], "this file")
        self.assertEqual(target.read_bytes(), SCRIPT.read_bytes())

    def test_warns_about_copies_with_the_old_name(self) -> None:
        self.remote.write_bytes(with_version("0.0.1"))
        self.assertNotIn("warnings", self.install())
        old = self.root / "usr/local/bin/cpu-tune"
        old.parent.mkdir(parents=True)
        old.write_text("old")
        for data in (self.install(), self.json("sudoers")):
            self.assertIn(f"sudo rm -f {old} /etc/sudoers.d/cpu-tune", data["warnings"][0])

    def test_only_one_copy_stays_installed(self) -> None:
        user_copy = self.root / "home/.local/bin/cputune"
        system_copy = self.root / "usr/local/bin/cputune"
        # A copy in /usr/local/bin: a plain install updates it, and makes no second copy.
        system_copy.parent.mkdir(parents=True)
        system_copy.write_bytes(with_version("0.0.1"))
        data = self.install()
        self.assertEqual(data["path"], str(system_copy))
        self.assertEqual(system_copy.read_bytes(), SCRIPT.read_bytes())
        self.assertFalse(user_copy.exists())
        self.assertIn("updated that copy", data["notes"][0])
        # A copy in ~/.local/bin hides the system copy: install --system deletes it.
        user_copy.parent.mkdir(parents=True)
        user_copy.write_bytes(with_version("0.0.1"))
        data = self.install("--system")
        self.assertFalse(user_copy.exists())
        self.assertIn(f"Deleted {user_copy}", data["notes"][0])
        # A symlink (for example to a checkout) goes, the file it points to stays.
        user_copy.symlink_to(self.copy)
        self.install("--system")
        self.assertFalse(user_copy.is_symlink())
        self.assertTrue(self.copy.exists())

    def test_keeps_a_file_that_is_not_cputune(self) -> None:
        user_copy = self.root / "home/.local/bin/cputune"
        user_copy.parent.mkdir(parents=True)
        user_copy.write_text("#!/bin/sh\necho something else\n")
        data = self.install("--system")
        self.assertTrue(user_copy.exists())
        self.assertIn("is not a cputune script, so it stays", data["warnings"][-1])

    def test_install_system_and_offline(self) -> None:
        data = self.install("--system")  # no remote file: the download fails
        self.assertEqual(data["path"], str(self.root / "usr/local/bin/cputune"))
        self.assertIn("cannot download", data["warnings"][0])
        self.assertEqual((self.root / "usr/local/bin/cputune").read_bytes(), SCRIPT.read_bytes())

    def test_install_from_stdin_downloads(self) -> None:
        self.remote.write_bytes(with_version("99.0.0"))
        env = {k: v for k, v in os.environ.items()} | {
            "CPUTUNE_ROOT": str(self.root), "CPUTUNE_SOURCE_URL": self.url}
        result = subprocess.run([sys.executable, "-", "install"], input=SCRIPT.read_bytes(), env=env,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "home/.local/bin/cputune").read_bytes(), with_version("99.0.0"))

    def test_update(self) -> None:
        self.remote.write_bytes(SCRIPT.read_bytes())
        data = json.loads(self.update().stdout)
        self.assertFalse(data["updated"])
        self.remote.write_bytes(with_version("99.0.0"))
        data = json.loads(self.update().stdout)
        self.assertTrue(data["updated"])
        self.assertEqual(data["new_version"], "99.0.0")
        self.assertEqual(self.copy.read_bytes(), with_version("99.0.0"))
        self.assertEqual(self.copy.stat().st_mode & 0o777, 0o755)
        self.remote.write_bytes(with_version("98.0.0"))  # never downgrade
        self.assertFalse(json.loads(self.update().stdout)["updated"])
        self.assertEqual(self.copy.read_bytes(), with_version("99.0.0"))

    def test_update_through_symlink(self) -> None:
        link = self.root / "link/cputune"
        link.parent.mkdir()
        link.symlink_to(self.copy)
        self.remote.write_bytes(with_version("99.0.0"))
        self.assertEqual(self.update(link).returncode, 0)
        self.assertTrue(link.is_symlink())
        self.assertEqual(self.copy.read_bytes(), with_version("99.0.0"))

    def test_update_refuses_bad_downloads(self) -> None:
        for name, content in (("html", b"<html>404</html>"),
                              ("syntax", with_version("99.0.0") + b"\ndef broken(:\n")):
            with self.subTest(name):
                self.remote.write_bytes(content)
                result = self.update()
                self.assertEqual(result.returncode, 1)
                self.assertIn("not a cputune script" if name == "html" else "does not compile",
                              json.loads(result.stdout)["error"])
                self.assertEqual(self.copy.read_bytes(), SCRIPT.read_bytes())
        self.assertEqual(sorted(p.name for p in self.copy.parent.iterdir()), ["cputune"])  # no temp files

    def test_update_refuses_git_checkout(self) -> None:
        self.remote.write_bytes(with_version("99.0.0"))
        for kind in ("dir", "file"):  # a clone has a .git folder, a worktree a .git file
            with self.subTest(kind):
                checkout = self.root / f"checkout-{kind}"
                (checkout / "sub").mkdir(parents=True)
                (checkout / ".git").mkdir() if kind == "dir" else (checkout / ".git").write_text("gitdir: x\n")
                script = checkout / "sub/cputune"
                script.write_bytes(SCRIPT.read_bytes())
                result = self.update(script)
                self.assertEqual(result.returncode, 1)
                self.assertIn("git pull", json.loads(result.stdout)["error"])
                self.assertEqual(script.read_bytes(), SCRIPT.read_bytes())

    def test_downloads_the_latest_release(self) -> None:
        # Not raw.githubusercontent.com/.../main: it is cached for 5 minutes, so "update" right
        # after a release found the old version.
        m = load_module()
        self.assertEqual(m.SOURCE_URL, "https://github.com/martinus/cputune/releases/latest/download/cputune")
        readme = (SCRIPT.parent / "README.md").read_text()
        self.assertIn(f"curl -fsSL {m.SOURCE_URL} | python3 - install", readme)

    def test_version(self) -> None:
        version = re.search(r'^__version__ = "([^"]+)"', SCRIPT.read_text(), re.M)[1]
        self.assertEqual(self.ok("--version").stdout.strip(), f"cputune {version}")
        self.assertEqual(self.json("show")["version"], version)


def load_module():
    """cputune as a module, to test functions that need states the tests cannot create.

    The module reads the environment when it loads: patch it before the call.
    """
    loader = importlib.machinery.SourceFileLoader("cputune", str(SCRIPT))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("cputune", loader))
    sys.modules["cputune"] = module  # @dataclass looks the module up here
    loader.exec_module(module)
    return module


class SudoersTest(Case):
    def setUp(self) -> None:
        super().setUp()
        # A copy that root does not own, also when the tests run as root in a checkout owned by root.
        self.script = self.root / "checkout/cputune"
        self.script.parent.mkdir()
        self.script.parent.chmod(0o777)
        shutil.copy(SCRIPT, self.script)

    def run_tool(self, *args: str, script: Path | None = None, **kwargs):
        return super().run_tool(*args, script=script or self.script, **kwargs)

    def test_guide_when_nothing_is_set_up(self) -> None:
        # This copy is not owned by root, so the rule must name the system copy.
        system_copy = self.root / "usr/local/bin/cputune"
        data = self.json("sudoers")
        self.assertEqual(data["path"], str(system_copy))
        self.assertFalse(data["installed"])
        self.assertIsNone(data["works"])  # sudo is not run in tests
        self.assertTrue(data["rule"].endswith(f"ALL=(root) NOPASSWD: {system_copy}"))
        self.assertTrue(data["setup"].endswith("install --no-password"))
        text = self.ok("sudoers").stdout
        self.assertIn("Set it up with one command:", text)
        self.assertIn(f"install the newest cputune to {system_copy}, owned by root", text)
        self.assertIn("Undo: sudo rm /etc/sudoers.d/cputune", text)

    def test_install(self) -> None:
        for args in (("install", "--no-password"), ("sudoers", "--install")):  # sudoers: hidden, for scripts
            with self.subTest(args=args):
                shutil.rmtree(self.root / "etc", ignore_errors=True)
                shutil.rmtree(self.root / "usr", ignore_errors=True)
                self.check_install(*args)

    def test_install_no_password_no_check_stays_offline(self) -> None:
        offline = (self.root / "missing").as_uri()
        result = self.run_tool("install", "--no-password", "--no-check", "--json", CPUTUNE_SOURCE_URL=offline)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("cannot download", result.stderr)  # --no-check: GitHub is not asked
        self.assertEqual((self.root / "usr/local/bin/cputune").read_bytes(), SCRIPT.read_bytes())

    def check_install(self, *args: str) -> None:
        offline = (self.root / "missing").as_uri()  # GitHub is not used in tests
        result = self.run_tool(*args, "--json", CPUTUNE_SOURCE_URL=offline)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["command"], args[0])
        self.assertEqual(len(data["done"]), 2)
        system_copy = self.root / "usr/local/bin/cputune"
        self.assertEqual(system_copy.read_bytes(), SCRIPT.read_bytes())
        self.assertEqual(system_copy.stat().st_mode & 0o777, 0o755)
        rule_file = self.root / "etc/sudoers.d/cputune"
        self.assertEqual(rule_file.stat().st_mode & 0o777, 0o440)
        lines = rule_file.read_text().splitlines()
        self.assertTrue(lines[0].startswith("# /etc/sudoers.d/cputune"))
        self.assertEqual(lines[1], data["rule"])
        self.assertEqual(len(lines), 2)

    def test_rule_only(self) -> None:
        result = self.ok("sudoers", "--rule")
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("# /etc/sudoers.d/cputune"))
        self.assertTrue(lines[1].endswith(f"NOPASSWD: {self.root / 'usr/local/bin/cputune'}"))
        self.assertIn("install --system", result.stderr)
        # --json: the rule is the result, so it prints also then
        self.assertEqual(self.ok("sudoers", "--rule", "--json").stdout, result.stdout)

    def test_guide_for_a_copy_owned_by_root(self) -> None:
        m = load_module()
        status = {"user": "ann", "path": "/usr/local/bin/cputune", "installed": True, "works": False,
                  "version": "2.0.0", "outdated": False,
                  "rule": "ann ALL=(root) NOPASSWD: /usr/local/bin/cputune", "file": "/etc/sudoers.d/cputune",
                  "setup": "cputune install --no-password", "undo": "sudo rm /etc/sudoers.d/cputune"}
        text = "\n".join(m.render_sudoers(status))
        self.assertIn("cputune install --no-password", text)
        self.assertNotIn("install the newest", text)  # the copy is there already
        self.assertIn("ann ALL=(root) NOPASSWD: /usr/local/bin/cputune", text)
        text = "\n".join(m.render_sudoers(status | {"works": True}))
        self.assertIn("Everything is set up.", text)
        self.assertNotIn("install --no-password", text)
        # The rule works, but the copy is older than the one the user runs: root would run old code.
        text = "\n".join(m.render_sudoers(status | {"works": True, "outdated": True}))
        self.assertIn("/usr/local/bin/cputune is version 2.0.0: older than this copy", text)
        self.assertIn("install the newest cputune to /usr/local/bin/cputune", text)
        self.assertNotIn("add this rule", text)
        self.assertNotIn("Everything is set up.", text)

    def test_install_updates_an_old_copy_and_keeps_a_working_rule(self) -> None:
        m = load_module()
        status = {"user": "ann", "path": "/usr/local/bin/cputune", "installed": True, "works": True,
                  "version": "1.0.0", "outdated": True, "rule": "r"}
        install_copy = mock.Mock(return_value=(Path("/usr/local/bin/cputune"), "2.0.0", "GitHub", []))
        sudo_install = mock.Mock()
        with mock.patch.multiple(m, install_copy=install_copy, sudo_install=sudo_install):
            done = m.sudoers_install(status)
        install_copy.assert_called_once_with(system=True, no_check=False)
        sudo_install.assert_not_called()  # the rule works: it stays as it is
        self.assertEqual(done, ["Installed cputune 2.0.0 from GitHub to /usr/local/bin/cputune"])

    def test_owned_by_root(self) -> None:
        m = load_module()
        if os.stat(SCRIPT).st_uid != 0:
            self.assertFalse(m.owned_by_root(SCRIPT))
        self.assertFalse(m.owned_by_root(self.root / "missing"))
        if Path("/usr/bin/env").stat().st_uid == 0:
            self.assertTrue(m.owned_by_root(Path("/usr/bin/env")))
        tmp = Path("/tmp").stat()
        if tmp.st_uid == 0 and tmp.st_mode & 0o002:  # owned by root, but everyone can write it
            self.assertFalse(m.owned_by_root(Path("/tmp")))


class RootCopyTest(Case):
    """With a sudo rule, root must run the root-owned /usr/local/bin/cputune, whichever copy the
    user started: the rule names only that copy, and root must not run a copy the user can change."""

    tmp = None  # root_command checks that the copies are executable: not on a noexec /dev/shm

    def setUp(self) -> None:
        super().setUp()
        self.m = load_module()
        self.mine = self.root / "home/.local/bin/cputune"
        self.system = self.root / "usr/local/bin/cputune"
        for path, version in ((self.mine, "2.1.0"), (self.system, "2.1.0")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(with_version(version))
            path.chmod(0o755)
        self.patches = [
            mock.patch("os.geteuid", return_value=1000),  # CI runs the tests as root
            mock.patch.object(self.m, "SYSTEM_BIN", self.system.parent),
            mock.patch.object(self.m, "script_path", lambda: str(self.mine)),
            mock.patch.object(sys.stdin, "isatty", lambda: False),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def root_owned(self, *paths: Path):
        return mock.patch.object(self.m, "owned_by_root", lambda p: Path(p) in paths)

    def test_sudo_runs_the_root_owned_copy(self) -> None:
        with self.root_owned(self.system):
            self.assertEqual(self.m.root_command(["lock"]), ["sudo", "-n", str(self.system), "lock"])
        self.assertEqual(self.m.WARNINGS, [])

    def test_sudo_runs_the_copy_that_the_rule_names(self) -> None:
        # One decision (trusted_copy) for both: the rule must never name a copy that sudo does not run.
        for owned in ((), (self.system,), (self.mine,), (self.mine, self.system)):
            with self.subTest(owned=owned), self.root_owned(*owned), contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(self.m, "sudo_works", lambda p: None):
                status, copy = self.m.sudoers_status(), self.m.root_copy()
                if status["installed"]:
                    self.assertEqual(copy, status["path"])
                    self.assertIn(Path(copy), owned)  # root runs only a copy that only root can change
                else:  # no such copy: sudo runs this file and asks for a password, no rule names it
                    self.assertEqual(copy, str(self.mine))
                    self.assertEqual(status["path"], str(self.system))

    def test_without_root_owned_copy_sudo_runs_this_copy(self) -> None:
        with self.root_owned():
            self.assertEqual(self.m.root_command(["lock"]), ["sudo", "-n", str(self.mine), "lock"])

    def test_warns_when_the_root_owned_copy_is_older(self) -> None:
        self.system.write_bytes(with_version("2.0.0"))
        with self.root_owned(self.system), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.m.root_command(["lock"])[2], str(self.system))
        self.assertIn("older than this copy (2.1.0)", self.m.WARNINGS[0])

    def test_update_also_updates_the_root_owned_copy(self) -> None:
        self.system.write_bytes(with_version("2.0.0"))
        run_root = mock.Mock(return_value=subprocess.CompletedProcess([], 0, ""))
        with self.root_owned(self.system), mock.patch.object(self.m, "run_root", run_root):
            self.assertEqual(self.m.update_root_copy(self.mine, "2.1.0", quiet=False),
                             {"path": str(self.system), "updated": True})
            run_root.assert_called_once_with(["update"], capture=False)
            run_root.reset_mock()
            self.system.write_bytes(with_version("2.1.0"))  # up to date: no sudo
            self.assertFalse(self.m.update_root_copy(self.mine, "2.1.0", quiet=False)["updated"])
            run_root.assert_not_called()
        with self.root_owned(), mock.patch.object(self.m, "run_root", run_root):  # no root-owned copy
            self.assertIsNone(self.m.update_root_copy(self.mine, "9.0.0", quiet=False))
            run_root.assert_not_called()


    def test_status_finds_an_old_root_owned_copy(self) -> None:
        self.system.write_bytes(with_version("2.0.0"))
        with self.root_owned(self.system), mock.patch.object(self.m, "sudo_works", lambda p: True):
            status = self.m.sudoers_status()
        self.assertEqual((status["path"], status["version"], status["outdated"]), (str(self.system), "2.0.0", True))
        self.system.write_bytes(with_version("2.1.0"))
        with self.root_owned(self.system), mock.patch.object(self.m, "sudo_works", lambda p: True):
            self.assertFalse(self.m.sudoers_status()["outdated"])

    def test_update_command_calls_the_root_copy_update(self) -> None:
        update_root_copy = mock.Mock(return_value=None)
        with mock.patch.multiple(self.m, download=lambda: with_version("2.1.0"),
                                 update_root_copy=update_root_copy, OUT=io.StringIO()):  # OUT: print_json
            self.m.cmd_update(types.SimpleNamespace(json=True))
        update_root_copy.assert_called_once_with(self.mine, "2.1.0", True)


class SecurityTest(Case):
    """A sudo rule lets every program of the user run cputune as root. These tests make
    sure that this gives no more than the CPU settings."""

    def load_as_sudo_root(self):
        env = {"SUDO_UID": "1000", "CPUTUNE_ROOT": str(self.root), "CPUTUNE_SOURCE_URL": "file:///evil"}
        with mock.patch("os.geteuid", return_value=0), mock.patch.dict(os.environ, env):
            return load_module()

    def test_sudo_allows_only_root_commands(self) -> None:
        m = self.load_as_sudo_root()
        self.assertTrue(m.UNDER_SUDO)
        self.assertEqual(m.SUDO_COMMANDS, ("lock", "limit", "reset", "update"))
        # Stubs instead of the real commands: under the simulated sudo, the test hooks are
        # off, so a real command would change this machine if the check ever broke.
        called: list[str] = []

        def stub(name: str):
            def cmd(args) -> None:
                called.append(name)
            cmd.__name__ = f"cmd_{name}"
            return cmd

        names = ("show", "watch", "check", "lock", "run", "limit", "reset", "sudoers", "install", "update")
        with mock.patch.multiple(m, **{f"cmd_{n}": stub(n) for n in names}):
            for argv in (["run", "--", "sh"], ["install"], ["install", "--system"], ["sudoers", "--install"],
                         ["show"], ["watch"]):
                with self.subTest(argv=argv):
                    stderr = io.StringIO()
                    with mock.patch.object(sys, "argv", ["cputune", *argv]), contextlib.redirect_stderr(stderr):
                        self.assertEqual(m.main(), 1)
                    self.assertIn("do not start", stderr.getvalue())
            self.assertEqual(called, [])
            with mock.patch.object(sys, "argv", ["cputune", "lock"]):
                self.assertEqual(m.main(), 0)
            self.assertEqual(called, ["lock"])
        with mock.patch.object(sys, "argv", ["cputune", "--version"]), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:
                m.main()
        self.assertEqual(exit.exception.code, 0)  # sudoers --install checks the rule with --version

    def test_no_test_hooks_under_sudo(self) -> None:
        m = self.load_as_sudo_root()
        self.assertFalse(m.TEST_MODE)
        self.assertEqual(m.ROOT, Path("/"))
        self.assertEqual(m.CPU_BASE, Path("/sys/devices/system/cpu"))

    def test_owned_by_root_checks_every_folder(self) -> None:
        m = load_module()
        path = Path("/usr/bin/env").resolve()
        real_stat = os.stat

        def fake_stat(writable: str):
            def stat(p, *args, **kwargs):
                if Path(p) not in (path, *path.parents):
                    return real_stat(p, *args, **kwargs)
                return types.SimpleNamespace(st_uid=0, st_mode=0o40775 if str(p) == writable else 0o40755)
            return stat

        with mock.patch("os.stat", fake_stat(writable="")):
            self.assertTrue(m.owned_by_root(path))
        top = str(list(path.parents)[-2])  # for example /usr: a folder far above the file
        with mock.patch("os.stat", fake_stat(writable=top)):
            self.assertFalse(m.owned_by_root(path))

    def test_owned_by_root_resolves_symlinks(self) -> None:
        m = load_module()
        own = self.root / "own-file"
        own.write_text("x")
        with mock.patch.object(Path, "resolve", lambda self, strict=False: own):
            self.assertFalse(m.owned_by_root(Path("/usr/bin/env")))  # really a link to a user file

    def test_rule_names_the_real_user(self) -> None:
        result = self.run_tool("sudoers", "--json", USER="mallory", LOGNAME="mallory")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["rule"].startswith(f"{pwd.getpwuid(os.getuid()).pw_name} ALL="))

    def test_replace_file_does_not_follow_planted_symlinks(self) -> None:
        m = load_module()
        folder = self.root / "bin"
        folder.mkdir()
        victim = self.root / "victim"
        victim.write_text("keep")
        for pid in range(4240, 4245):  # the names that a predictable temporary file would use
            (folder / f".cputune.{pid}.tmp").symlink_to(victim)
        with mock.patch("os.getpid", return_value=4242):
            m.replace_file(folder / "cputune", b"new")
        self.assertEqual(victim.read_text(), "keep")
        self.assertEqual((folder / "cputune").read_bytes(), b"new")


if __name__ == "__main__":
    unittest.main()
