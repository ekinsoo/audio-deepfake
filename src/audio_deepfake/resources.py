"""Small process/system memory probes without GPU or monitoring dependencies."""

import ctypes
import os
import sys


if os.name == "nt":
    from ctypes import wintypes

    # ctypes retains POINTER(Structure) types globally. Defining these inside
    # every probe leaks a new pair of types per call during long experiments.
    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage",
            )
        ]

    class SystemMemory(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_phys", "available_phys", "total_page", "available_page",
                "total_virtual", "available_virtual", "available_extended",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.argtypes = []
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(SystemMemory)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL


def memory_snapshot() -> dict:
    if os.name == "nt":
        process = ProcessMemory()
        process.cb = ctypes.sizeof(process)
        system = SystemMemory()
        system.length = ctypes.sizeof(system)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(system)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "method": "Windows GetProcessMemoryInfo / GlobalMemoryStatusEx",
            "rss_bytes": process.WorkingSetSize,
            "peak_rss_bytes": process.PeakWorkingSetSize,
            "system_total_ram_bytes": system.total_phys,
            "system_available_ram_bytes": system.available_phys,
        }
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "method": "resource.getrusage; lifetime peak, current RSS not measured",
        "rss_bytes": None,
        "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
        "system_total_ram_bytes": None,
        "system_available_ram_bytes": None,
    }
