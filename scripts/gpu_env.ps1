<#
.SYNOPSIS
    Imports the MSVC x64 build environment into the current PowerShell session.

.DESCRIPTION
    torch.compile's inductor backend shells out to the MSVC host compiler
    (cl.exe) to build its generated C++ wrappers. Visual Studio does not put
    cl.exe on the global PATH, so a bare `python -c "torch.compile(...)"` fails
    with:

        InductorError: RuntimeError: Compiler: cl is not found.

    Dot-source this before running anything that compiles:

        . .\scripts\gpu_env.ps1
        python -m pytest simulator2/tests/test_torch_rhs.py

    This is the native-Windows substitute for the WSL2 toolchain assumed by
    TORCH_PORT_SPEC.md section 3. WSL is not installed on this workstation.
#>

$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found. Is Visual Studio installed?"
}

$vsRoot = & $vswhere -latest -products * `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $vsRoot) {
    throw "No Visual Studio install with the C++ x64 toolchain was found."
}

$vcvars = Join-Path $vsRoot 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found under $vsRoot"
}

# Run vcvars64 in a child cmd and lift the resulting environment back out.
# `set` after the batch file gives us the fully-populated PATH/INCLUDE/LIB.
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}

$cl = (Get-Command cl.exe -ErrorAction SilentlyContinue).Source
if (-not $cl) {
    throw "vcvars64 ran but cl.exe is still not on PATH."
}
Write-Host "MSVC ready: $cl"

# ---------------------------------------------------------------------------
# Workaround: triton's driver.c calls alloca(), and MSVC only defines the
# alloca -> _alloca compatibility macro in <malloc.h> when non-standard names
# are enabled. triton compiles with /std:c11, which turns them off, so the call
# survives compilation as an implicit declaration and then fails at link:
#
#     error LNK2019: unresolved external symbol alloca
#                    referenced in function launchKernel
#
# cl.exe prepends the CL variable to its command line, and triton's
# subprocess.check_call inherits our environment, so defining the macro here
# fixes the build without patching site-packages.
# ---------------------------------------------------------------------------
$env:CL = "/Dalloca=_alloca"
Write-Host "CL set to: $env:CL"
