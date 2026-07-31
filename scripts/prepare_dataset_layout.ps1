<#
.SYNOPSIS
    Expose the QualCompare rendered dataset in the Source/<N>VP + Distorted/<N>VP
    layout expected by Light_GraphicsLPIPS_csv.py, train.py and correlation_VP.py.

.DESCRIPTION
    The published rendered dataset (`qualcomparerendered`) extracts to a flat,
    per-dataset layout:

        <DatasetRoot>/<NAME>_source/<REF_OBJECT>/{views,masks,patchs}
        <DatasetRoot>/<NAME>_distorted/<DISTORTED_OBJECT>/{views,masks}

    but the tools expect:

        <SRC_ROOT>/Source/<N>VP/<REF_OBJECT>/{views,patchs}
        <SRC_ROOT>/Distorted/<N>VP/<DISTORTED_OBJECT>/views

    This helper creates directory JUNCTIONS (no copy) exposing the extracted
    folders in the expected layout. Two layouts are supported:

      * default  -> <OutRoot>/<DB>/Source/<N>VP
                    Use with the Python scripts directly: --src_root <OutRoot>/<DB>

      * -ForBat  -> <OutRoot>/<DB>/<RENDER_METHOD>/<VIEW_METHOD>/Source/<N>VP
                    Matches the paper_revalidation *.bat presets, which build
                    SRC_ROOT = <QUALCOMPARE_OUT_ROOT>/<DB>/<RENDER_METHOD>/<VIEW_METHOD>.
                    Then: set QUALCOMPARE_OUT_ROOT=<OutRoot>

    The dataset list (archive names, number of views) is read from
    <DatasetRoot>\dataset_info.json when present; otherwise a built-in fallback
    is used. The RENDER_METHOD/VIEW_METHOD labels used by -ForBat mirror the
    revalidate_*_qualcompare.bat presets.

.PARAMETER DatasetRoot
    Folder containing the extracted <NAME>_source / <NAME>_distorted folders.

.PARAMETER OutRoot
    Where to create the layout. Default: <DatasetRoot>\_run.

.PARAMETER ForBat
    Build the deeper <DB>/<RENDER_METHOD>/<VIEW_METHOD> layout for the .bat flow.

.PARAMETER Remove
    Remove the junctions previously created (source data is never touched).

.EXAMPLE
    .\scripts\prepare_dataset_layout.ps1 -DatasetRoot "D:\path\to\qualcomparerendered"
    # -> --src_root "D:\path\to\qualcomparerendered\_run\TSMD"

.EXAMPLE
    .\scripts\prepare_dataset_layout.ps1 -DatasetRoot "D:\path\to\qualcomparerendered" -ForBat
    # -> set QUALCOMPARE_OUT_ROOT=D:\path\to\qualcomparerendered\_run
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $DatasetRoot,
    [string] $OutRoot,
    [switch] $ForBat,
    [switch] $Remove
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $DatasetRoot)) { throw "DatasetRoot not found: $DatasetRoot" }
if (-not $OutRoot -or $OutRoot -eq "") { $OutRoot = Join-Path $DatasetRoot "_run" }

# RENDER_METHOD|VIEW_METHOD labels used by the paper_revalidation .bat presets.
# NB: the two .bat files disagree on TSMD's view label (Y_fixed_0 in
# revalidate_table, Y_fixed_0.3 in revalidate_fixed_baselines); both are created.
$batLabels = @{
    "TMQ"       = @("New_Render|Y_fixed_0.3")
    "TSMD"      = @("New_Render|Y_fixed_0", "New_Render|Y_fixed_0.3")
    "SJTU-TMQA" = @("0_0_light|Y_fixed_0")
    "BASICS"    = @("SP_960x960|Y_fixed_0.3")
    "WPC"       = @("SP_960x960|Y_fixed_0.3")
}

# --- Discover datasets ------------------------------------------------------
$infoPath = Join-Path $DatasetRoot "dataset_info.json"
$datasets = @()
if (Test-Path -LiteralPath $infoPath) {
    Write-Host "Reading $infoPath"
    $info = Get-Content -LiteralPath $infoPath -Raw | ConvertFrom-Json
    foreach ($d in $info.datasets) {
        $datasets += [pscustomobject]@{ Db = $d.base_dataset; Base = $d.name; Views = [int]$d.rendering.num_views }
    }
} else {
    Write-Warning "dataset_info.json not found; using built-in fallback list."
    $datasets = @(
        [pscustomobject]@{ Db = "TMQ";       Base = "TMQ_Circle_0.3_8VP";         Views = 8 },
        [pscustomobject]@{ Db = "TSMD";      Base = "TSMD_Circle_0.3_8VP";        Views = 8 },
        [pscustomobject]@{ Db = "SJTU-TMQA"; Base = "SJTU-TMQA_Circle_0_8VP";     Views = 8 },
        [pscustomobject]@{ Db = "BASICS";    Base = "BASICS_Circle_0.3_8VP_r003"; Views = 8 },
        [pscustomobject]@{ Db = "WPC";       Base = "WPC_Circle_0.3_8VP_r001";    Views = 8 }
    )
}

function Remove-JunctionSafe([string] $Path) {
    if (Test-Path -LiteralPath $Path) { & cmd /c rmdir "`"$Path`"" | Out-Null }   # removes the link only
}

function New-LayoutPair([string] $BaseDir, [string] $Vp, [string] $Src, [string] $Dis, [bool] $DoRemove) {
    $linkSrc = Join-Path $BaseDir (Join-Path "Source"    $Vp)
    $linkDis = Join-Path $BaseDir (Join-Path "Distorted" $Vp)
    if ($DoRemove) { Remove-JunctionSafe $linkSrc; Remove-JunctionSafe $linkDis; return }
    New-Item -ItemType Directory -Force -Path (Split-Path $linkSrc -Parent) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path $linkDis -Parent) | Out-Null
    if (-not (Test-Path -LiteralPath $linkSrc)) { New-Item -ItemType Junction -Path $linkSrc -Target $Src | Out-Null }
    if (-not (Test-Path -LiteralPath $linkDis)) { New-Item -ItemType Junction -Path $linkDis -Target $Dis | Out-Null }
}

foreach ($d in $datasets) {
    $src = Join-Path $DatasetRoot ("{0}_source"    -f $d.Base)
    $dis = Join-Path $DatasetRoot ("{0}_distorted" -f $d.Base)
    $vp  = "{0}VP" -f $d.Views

    if (-not $Remove -and (-not (Test-Path -LiteralPath $src) -or -not (Test-Path -LiteralPath $dis))) {
        Write-Warning ("{0}: extracted archives not found (need '{1}' and '{2}'). Skipped." -f $d.Db, (Split-Path $src -Leaf), (Split-Path $dis -Leaf))
        continue
    }

    if ($ForBat) {
        $labels = $batLabels[$d.Db]
        if (-not $labels) { Write-Warning ("{0}: no .bat label mapping; skipped in -ForBat mode." -f $d.Db); continue }
        foreach ($rv in $labels) {
            $parts  = $rv.Split("|"); $render = $parts[0]; $view = $parts[1]
            $baseDir = Join-Path $OutRoot (Join-Path $d.Db (Join-Path $render $view))
            New-LayoutPair $baseDir $vp $src $dis $Remove.IsPresent
            if (-not $Remove) { Write-Host ("[ok] {0,-10} {1}/{2}  --src_root `"{3}`"" -f $d.Db, $render, $view, $baseDir) }
        }
    } else {
        $baseDir = Join-Path $OutRoot $d.Db
        New-LayoutPair $baseDir $vp $src $dis $Remove.IsPresent
        if (-not $Remove) { Write-Host ("[ok] {0,-10} --src_root `"{1}`"" -f $d.Db, $baseDir) }
    }
    if ($Remove) { Write-Host ("[removed] {0}" -f $d.Db) }
}

if (-not $Remove) {
    Write-Host ""
    Write-Host "Layout ready under: $OutRoot"
    if ($ForBat) {
        Write-Host "For the paper_revalidation .bat presets, set:"
        Write-Host "    set QUALCOMPARE_OUT_ROOT=$OutRoot"
    } else {
        Write-Host "Pass the printed --src_root to Light_GraphicsLPIPS_csv.py / train.py / correlation_VP.py."
    }
}
