# =============================================================================
# tasks.ps1 - PowerShell equivalent of the Makefile.
#
# Windows PowerShell does not have `make` installed by default, so use this
# script instead. Usage:
#
#   .\tasks.ps1 eda
#   .\tasks.ps1 pipeline1
#   .\tasks.ps1 all
#
# If PowerShell blocks the script with an "execution policy" error, run this
# once in the same terminal (session-only, does not change system settings):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# =============================================================================

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "eda", "discretize",
        "pipeline1", "pipeline2", "pipeline3", "pipeline4",
        "tune", "xai", "stats", "uncertainty", "learning-curves",
        "mlflow-ui", "all"
    )]
    [string]$Task
)

function Invoke-Step($Command) {
    Write-Host ">> $Command" -ForegroundColor Cyan
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed: $Command" -ForegroundColor Red
        exit 1
    }
}

switch ($Task) {
    "eda"              { Invoke-Step "python scripts\run_eda.py --all" }
    "discretize"       { Invoke-Step "python scripts\make_discrete_datasets.py" }
    "pipeline1"        { Invoke-Step "python scripts\run_pipeline1_single_original.py" }
    "pipeline2"        { Invoke-Step "python scripts\run_pipeline2_multi_original.py" }
    "pipeline3"        { Invoke-Step "python scripts\run_pipeline3_single_discrete.py" }
    "pipeline4"        { Invoke-Step "python scripts\run_pipeline4_multi_discrete.py" }
    "tune"             { Invoke-Step "python scripts\run_tuning_top3.py" }
    "xai"              { Invoke-Step "python scripts\run_xai.py" }
    "stats"            { Invoke-Step "python scripts\run_stat_tests.py" }
    "uncertainty"      { Invoke-Step "python scripts\run_uncertainty.py" }
    "learning-curves"  { Invoke-Step "python scripts\run_learning_curves.py" }
    "mlflow-ui"        { Invoke-Step "mlflow ui --backend-store-uri sqlite:///mlflow.db" }
    "all" {
        Invoke-Step "python scripts\run_eda.py --all"
        Invoke-Step "python scripts\run_pipeline1_single_original.py"
        Invoke-Step "python scripts\run_pipeline2_multi_original.py"
        Invoke-Step "python scripts\make_discrete_datasets.py"
        Invoke-Step "python scripts\run_pipeline3_single_discrete.py"
        Invoke-Step "python scripts\run_pipeline4_multi_discrete.py"
        Invoke-Step "python scripts\run_tuning_top3.py"
        Invoke-Step "python scripts\run_xai.py"
        Invoke-Step "python scripts\run_uncertainty.py"
        Invoke-Step "python scripts\run_learning_curves.py"
        Invoke-Step "python scripts\run_stat_tests.py"
    }
}
