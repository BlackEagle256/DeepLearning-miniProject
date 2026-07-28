# =============================================================================
# tasks.ps1 - PowerShell equivalent of the Makefile.
#
# Windows PowerShell does not have `make` installed by default, so use this
# script instead. Every phase has an INDEPENDENT script per dataset under
# scripts/Dataset_0136/, scripts/Dataset_0172/, scripts/Dataset_3772/ (the
# assignment's "har dataset bayad pipeline jodagane / script-haye mostaghel
# dashte bashad" requirement).
#
# Usage:
#   .\tasks.ps1 eda-Dataset_0136     # one dataset only
#   .\tasks.ps1 eda                  # convenience: all 3 datasets
#   .\tasks.ps1 all                  # convenience: every phase, all datasets
#
# If PowerShell blocks the script with an "execution policy" error, run this
# once in the same terminal (session-only, does not change system settings):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# =============================================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$Task
)

$Datasets = @("Dataset_0136", "Dataset_0172", "Dataset_3772")

$PerDatasetScript = @{
    "eda"             = "run_eda.py"
    "discretize"      = "run_discretize.py"
    "pipeline1"       = "run_pipeline1_single_original.py"
    "pipeline2"       = "run_pipeline2_multi_original.py"
    "pipeline3"       = "run_pipeline3_single_discrete.py"
    "pipeline4"       = "run_pipeline4_multi_discrete.py"
    "tune"            = "run_tuning.py"
    "xai"             = "run_xai.py"
    "uncertainty"     = "run_uncertainty.py"
    "learning-curves" = "run_learning_curves.py"
    "stats"           = "run_stat_tests.py"
}

# Flat "all 3 datasets" convenience wrappers (loop the per-dataset scripts).
$FlatScript = @{
    "eda"             = "scripts\run_eda.py --all"
    "discretize"      = "scripts\make_discrete_datasets.py"
    "pipeline1"       = "scripts\run_pipeline1_single_original.py"
    "pipeline2"       = "scripts\run_pipeline2_multi_original.py"
    "pipeline3"       = "scripts\run_pipeline3_single_discrete.py"
    "pipeline4"       = "scripts\run_pipeline4_multi_discrete.py"
    "tune"            = "scripts\run_tuning_top3.py"
    "xai"             = "scripts\run_xai.py"
    "uncertainty"     = "scripts\run_uncertainty.py"
    "learning-curves" = "scripts\run_learning_curves.py"
    "stats"           = "scripts\run_stat_tests.py"
}

function Invoke-Step($Command) {
    Write-Host ">> $Command" -ForegroundColor Cyan
    Invoke-Expression "python $Command"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed: $Command" -ForegroundColor Red
        exit 1
    }
}

# --- per-dataset task, e.g. "eda-Dataset_0136" --------------------------
if ($Task -match "^(?<phase>[a-z0-9-]+)-(?<dataset>Dataset_\d+)$") {
    $phase = $Matches.phase
    $dataset = $Matches.dataset
    if (-not $PerDatasetScript.ContainsKey($phase)) {
        Write-Host "Unknown phase '$phase' in task '$Task'" -ForegroundColor Red
        exit 1
    }
    Invoke-Step "scripts\$dataset\$($PerDatasetScript[$phase])"
    exit 0
}

switch ($Task) {
    "select-top3"   { Invoke-Step "scripts\select_top3.py" }
    "describe-top3" { Invoke-Step "scripts\describe_top3.py" }
    "mlflow-ui"     { Invoke-Expression "mlflow ui --backend-store-uri sqlite:///mlflow.db" }
    "all" {
        Invoke-Step $FlatScript["eda"]
        Invoke-Step $FlatScript["pipeline1"]
        Invoke-Step $FlatScript["pipeline2"]
        Invoke-Step $FlatScript["discretize"]
        Invoke-Step "scripts\select_top3.py"
        Invoke-Step $FlatScript["pipeline3"]
        Invoke-Step $FlatScript["pipeline4"]
        Invoke-Step $FlatScript["tune"]
        Invoke-Step $FlatScript["xai"]
        Invoke-Step $FlatScript["uncertainty"]
        Invoke-Step $FlatScript["learning-curves"]
        Invoke-Step $FlatScript["stats"]
        Invoke-Step "scripts\describe_top3.py"
    }
    default {
        if ($FlatScript.ContainsKey($Task)) {
            Invoke-Step $FlatScript[$Task]
        } else {
            Write-Host "Unknown task '$Task'." -ForegroundColor Red
            Write-Host "Valid: eda, pipeline1..4, discretize, tune, xai, uncertainty, learning-curves, stats, select-top3, describe-top3, mlflow-ui, all"
            Write-Host "Or per-dataset: <phase>-Dataset_0136 / -Dataset_0172 / -Dataset_3772"
            exit 1
        }
    }
}
