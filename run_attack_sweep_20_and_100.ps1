$ErrorActionPreference = "Stop"

# Assumes this script is placed in the ROBIN project root.
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Inputs
$ModelId = "runwayml/stable-diffusion-v1-5"
$WmPath  = "ckpts\optimized_wm5-30_embedding-step-50.pt"

# Prompt sets
$PromptSets = @(
    @{ Name = "unseen20"; PromptFile = "new_unseen_prompts_20.txt"; End = 20 },
    @{ Name = "prompts100"; PromptFile = "prompts_100.txt"; End = 100 }
)

# Scripts
$BaselineScript = "inject_wm_baseline.py"
$PredictorScript = "inject_wm_with_predictor.py"

# Output folders
$OutRoot = "results\sweep_all"
$LogRoot = Join-Path $OutRoot "logs"
$BaselineLogRoot = Join-Path $LogRoot "baseline"
$PredictorLogRoot = Join-Path $LogRoot "predictor"
New-Item -ItemType Directory -Force -Path $BaselineLogRoot | Out-Null
New-Item -ItemType Directory -Force -Path $PredictorLogRoot | Out-Null

# CSV output
$CsvPath = Join-Path $OutRoot "attack_sweep_results.csv"
New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
if (Test-Path $CsvPath) { Remove-Item $CsvPath -Force }

function Write-RunRows {
    param(
        [string]$LogFile,
        [string]$PromptSet,
        [string]$Method,
        [string]$CaseName,
        [string]$CsvFile
    )

    $text = Get-Content $LogFile -Raw

    $meta = [regex]::Match(
        $text,
        'steps:\s*(?<steps>\d+),\s*radius:\s*(?<radius>[^,]+),\s*wm_seed:\s*(?<wm_seed>\d+),\s*opt wi&opt wt'
    )

    $quality = [regex]::Match(
        $text,
        'psnr:\s*(?<psnr>[-0-9.eE]+),\s*ssim:\s*(?<ssim>[-0-9.eE]+),\s*msssim:\s*(?<msssim>[-0-9.eE]+)'
    )

    $attackPattern = 'attack:\s*(?<attack>[^\r\n]+)\r?\nauc:\s*(?<auc>[-0-9.eE]+),\s*acc:\s*(?<acc>[-0-9.eE]+),\s*TPR@1%FPR:\s*(?<tpr>[-0-9.eE]+)\r?\nmse_mean:\s*(?<mse>[-0-9.eE]+),\s*w_mse_mean:\s*(?<wmse>[-0-9.eE]+)'
    $matches = [regex]::Matches($text, $attackPattern, [System.Text.RegularExpressions.RegexOptions]::Multiline)

    if ($matches.Count -eq 0) {
        throw "No attack blocks found in log: $LogFile"
    }

    $rows = foreach ($m in $matches) {
        [pscustomobject]@{
            prompt_set   = $PromptSet
            method       = $Method
            case_name    = $CaseName
            attack       = $m.Groups['attack'].Value.Trim()
            steps        = if ($meta.Success) { $meta.Groups['steps'].Value } else { "" }
            radius       = if ($meta.Success) { $meta.Groups['radius'].Value.Trim() } else { "" }
            wm_seed      = if ($meta.Success) { $meta.Groups['wm_seed'].Value } else { "" }
            auc          = [double]$m.Groups['auc'].Value
            acc          = [double]$m.Groups['acc'].Value
            tpr_at_1pct  = [double]$m.Groups['tpr'].Value
            mse_mean     = [double]$m.Groups['mse'].Value
            w_mse_mean   = [double]$m.Groups['wmse'].Value
            psnr         = if ($quality.Success) { [double]$quality.Groups['psnr'].Value } else { $null }
            ssim         = if ($quality.Success) { [double]$quality.Groups['ssim'].Value } else { $null }
            msssim       = if ($quality.Success) { [double]$quality.Groups['msssim'].Value } else { $null }
            log_file     = Split-Path -Leaf $LogFile
        }
    }

    if (-not (Test-Path $CsvFile)) {
        $rows | Export-Csv -Path $CsvFile -NoTypeInformation -Encoding UTF8
    } else {
        $rows | Export-Csv -Path $CsvFile -NoTypeInformation -Append -Encoding UTF8
    }
}

function Run-Case {
    param(
        [string]$Label,
        [string]$ScriptPath,
        [string[]]$Args,
        [string]$LogFile,
        [string]$PromptSetName,
        [string]$MethodName,
        [string]$CaseName,
        [string]$CsvFile
    )

    Write-Host "========================================"
    Write-Host "Running: $Label"
    Write-Host "Log: $LogFile"
    Write-Host "========================================"

    & python $ScriptPath @Args 2>&1 | Tee-Object -FilePath $LogFile

    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Label"
        exit $LASTEXITCODE
    }

    Write-RunRows -LogFile $LogFile -PromptSet $PromptSetName -Method $MethodName -CaseName $CaseName -CsvFile $CsvFile
}

# Attack severity sweeps
$Cases = @(
    @{ Name = "jpeg_95";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--jpeg_ratio","95") },
    @{ Name = "jpeg_75";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--jpeg_ratio","75") },
    @{ Name = "jpeg_50";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--jpeg_ratio","50") },
    @{ Name = "jpeg_25";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--jpeg_ratio","25") },

    @{ Name = "crop_5";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--crop_scale","0.95","--crop_ratio","0.95") },
    @{ Name = "crop_20";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--crop_scale","0.80","--crop_ratio","0.80") },
    @{ Name = "crop_40";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--crop_scale","0.60","--crop_ratio","0.60") },
    @{ Name = "crop_60";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--crop_scale","0.40","--crop_ratio","0.40") },

    @{ Name = "blur_3";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--gaussian_blur_r","3") },
    @{ Name = "blur_5";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--gaussian_blur_r","5") },
    @{ Name = "blur_9";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--gaussian_blur_r","9") },
    @{ Name = "blur_13";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--gaussian_blur_r","13") },

    @{ Name = "rot_5";    Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--rotation_angle","5") },
    @{ Name = "rot_15";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--rotation_angle","15") },
    @{ Name = "rot_30";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--rotation_angle","30") },
    @{ Name = "rot_45";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--rotation_angle","45") },

    @{ Name = "noise_01"; Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--noise_std","0.01") },
    @{ Name = "noise_03"; Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--noise_std","0.03") },
    @{ Name = "noise_05"; Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--noise_std","0.05") },
    @{ Name = "noise_10"; Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--noise_std","0.10") },

    @{ Name = "cj_12";    Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--color_jitter_brightness","1.2") },
    @{ Name = "cj_2";     Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--color_jitter_brightness","2") },
    @{ Name = "cj_4";     Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--color_jitter_brightness","4") },
    @{ Name = "cj_6";     Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--color_jitter_brightness","6") }
)

$ChainCases = @(
    @{ Name = "jpeg75_crop20"; Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--jpeg_ratio","75","--crop_scale","0.80","--crop_ratio","0.80") },
    @{ Name = "blur5_noise03";  Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--gaussian_blur_r","5","--noise_std","0.03") },
    @{ Name = "rot15_jpeg50";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--rotation_angle","15","--jpeg_ratio","50") },
    @{ Name = "crop40_blur9";   Args = @("--dataset","custom","--prompt_file","__PROMPT__","--start","0","--end","__END__","--model_id",$ModelId,"--wm_path",$WmPath,"--crop_scale","0.60","--crop_ratio","0.60","--gaussian_blur_r","9") }
)

foreach ($set in $PromptSets) {
    $promptFile = $set.PromptFile
    $end = [string]$set.End
    $setName = $set.Name

    Write-Host ""
    Write-Host "########################################"
    Write-Host "PROMPT SET: $setName ($promptFile)"
    Write-Host "########################################"
    Write-Host ""

    foreach ($case in $Cases) {
        $argsBase = @()
        for ($i = 0; $i -lt $case.Args.Count; $i++) {
            $val = $case.Args[$i]
            if ($val -eq "__PROMPT__") { $val = $promptFile }
            if ($val -eq "__END__") { $val = $end }
            $argsBase += $val
        }

        Run-Case -Label ("baseline_" + $setName + "_" + $case.Name) `
                 -ScriptPath $BaselineScript `
                 -Args $argsBase `
                 -LogFile (Join-Path $BaselineLogRoot ($setName + "_" + $case.Name + ".txt")) `
                 -PromptSetName $setName `
                 -MethodName "baseline" `
                 -CaseName $case.Name `
                 -CsvFile $CsvPath

        Run-Case -Label ("predictor_" + $setName + "_" + $case.Name) `
                 -ScriptPath $PredictorScript `
                 -Args $argsBase `
                 -LogFile (Join-Path $PredictorLogRoot ($setName + "_" + $case.Name + ".txt")) `
                 -PromptSetName $setName `
                 -MethodName "predictor" `
                 -CaseName $case.Name `
                 -CsvFile $CsvPath
    }

    foreach ($case in $ChainCases) {
        $argsChain = @()
        for ($i = 0; $i -lt $case.Args.Count; $i++) {
            $val = $case.Args[$i]
            if ($val -eq "__PROMPT__") { $val = $promptFile }
            if ($val -eq "__END__") { $val = $end }
            $argsChain += $val
        }

        Run-Case -Label ("baseline_" + $setName + "_" + $case.Name) `
                 -ScriptPath $BaselineScript `
                 -Args $argsChain `
                 -LogFile (Join-Path $BaselineLogRoot ($setName + "_" + $case.Name + ".txt")) `
                 -PromptSetName $setName `
                 -MethodName "baseline" `
                 -CaseName $case.Name `
                 -CsvFile $CsvPath

        Run-Case -Label ("predictor_" + $setName + "_" + $case.Name) `
                 -ScriptPath $PredictorScript `
                 -Args $argsChain `
                 -LogFile (Join-Path $PredictorLogRoot ($setName + "_" + $case.Name + ".txt")) `
                 -PromptSetName $setName `
                 -MethodName "predictor" `
                 -CaseName $case.Name `
                 -CsvFile $CsvPath
    }
}

Write-Host ""
Write-Host "All attack sweeps completed."
Write-Host "CSV saved to: $CsvPath"