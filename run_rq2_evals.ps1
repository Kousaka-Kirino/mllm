# 依次评测三臂 + 与基线配对比较（GPU 只能串行跑，所以排成一个脚本）
$ErrorActionPreference = "Continue"
Set-Location "D:\TheZe\复试项目\mLLM"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_OFFLINE = "1"
$env:PICOVLM_DATA = "D:\TheZe\复试项目\mLLM\data\full"
$py = "D:\Miniconda3\envs\pandavla\python.exe"

foreach ($arm in @("curriculum", "counterfactual", "grpo")) {
    Write-Output "=== eval arm_$arm ==="
    & $py -m stage4_eval.run --model sft --suite dev_test `
        --conditions correct_image,mismatched_image `
        --connector "outputs/ckpt/${arm}_full/connector.pt" `
        --adapter "outputs/ckpt/${arm}_full/lora" `
        --run-id "arm_$arm" 2>&1 | Select-Object -Last 2
    & $py -m stage4_eval.report --run-id "arm_$arm" --suite dev_test 2>&1 | Select-Object -Last 2
    Write-Output "=== compare baseline vs arm_$arm ==="
    & $py -m stage5_intervene.compare_arms --a sft_fulltrain_dev --b "arm_$arm" 2>&1 | Select-Object -Last 20
}
Write-Output "ALL DONE"
