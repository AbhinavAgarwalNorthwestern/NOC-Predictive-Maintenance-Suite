# Bundles a model for SageMaker deployment (model.tar.gz at S3)
param([string]$ModelName = "autonomy")

$src = "outputs/models/$ModelName"
$tar = "outputs/models/${ModelName}_model.tar.gz"
$bucket = "battery-pdm-dev-models-998716768706"

# Copy serve.py into the model dir so SageMaker container finds it
cp src/battery_pdm/inference/serve.py "$src/"
tar -czf $tar -C $src .
aws s3 cp $tar "s3://${bucket}/$ModelName/model.tar.gz"
Write-Host "Uploaded: s3://${bucket}/$ModelName/model.tar.gz"
