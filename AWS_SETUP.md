# AWS setup for the SSAE sweep

Runbook for moving the sweep off Windows onto a single-GPU AWS box. Covers instance sizing, storage staging via S3, and running `scripts/compare_topk.sh` end-to-end.

## Instance sizing

**Recommendation: `g6.2xlarge`** (1x L4 24 GB, 8 vCPU, 32 GB RAM, ~$1/hr on-demand in `us-east-1`). Fallback to `g5.2xlarge` (1x A10G 24 GB, ~$1.20/hr) if L4 quota is not approved.

Why 24 GB VRAM: SD3.5 with NF4-quantized T5 needs ~16-20 GB for `run_image_benchmark` inference; the remainder covers CLIP, LPIPS, and DINOv2 scoring on the same card. Smaller cards (T4 16 GB) risk OOM during the bench. Skip A100/H100 unless you plan to parallelize sweep points across GPUs; the single-card bottleneck is the diffusion step, not raw FLOPs.

Rough cost for the no-interaction-terms ablation (4 configs, train plus image bench with drop, swap, and DINO): ~30-50 GPU-hours, so $30-50 on-demand.

**Storage**: 500 GB gp3 root volume. T5 embeddings for 74k train and 18k holdout prompts at 333 x 4096 float32 come to roughly 500 GB uncompressed. Bench PNGs add tens of GB on top.

## 1. Request G-family quota

New AWS accounts default to 0 vCPUs for G instances. Do this first, since approval sometimes takes hours.

1. Console → **Service Quotas** → EC2 → **Running On-Demand G and VT instances**.
2. Request at least 8 vCPUs (one `g6.2xlarge`). Request 16 if you want headroom for `g6.4xlarge`.
3. Pick the region you will actually launch in. `us-east-1` and `us-west-2` have the best G6 availability.

`g5.2xlarge` shares the same quota bucket and has broader availability if you need to fall back.

## 2. Create the S3 bucket

Use S3 as the staging layer between your Windows PC and the EC2 box. Faster than `rsync` over SSH for TB-scale data and gives you a persistent copy of the extracted embeddings.

### 2a. Create the bucket

1. Console → **S3** → **Create bucket**.
2. Name: something like `<yourname>-ssae-data`. Bucket names are globally unique.
3. Region: **same region as the EC2 instance**. Cross-region transfer costs money and is slow.
4. Block all public access: leave enabled.
5. Versioning: off (embeddings are large and immutable; versioning would double storage cost).
6. Default encryption: SSE-S3 is fine.
7. Create.

### 2b. Create an IAM user for the Windows PC

The EC2 side will use an instance role (step 3d). Windows needs long-lived credentials.

1. Console → **IAM** → **Users** → **Create user**.
2. Name: `ssae-upload`. Do not attach console access.
3. Permissions: attach **inline policy** with just what you need:

    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Action": ["s3:ListBucket"],
          "Resource": "arn:aws:s3:::<your-bucket-name>"
        },
        {
          "Effect": "Allow",
          "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
          "Resource": "arn:aws:s3:::<your-bucket-name>/*"
        }
      ]
    }
    ```

4. Create the user, then go to **Security credentials** → **Create access key** → **Command Line Interface (CLI)**. Save the access key ID and secret access key. You will not be able to view the secret again.

## 3. Upload embeddings from Windows

### 3a. Install the AWS CLI

Open PowerShell and run:

```powershell
winget install -e --id Amazon.AWSCLI
```

Or download the MSI from `https://aws.amazon.com/cli/`. Verify:

```powershell
aws --version
```

### 3b. Configure credentials

```powershell
aws configure
```

Paste the access key ID and secret from step 2b. Set the default region to match the bucket. Default output format `json` is fine.

Credentials land in `%USERPROFILE%\.aws\credentials`. Treat that file like an SSH key.

### 3c. Upload the split

From the repo root on Windows, `aws s3 sync` transfers only files that changed and is resumable if the connection drops.

```powershell
# from PowerShell in C:\path\to\decoder-only-ssae
aws s3 sync `
    .\results\compositional_split\ `
    s3://<your-bucket-name>/compositional_split/ `
    --exclude "*.pyc" --exclude "__pycache__/*"
```

For a ~500 GB upload on typical home broadband (100 Mbps up), budget 12-15 hours. On a corp line with higher upload, proportionally less. `aws s3 sync` is safe to Ctrl-C and rerun; it picks up where it left off.

If you want to compress before upload (helps if embeddings are stored as loose H5 files that compress well):

```powershell
# optional: tar + gzip the split first, upload the single archive
tar -czf compositional_split.tar.gz results\compositional_split
aws s3 cp compositional_split.tar.gz s3://<your-bucket-name>/
```

Only worth it if the compressed archive is meaningfully smaller than the raw files. H5 embeddings are already float32 and do not compress much.

### 3d. Attach an S3 role to EC2 (do this before launching)

The EC2 instance should read S3 via an IAM role, not the long-lived Windows credentials.

1. Console → **IAM** → **Roles** → **Create role**.
2. Trusted entity: **AWS service** → **EC2**.
3. Permissions: attach the same inline policy as step 2b (or **AmazonS3ReadOnlyAccess** if you only need to download).
4. Name: `ssae-ec2-s3`. Create.

You will select this role when launching the instance in step 4.

## 4. Launch the instance

Console → EC2 → **Launch instance**:

- **AMI**: "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)". CUDA 12.4 driver preinstalled, matches the `torch==2.5.1+cu121` pin in `requirements.txt`.
- **Instance type**: `g6.2xlarge`.
- **Key pair**: create or reuse one. Save the `.pem`.
- **Network settings** → security group: inbound SSH from your IP only.
- **Storage**: gp3, 500 GB, 3000 IOPS (default).
- **Advanced details** → **IAM instance profile**: pick `ssae-ec2-s3` from step 3d.
- Spot vs on-demand: on-demand for benchmark runs. Spot is fine for training-only sweeps that can restart from checkpoint, but the sweep script has no resume path, so stick with on-demand for now.

Launch, note the public DNS.

## 5. First-boot setup

```bash
ssh -i key.pem ubuntu@<public-dns>

# verify GPU visible
nvidia-smi

# activate the preinstalled pytorch env
source activate pytorch

# clone and install
git clone <your-repo-url> decoder-only-ssae
cd decoder-only-ssae
git checkout feature/additional-models
pip install -r requirements.txt

# sanity check
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `pip install` fails on `xformers==0.0.28.post3` against the AMI's torch: drop xformers (SD3.5 with diffusers works without it, just slower) or rebuild the env from scratch with a matching torch build.

## 6. Pull the embeddings from S3

The instance role means no credentials on-box.

```bash
# from ~/decoder-only-ssae on the EC2 box
mkdir -p results/compositional_split
aws s3 sync \
    s3://<your-bucket-name>/compositional_split/ \
    results/compositional_split/
```

S3-to-EC2 in the same region runs at ~1 GB/s on a G6. A 500 GB pull takes ~10 minutes.

If you uploaded a tarball instead:

```bash
aws s3 cp s3://<your-bucket-name>/compositional_split.tar.gz .
tar -xzf compositional_split.tar.gz
```

## 7. Run the sweep in a persistent session

```bash
tmux new -s sweep
cd ~/decoder-only-ssae

./scripts/compare_topk.sh \
    --topk 100000,300000 \
    --layers 1,2 --hidden-dim 2048 \
    --head-type block_diagonal \
    --run-image-benchmark --run-train-reconstruction \
    --locality-drop --locality-swap --benchmark-dino

# Ctrl-b d to detach
# `tmux attach -t sweep` to reattach after reconnecting
```

`tmux` (or `screen`) is important. Without it, an SSH drop kills the whole sweep.

Monitor progress in another shell:

```bash
watch -n 30 'ls -la results/topk_sweep/; nvidia-smi'
```

## 8. Ship results back

Push results (checkpoints, bench outputs, summary CSV) back to S3 when done:

```bash
aws s3 sync results/topk_sweep/     s3://<your-bucket-name>/runs/topk_sweep/
aws s3 sync results/bench_out/      s3://<your-bucket-name>/runs/bench_out/
aws s3 sync results/bench_baseline_cache/ s3://<your-bucket-name>/runs/bench_baseline_cache/
```

Then pull to Windows for local analysis:

```powershell
aws s3 sync s3://<your-bucket-name>/runs/ .\results\aws_runs\
```

## 9. Housekeeping

- **Stop, do not terminate** the instance between sessions if you will come back within a few days. Stopped EC2 costs only EBS (~$40/mo for a 500 GB gp3 volume).
- **Snapshot the volume** once your env and data are staged, so you can rebuild quickly on a bigger box or a spot fleet.
- **Terminate the instance and delete the volume + snapshot** when the paper is out. This is where forgotten spend hides.
- **S3 lifecycle rule**: consider a rule that transitions older run artifacts to S3 Glacier Instant Retrieval after 30 days. Cuts storage cost ~4x.

## Things you must do yourself

- Quota request (needs your console).
- Launching the EC2 instance and generating the key pair.
- `aws configure` on Windows (interactive prompt for the secret key).
- SSHing in the first time to accept the host key.

Everything else can be driven from a Claude session on the EC2 box once you paste the SSH command in.
