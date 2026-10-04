# BiasPruner on HAM10000

A modular PyTorch implementation of BiasPruner: Debiased Continual Learning for Medical Image Classification for the HAM10000 age-bias experiment. It uses only locally mounted HAM10000 data; no download or dataset copy is included.

**This is a HAM10000 implementation/evaluation of BiasPruner, not a complete reproduction of the paper's FITZ + HAM + NIH experiments.**

## Implementation plan

1. Read HAM metadata and locate images; create reproducible, lesion-disjoint train/validation/test partitions.
2. Build the paper's three non-overlapping class-incremental tasks with 2, 2, and 3 diagnosis classes.
3. Train a ResNet-50 with GCE, identify easy/hard samples, score bottleneck channel units, prune the highest-bias gamma fraction, then fine-tune using weighted cross entropy.
4. Freeze previous task parameters and preserve task channel masks while later tasks learn with free channels.
5. Evaluate with max-output task selection and report classification and age-group fairness metrics.

## Paper-aligned components

- HAM10000 has 7 classes, three tasks with [2,2,3] classes, and sensitive groups age <60 / age >=60.
- ResNet-50 backbone; Adam; GCE with q=0.7; easy threshold tau=0.70; prune fraction gamma=0.60; WCE fine-tuning for 20 epochs.
- GCE is (1 - p_y^q)/q; easy items are correctly classified with p_y >= tau. Hard items are misclassified, following the method text's prose. The paper's displayed hard-set notation is ambiguous. The public authors' helper named GeneralizedCELoss instead computes a CE-times-p_y^q surrogate, so this project follows the paper's stated GCE equation and records that code discrepancy.
- Bias score compares average per-image spatial variance of ReLU feature maps for easy versus hard examples, class-by-class, then averages class scores.
- WCE uses exp(alpha * cached GCE) * CE. The paper text calls alpha trainable, while the authors' released WCE class stores alpha as a fixed scalar and reads cached per-image GCE values. This implementation follows the released loss behavior: alpha is fixed and configurable.
- Earlier task subnetworks are frozen; inference selects the task with maximum output response, without a task ID.

## Explicit implementation decisions / limits

The paper does not prescribe a specific class order, lesion-level splitting rule, or numeric alpha. Its supplement reports HAM per-task train/validation/test counts of 519/85/167, 731/123/240, and 4,441/809/1,563, corresponding approximately to a 65/12/23 split on its 8,678-image experimental subset. This implementation uses those proportions as defaults but applies them to all locally available HAM rows with lesion-disjoint splits. The supplied material does not explain the 8,678-row subset selection or provide per-image split IDs, so exact counts/assignments are not fabricated. The default class order is canonical HAM metadata order (akiec,bcc,bkl,df,mel,nv,vasc), split into groups of 2/2/3; it can be changed with --task-order. Splits use seed 42. The released WCE code confirms fixed alpha and cached per-image GCE but does not expose a single experiment alpha value in the public runner; alpha therefore defaults to configurable 0.5 and is not learned. ImageNet pretraining is off by default because the paper does not specify it and HAM10000 remains the only training data; --pretrained is available if desired.

The reference paper scores/prunes units through the network. This implementation applies binary masks at the input stem and each bottleneck's conv1, conv2, and residual output activations (post-ReLU), computes activation-variance scores for each such unit, and removes the top gamma fraction globally while retaining one channel in each masked tensor. The ResNet-50 remains fixed-shape rather than physically compacting each convolution, which avoids changing residual wiring and keeps the run stable on an 8 GB GPU. Masks represent each task's active subnetwork; gradients for previously allocated unit filters, their BatchNorm affine values/statistics, and heads are protected. BatchNorm running statistics are frozen after task 1 to keep protected representations stable; free-unit convolution and affine parameters remain trainable. New tasks train genuinely unallocated units and can reuse surviving frozen units from prior tasks. Classes without both easy and hard examples are excluded from the class-average score because their score is undefined. Runtime assertions verify protected parameter slices remain unchanged.

Fairness is calculated across the two observed age groups, excluding missing-age rows from fairness metrics but retaining them for diagnosis training. The authors' released metric code uses local class index 1 as the binary positive class, computes DPR as the ratio of the lower to higher group selection rate, and computes TPR gap; this implementation follows that convention within each task and averages task DPR/EOD. Accuracy, weighted F1 (matching their runner), and balanced accuracy are also macro-averaged across observed tasks, with overall pooled classification scores included as global_* fields. GCE checkpointing uses validation accuracy, matching the released trainer. WCE checkpointing exposes a priority rule because “highest ACC and EOD” does not define a joint selection rule; default is lexicographic balanced accuracy first, then lower EOD, with the reverse priority also configurable. Max-output task selection runs independently per image (the paper's displayed equation aggregates a batch). These choices are documented deviations/clarifications. The authors' implementation is available at [nourhanb/BiasPruner](https://github.com/nourhanb/BiasPruner).

The experiment defaults to the paper's three random task orders and reports mean and standard deviation. Use one order for a quick debugging run.

## Data layout

In Kaggle, attach HAM10000 as an input dataset. The configured root should contain a HAM10000_metadata.csv and the image part directories containing JPG files. The loader searches nested images directories and the root for JPEGs. If Kaggle mounts differently, pass explicit metadata and image locations.

## Kaggle setup and run

1. Make this repository's Python files available in the Kaggle notebook working directory (or attach the repository as a Kaggle dataset), open experiments/ham10000_experiment.ipynb, and enable a GPU accelerator.
2. Attach the HAM10000 dataset to the notebook.
3. Set DATA_ROOT (and optionally METADATA_CSV / IMAGE_DIRS) in the notebook configuration cell.
4. Run the notebook cells. Run the sanity-check cell first; it checks the actual attached images and runs lightweight synthetic forward/loss/mask/metric checks. The final training cell runs the configured three task orders. Set cfg.num_task_orders=1 for quick debugging. The defaults retain 200 GCE epochs (early stopping) and 20 WCE epochs.

Equivalent shell commands from a Kaggle notebook terminal:

    pip install -r requirements.txt
    python sanity_check.py --data-root /kaggle/input/ham10000
    python train.py --data-root /kaggle/input/ham10000 --output-dir results --seed 42 --orders 3

To override class order, provide all seven labels, e.g.:

    python train.py --task-order mel nv bcc akiec bkl df vasc

Main options and hyperparameters are in config.py; training CLI overrides include --gce-epochs, --finetune-epochs, --batch-size, --image-size, --workers, --seed, --output-dir, --task-order, --orders, --alpha, and --checkpoint-priority.

## Outputs

- order_NN/checkpoints/task_N.pt and order_NN/checkpoints/final.pt: model weights, task masks, config.
- order_NN/logs/config.json, training.jsonl, evaluation.json: settings, per-epoch losses/metrics, final test metrics.
- summary.json: mean and standard deviation over task orders.

## Local checks and unverified items

Python syntax, notebook JSON/code cells, config validation, and the metric helpers were checked locally. The local environment does not have PyTorch installed, so runtime model/loss checks and the synthetic sanity suite could not be completed here. End-to-end image loading, GPU memory use, and measured model quality must be verified in Kaggle with the attached dataset and GPU. Training is not launched by repository setup.

## Citation

Bayasi et al., “BiasPruner: Debiased Continual Learning for Medical Image Classification,” MICCAI 2024. [Open access paper](https://papers.miccai.org/miccai-2024/099-Paper2799.html).
