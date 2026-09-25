import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, RichProgressBar, ModelCheckpoint


def build_callbacks(cfg, logger=None, phase='test', **kwargs):
    callbacks = []
    logger = logger

    # Rich Progress Bar
    callbacks.append(progressBar())

    # Checkpoint Callback
    if phase == 'train':
        callbacks.extend(getCheckpointCallback(cfg, logger=logger, **kwargs))

    return callbacks

def getCheckpointCallback(cfg, logger=None, **kwargs):
    callbacks = []
    # Logging
    metric_monitor = {
        "loss_total": "total/train",
        "r_f": "recons/feature/train",
        "m_loss": "motion/loss/train",
        "t_loss": "text/loss/train",
        "r_f_h": "recons/feature/hand/train",
        "r_f_b": "recons/feature/body/train",
        "expression": "recons/expression/train",
        "position": "recons/position/train",
        "acc": "recons/acceleration/train",
        "angle": "recons/hand/angle/train",
        "semantic": "semantic/commit/train",
        "commit": "vq/commit/train",

        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "FID": "Metrics/FID",
        "gt_FID": "Metrics/gt_FID",
        "Diversity": "Metrics/Diversity",
        "DTW_avg": "Metrics/DTW_avg",
        "DTW_body": "Metrics/DTW_body",
        "DTW_hand": "Metrics/DTW_hand",
        "Test_DTW_avg": "TestMetrics/DTW_avg",
        "Test_DTW_body": "TestMetrics/DTW_body",
        "Test_DTW_hand": "TestMetrics/DTW_hand",
        "Test_ROUGE_L": "TestMetrics/ROUGE_L",
        "Test_Bleu_4": "TestMetrics/Bleu_4",
        "Accuracy": "Metrics/accuracy",
        "WER": "Metrics/WER",
        "ROUGE_L": "Metrics/ROUGE_L",
        "Bleu_4": "Metrics/Bleu_4",
        "MPJPE": "Metrics/MPJPE",
        "PAMPJPE": "Metrics/PAMPJPE",
        "ACCEL": "Metrics/ACCEL",
        "Body_DTW": "Metrics/Body_DTW",
        "Hand_DTW": "Metrics/Hand_DTW",
        "AvgDTW": "Metrics/AvgDTW",
    }
    callbacks.append(
        progressLogger(logger,metric_monitor=metric_monitor,log_every_n_steps=1))

    # Track the latest training checkpoint.
    checkpointParams = {
        'dirpath': os.path.join(cfg.FOLDER_EXP, "checkpoints"),
        'filename': "{epoch}",
        'monitor': "step",
        'mode': "max",
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
        'save_last': True,
        'save_on_train_epoch_end': False
    }
    callbacks.append(ModelCheckpoint(**checkpointParams))

    # Save checkpoint every n*10 epochs
    checkpointParams.update({
        'every_n_epochs':
        cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k':
        -1,
        'save_last':
        False
    })
    callbacks.append(ModelCheckpoint(**checkpointParams))

    metrics = cfg.METRIC.TYPE
    metric_monitor_map = {
        'TemosMetric': {
            'Metrics/APE_root': {
                'abbr': 'APEroot',
                'mode': 'min'
            },
        },
        'TM2TMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            },
            'Metrics/R_precision_top_3': {
                'abbr': 'R3',
                'mode': 'max'
            }
        },
        'MRMetrics': {
            'Metrics/MPJPE': {
                'abbr': 'MPJPE',
                'mode': 'min'
            }
        },
        'HUMANACTMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UESTCMetrics': {
            'Metrics/Accuracy': {
                'abbr': 'Accuracy',
                'mode': 'max'
            }
        },
        'UncondMetrics': {
            'Metrics/FID': {
                'abbr': 'FID',
                'mode': 'min'
            }
        }
    }

    checkpointParams.update({
        'every_n_epochs': cfg.LOGGER.VAL_EVERY_STEPS,
        'save_top_k': 1,
    })

    for metric in metrics:
        if metric in metric_monitor_map.keys():
            metric_monitors = metric_monitor_map[metric]

            # Delete R3 if training VAE
            if cfg.TRAIN.STAGE == 'vae' and metric == 'TM2TMetrics':
                del metric_monitors['Metrics/R_precision_top_3']

            for metric_monitor in metric_monitors:
                checkpointParams.update({
                    'filename':
                    metric_monitor_map[metric][metric_monitor]['mode']
                    + "-" +
                    metric_monitor_map[metric][metric_monitor]['abbr']
                    + "{ep}",
                    'monitor':
                    metric_monitor,
                    'mode':
                    metric_monitor_map[metric][metric_monitor]['mode'],
                })
                callbacks.append(
                    ModelCheckpoint(**checkpointParams))
    return callbacks

class progressBar(RichProgressBar):
    def __init__(self, ):
        super().__init__()

    def get_metrics(self, trainer, model):
        # Don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        # Suppress zero-valued metrics so the progress bar stays compact.
        return {k: v for k, v in items.items() if not (isinstance(v, float) and v == 0.0)}

class progressLogger(Callback):
    def __init__(self,
                 logger,
                 metric_monitor: dict,
                 precision: int = 3,
                 log_every_n_steps: int = 1):
        # Metric to monitor
        self.logger = logger
        self.metric_monitor = metric_monitor
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule,
                       **kwargs) -> None:
        self.logger.info("Training started")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule,
                     **kwargs) -> None:
        self.logger.info("Training done")

    def on_validation_epoch_end(self, trainer: Trainer,
                                pl_module: LightningModule, **kwargs) -> None:
        if trainer.sanity_checking:
            self.logger.info("Sanity checking ok.")
            return

        metric_format = f"{{:.{self.precision}e}}"
        train_metrics_str = []
        test_metrics_str = []
        losses_dict = trainer.callback_metrics
        for metric_name, dico_name in self.metric_monitor.items():
            if dico_name in losses_dict:
                metric = losses_dict[dico_name].item()
                if metric == 0.0:
                    continue
                metric = metric_format.format(metric)
                metric = f"{metric_name} {metric}"
                if dico_name.startswith("TestMetrics/"):
                    test_metrics_str.append(metric)
                else:
                    train_metrics_str.append(metric)
        if train_metrics_str:
            if trainer.optimizers:
                opt = trainer.optimizers[0]
                current_lr = opt.param_groups[0]['lr']
                train_metrics_str.append(f"lr {current_lr:.3e}")
            line = f"Epoch {trainer.current_epoch} [Val/train_part]: " + "   ".join(train_metrics_str)
            self.logger.info(line)
        if test_metrics_str:
            line = f"Epoch {trainer.current_epoch} [Val/test]: " + "   ".join(test_metrics_str)
            self.logger.info(line)

    def on_train_epoch_end(self,
                           trainer: Trainer,
                           pl_module: LightningModule,
                           padding=False,
                           **kwargs) -> None:
        metric_format = f"{{:.{self.precision}e}}"
        line = f"Epoch {trainer.current_epoch}"
        if padding:
            line = f"{line:>{len('Epoch xxxx')}}"  # Right padding

        if trainer.current_epoch % self.log_every_n_steps == 0:
            metrics_str = []

            losses_dict = trainer.callback_metrics
            for metric_name, dico_name in self.metric_monitor.items():
                if dico_name in losses_dict:
                    metric = losses_dict[dico_name].item()
                    if metric == 0.0:
                        continue
                    metric = metric_format.format(metric)
                    metric = f"{metric_name} {metric}"
                    metrics_str.append(metric)

            if trainer.optimizers:
                opt = trainer.optimizers[0]
                current_lr = opt.param_groups[0]['lr']
                metrics_str.append(f"lr {current_lr:.3e}")

            line = line + ": " + "   ".join(metrics_str)

        self.logger.info(line)
