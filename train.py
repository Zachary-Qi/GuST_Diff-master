import os
import yaml
import shutil
import torch
import datetime
import numpy as np

from types import SimpleNamespace
from argparse import ArgumentParser

from lib.data_processing import get_target_data
from models.model import GuSTDiff
from lib.enginer import train, evaluate


def load_config_as_namespace(cfg_path: str) -> SimpleNamespace:
    with open(cfg_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    # config 里必须指定 mode: "train" or "test"
    if "mode" not in raw_cfg:
        raise ValueError("Config must contain a 'mode' field: 'train' or 'test'.")

    # 保存配置文件路径，后面可能会备份
    raw_cfg["config_path"] = cfg_path
    return SimpleNamespace(**raw_cfg)


def _prepare_run_folder(args: SimpleNamespace) -> str:
    """
    负责决定本次运行写到哪个 foldername。

    规则：
    1. mode == "train" AND args.modelfolder == ""
       -> 全新实验:
          新建一个时间戳目录，并把 config 备份进去

    2. mode == "train" AND args.modelfolder != ""
       -> 断点续训:
          直接复用 ./save/<modelfolder>/ 作为 foldername
          (不新建，不备份 config，因为这个目录应当已经有老的 config_used.yaml、
           best_model.pth、checkpoint_latest.pth 等信息)

    3. mode == "test"
       -> 我们通常希望单独保存本次评估的指标，不污染训练目录，
          所以像“全新实验”一样给它创建一个带 timestamp 的新目录，
          并备份本次使用的配置。
          （如果你想复用原目录，也可以改成和2一样的逻辑。）
    """
    base_save_dir = "./save"

    # 情况 2: 断点续训
    if args.mode == "train" and args.modelfolder != "":
        reuse_dir = os.path.join(base_save_dir, args.modelfolder)
        if not os.path.isdir(reuse_dir):
            raise RuntimeError(
                f"Expected to resume training in {reuse_dir}, "
                f"but that directory does not exist."
            )
        # 直接返回已有目录
        return reuse_dir

    # 情况 1 或 情况 3: 新目录
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = os.path.join(
        base_save_dir,
        f"{args.dataset_name}_{args.missing_pattern}_{current_time}"
    )
    os.makedirs(foldername, exist_ok=True)

    # 把本次使用的配置文件备份进去，方便复现
    try:
        shutil.copy2(args.config_path, os.path.join(foldername, "config_used.yaml"))
    except Exception as e:
        print(f"[WARN] failed to back up config: {e}")

    return foldername


def main(args: SimpleNamespace):
    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # 准备输出目录（可能是新的run目录，也可能是已有的断点目录）
    foldername = _prepare_run_folder(args)
    print(f"[run] folder = {foldername}")

    # data
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_target_data(args)
    scaler = scaler.to(device).float()
    mean_scaler = mean_scaler.to(device).float()

    # model
    model = GuSTDiff(args).to(device)

    # =============== MODE: TRAIN ===============
    if args.mode == "train":
        # 判断是全新训练还是断点续训
        if args.modelfolder == "":
            # 全新训练
            train(
                model,
                args,
                train_loader,
                valid_loader=valid_loader,
                foldername=foldername,
                resume_ckpt_path=None,
            )
        else:
            # 断点续训
            resume_dir = os.path.join("./save", args.modelfolder)
            latest_ckpt = os.path.join(resume_dir, "checkpoint_latest.pth")

            if os.path.isfile(latest_ckpt):
                print(f"[info] resume training from {latest_ckpt}")
                train(
                    model,
                    args,
                    train_loader,
                    valid_loader=valid_loader,
                    foldername=foldername,          # 现在是原目录
                    resume_ckpt_path=latest_ckpt,
                )
            else:
                raise RuntimeError(
                    f"mode=train but no checkpoint_latest.pth found in {resume_dir}"
                )

        # 训练(或续训)完成后：评估一次，评估结果也写到 foldername 里面
        evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
        )
        return

    # =============== MODE: TEST ===============
    elif args.mode == "test":
        # 只评估，不训练
        if args.modelfolder == "":
            raise RuntimeError(
                "mode=test but args.modelfolder is empty; "
                "please set modelfolder to an existing run dir under ./save"
            )

        resume_dir   = os.path.join("./save", args.modelfolder)
        best_model   = os.path.join(resume_dir, "best_model.pth")
        latest_ckpt  = os.path.join(resume_dir, "checkpoint_latest.pth")
        final_model  = os.path.join(resume_dir, "model.pth")

        # 优先加载验证最优
        if os.path.isfile(best_model):
            print(f"[info] loading BEST model (val best): {best_model}")
            # best_model.pth 是 torch.save(model.state_dict(), ...)
            # -> 只有权重张量，安全，可以 weights_only=True
            state_dict = torch.load(
                best_model,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)

        elif os.path.isfile(latest_ckpt):
            print(f"[info] loading checkpoint_latest for test: {latest_ckpt}")
            # checkpoint_latest.pth 是完整训练快照(dict包含optimizer等)
            # -> 我们需要里面的 "model_state"，要 weights_only=False
            ckpt = torch.load(
                latest_ckpt,
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(ckpt["model_state"])

        elif os.path.isfile(final_model):
            print(f"[info] loading final model weights for test: {final_model}")
            # model.pth 也是 torch.save(model.state_dict(), ...)
            state_dict = torch.load(
                final_model,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)

        else:
            raise RuntimeError(
                f"mode=test but no usable weights found in {resume_dir} "
                f"(expected best_model.pth or checkpoint_latest.pth or model.pth)"
            )

        # 评估结果写到我们这次 _prepare_run_folder() 创建的新目录 foldername
        # （这个目录可能与 resume_dir 相同吗？不会。
        #  对 test 来说我们总是新建一个 fresh foldername 用于保存评估输出，
        #  避免污染训练日志。你也可以改成复用 resume_dir，如果你真的想覆盖。）
        evaluate(
            model,
            test_loader,
            nsample=args.nsample,
            scaler=scaler,
            mean_scaler=mean_scaler,
            foldername=foldername,
        )
        return

    # =============== MODE: OTHER ===============
    else:
        raise ValueError(f"Unsupported mode {args.mode!r}, expected 'train' or 'test'")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config, e.g. configs/metrla_point.yaml",
    )
    cli_args = parser.parse_args()

    args = load_config_as_namespace(cli_args.config)
    main(args)