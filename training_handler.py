"""
training_handler.py

Usage (CLI):
python training_handler.py \
  --train_dir /tmp/train_dir \
  --existing_model path/to/copy_of_best.pt \
  --epochs 50 --batch 4 --imgsz 1280 \
  --lr0 0.001 --lrf 0.01 --optimizer Adam \
  --mosaic 0.5 --degrees 10 --translate 0.1 --scale 0.5 \
  --weight_decay 0.0005 --project /tmp/training_output --name model_run
"""

import argparse
import os
import shutil
import yaml
import traceback
from ultralytics import YOLO

def run_training(train_dir,
                 existing_model,
                 epochs=100,
                 batch=8,
                 imgsz=1280,
                 lr0=0.001,
                 lrf=0.01,
                 optimizer='Adam',
                 weight_decay=0.0005,
                 mosaic=0.5,
                 degrees=10.0,
                 translate=0.1,
                 scale=0.5,
                 fliplr=0.5,
                 box=7.5,
                 cls=0.5,
                 dfl=1.5,
                 project='/tmp/training_output',
                 name='model',
                 device=None,
                 patience=50,
                 save=True,
                 plots=True,
                 exist_ok=True,
                 use_clahe=False):
    """
    Runs Ultralytics YOLO training with the supplied arguments.
    Returns tuple (success_bool, path_to_best_model_or_error_message)
    """
    try:
        data_yaml = os.path.join(train_dir, 'data.yaml')
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

        # Load model
        if existing_model and os.path.exists(existing_model):
            model = YOLO(existing_model)
        else:
            model = YOLO('yolo11s.pt')  # fallback

        train_kwargs = dict(
            data=data_yaml,
            epochs=int(epochs),
            imgsz=int(imgsz),
            batch=int(batch),
            device=device,
            lr0=float(lr0),
            lrf=float(lrf),
            optimizer=str(optimizer),
            weight_decay=float(weight_decay),
            degrees=float(degrees),
            translate=float(translate),
            scale=float(scale),
            fliplr=float(fliplr),
            mosaic=float(mosaic),
            box=float(box),
            cls=float(cls),
            dfl=float(dfl),
            patience=int(patience),
            save=save,
            plots=plots,
            project=project,
            name=name,
            exist_ok=exist_ok,
            verbose=True
        )

        # note: ultralytics.YOLO.train will write logs to stdout which we stream in Streamlit
        results = model.train(**train_kwargs)

        # ultralytics stores outputs under project/name/weights/best.pt
        best_path = os.path.join(project, name, 'weights', 'best.pt')
        if os.path.exists(best_path):
            print(f"TRAINING_SUCCESS: {best_path}")
            return True, best_path
        else:
            # If training finished but no best found, maybe last.pt exists
            last_path = os.path.join(project, name, 'weights', 'last.pt')
            if os.path.exists(last_path):
                print(f"TRAINING_SUCCESS_LAST: {last_path}")
                return True, last_path
            else:
                return False, "Training finished but no best.pt / last.pt found."

    except Exception as e:
        print("TRAINING_ERROR:", e)
        traceback.print_exc()
        return False, str(e)


# CLI entrypoint
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help="Directory containing images/labels and data.yaml")
    p.add_argument("--existing_model", required=False, help="Path to model to use as starting weights")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--lrf", type=float, default=0.01)
    p.add_argument("--optimizer", default="Adam")
    p.add_argument("--weight_decay", type=float, default=0.0005)
    p.add_argument("--mosaic", type=float, default=0.5)
    p.add_argument("--degrees", type=float, default=10.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--cls", type=float, default=0.5)
    p.add_argument("--dfl", type=float, default=1.5)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--project", default="/tmp/training_output")
    p.add_argument("--name", default="model")
    p.add_argument("--device", default=None)
    p.add_argument("--use_clahe", action="store_true")
    args = p.parse_args()

    ok, result = run_training(
        train_dir=args.train_dir,
        existing_model=args.existing_model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        mosaic=args.mosaic,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
        use_clahe=args.use_clahe
    )

    if ok:
        print("EXIT_OK:", result)
    else:
        print("EXIT_FAIL:", result)
        raise SystemExit(1)
