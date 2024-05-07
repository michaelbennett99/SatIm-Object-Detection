from argparser import parse_training_args

from ultralytics import YOLO

def main(args):
    model = YOLO(args.weights)

    model.train(
        data=args.data, time=args.time, batch=args.batch_size,
        imgsz=args.img_size, workers=args.workers, project=args.project,
        name=args.name, pretrained=args.pretrained, cos_lr=args.cos_lr,
        resume=args.resume, fraction=args.fraction, lr0=args.lr0,
        dropout=args.dropout, degrees=args.degrees, translate=args.translate,
        scale=args.scale, flipud=args.flipud, fliplr=args.fliplr,
        mosaic=args.mosaic, mixup=args.mixup
    )

    model.val(save_json=True)

if __name__ == '__main__':
    args = parse_training_args()
    main(args)
