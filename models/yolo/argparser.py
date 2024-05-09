import argparse

def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='DOTAv1.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolov8n-obb.yaml', help='initial weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--time', type=int, default=2, help='hours of time to train for')
    parser.add_argument('--img-size', type=int, default=640, help='image sizes')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for data loader')
    parser.add_argument('--project', type=str, default=None, help='save to project/name')
    parser.add_argument('--name', type=str, default=None, help='save to project/name')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--cos_lr', action='store_true', help='use cosine lr scheduler')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')

    parser.add_argument('--degrees', type=float, default=0.0, help='image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='image scale (+/- gain)')
    parser.add_argument('--flipud', type=float, default=0.0, help='image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5, help='image flip left-right (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0, help='image mixup (probability)')


    return parser.parse_args()
