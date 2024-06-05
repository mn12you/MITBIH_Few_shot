import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./mit_bih', help='Directory for data dir')
    parser.add_argument('--seed', type=int, default=42, help="Random seeds")
    parser.add_argument('--model_name', type=str, default="Siamese_Sembed", help="Transform net after distance metric")
    parser.add_argument('--phase', type=str, default="Train", help="Train or Test")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch_size")
    parser.add_argument('--num_workers', type=int, default=1, help="how many cpu cores you want to use")
    parser.add_argument('--use_gpu', type=bool, default=True, help="To use GPU or not")
    parser.add_argument('--lr', type=float, default= 0.00006, help="Learning rate")
    parser.add_argument('--model_path', type=str, default="./model", help="Model saving dir")
    parser.add_argument('--result_path', type=str, default="./result", help="Result saving dir")
    parser.add_argument('--epochs', type=int, default=2500, help="How many epochs to train")
    parser.add_argument('--resume', type=bool, default=False, help="Retrain or not")
    parser.add_argument('--best_metric', type=float, default=100.0, help="Best performance of validation set")
    parser.add_argument('--patience', type=int, default=100, help="How many epoch the model stop getting better")
    parser.add_argument('--test_set', type=str, default="normal", help="Be in test situation")
    return parser.parse_args()