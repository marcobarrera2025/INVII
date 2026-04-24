from split_dataset import split_by_bag, print_split_counts
from train import train_oneclass
from hard_fake import generate_fake_hard
from evaluate import evaluate_roc_auc

def main():
    print("INVII - Pipeline fiel al notebook")
    print("1. Split por cartera")
    ok_split = split_by_bag()
    if not ok_split:
        print("No se pudo hacer split. Revisa data/genuine/chanel/")
        return

    print_split_counts()

    print("\n2. Entrenamiento One-Class EfficientNet")
    ok_train = train_oneclass()
    if not ok_train:
        print("No se pudo entrenar.")
        return

    print("\n3. Generación de fake hard")
    generate_fake_hard()

    print("\n4. Evaluación ROC-AUC")
    evaluate_roc_auc()

    print("\nPipeline completo terminado.")

if __name__ == "__main__":
    main()
