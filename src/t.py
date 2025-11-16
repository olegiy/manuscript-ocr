from manuscript.recognizers import TRBA

if __name__ == "__main__":
    train_summary = TRBA.train(
        train_csvs=[
            r"C:\shared\orig_cyrillic\train.tsv",
        ],
        train_roots=[
            r"C:\shared\orig_cyrillic\train",
        ],
        val_csvs=[
            r"C:\shared\orig_cyrillic\test.tsv",
        ],
        val_roots=[
            r"C:\shared\orig_cyrillic\test",
        ],
        exp_dir="experiments/trba_exp_lite",
        max_len=40,
        img_h=32,
        img_w=128,
        hidden_size=128,
        num_encoder_layers=1,
        batch_size=64,
        cnn_backbone="seresnet31lite",
        epochs=110,
        lr=5e-3,
        optimizer="AdamW",
        scheduler="OneCycleLR",
        device="cuda",
        pretrain_weights=True,
        val_interval=1,
    )

    print(train_summary)
