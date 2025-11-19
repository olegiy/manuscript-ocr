from manuscript.recognizers import TRBA

if __name__ == "__main__":
    train_summary = TRBA.train(
        train_csvs=[
            r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\train\labels.csv"
        ],
        train_roots=[
            r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\train\img", 
        ],
        val_csvs=[
            r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\labels.csv"
        ],
        val_roots=[
            r"C:\Users\USER\Desktop\archive_25_09\dataset\printed\val\img",
        ],
        exp_dir="experiments/trba_exp_printed_lite_384",
        max_len=40,
        img_h=32,
        img_w=384,
        hidden_size=256,
        num_encoder_layers=2,
        batch_size=16,
        cnn_backbone="seresnet31lite",
        epochs=110,
        lr=0.0005,
        optimizer="AdamW",
        scheduler="OneCycleLR",
        device="cuda",
        pretrain_weights=True,
        val_interval=1,
    )

    print(train_summary)
