from manuscript.recognizers import TRBA

if __name__ == "__main__":
    train_summary = TRBA.train(
        train_csvs=[r"C:\shared\orig_cyrillic\train.tsv"],
        train_roots=[
            r"C:\shared\orig_cyrillic\train",
        ],
        val_csvs=[r"C:\shared\orig_cyrillic\test.tsv"],
        val_roots=[
            r"C:\shared\orig_cyrillic\test",
        ],
        exp_dir="experiments/trba_exp_printed_lite_bpe_3",
        max_len=40,
        img_h=32,
        img_w=256,
        hidden_size=256,
        num_encoder_layers=2,
        batch_size=64,
        cnn_backbone="seresnet31lite",
        epochs=110,
        lr=0.001,
        optimizer="AdamW",
        scheduler="OneCycleLR",
        device="cuda",
        pretrain_weights=True,
        val_interval=1,
        bpe_vocab_size=1024,
    )

    print(train_summary)
