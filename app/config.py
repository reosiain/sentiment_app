import pathlib

source_root = pathlib.Path(__file__).parent.parent

# backend.transformer_model
sentiment_model = source_root / "source/lm_model"
tokenizer_model = "cointegrated/LaBSE-en-ru"
token_max_len = 100
