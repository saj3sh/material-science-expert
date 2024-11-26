from transformers import AutoTokenizer, AutoModel


def get_nomic_embed_text_v1():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained(
        'nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
    return tokenizer, model


def get_matscibert():
    tokenizer = AutoTokenizer.from_pretrained("m3rg-iitd/matscibert")
    model = AutoModel.from_pretrained("m3rg-iitd/matscibert")
    return tokenizer, model
