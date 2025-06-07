import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Any

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.long)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.long)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.long)
    

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx:Any) -> Any:
        
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Input sequence too long for seq_len={self.seq_len}. "
                             f"Source length: {len(enc_input_tokens)}, Target length: {len(dec_input_tokens)}")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)  
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)  
        ])

        labels = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)  
        ])

        assert encoder_input.shape[0] == self.seq_len, f"Encoder input length mismatch: {encoder_input.shape[0]} != {self.seq_len}"
        assert decoder_input.shape[0] == self.seq_len, f"Decoder input length mismatch: {decoder_input.shape[0]} != {self.seq_len}"
        assert labels.shape[0] == self.seq_len, f"Labels length mismatch: {labels.shape[0]} != {self.seq_len}"

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]), #(1, 1, seq_len) & (1, seq_len, seq_len)   
            'labels': labels,
            'src_text': src_text,
            'tgt_text': tgt_text    
        }
def causal_mask(size):
    mask = torch.triu(torch.ones(1,size, size), diagonal=1).type(torch.int)
    return mask == 0  # Add batch and head dimensions
      