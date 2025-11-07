import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import dataloader,Dataset

class BilingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len)-> None:
        super().__init__()

        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.seq_len=seq_len

        self.sos_token=torch.tensor([tokenizer_src.token_to_id['SOS']],dtype=torch.int64)
        self.eos_token=torch.tensor([tokenizer_tgt.token_to_id['EOS']],dtype=torch.int64)
        self.pad_token=torch.tensor([tokenizer_src.token_to_id['PAD']],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair=self.ds[idx] ## getting pair according to the idx
        
        ## getting src text and tgt text from each sentence
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]

        ## splitting sentence to word --> word to ID's (tokenization process)
        enc_input_tokens=self.tokenizer_src.encode(src_text).ids ## this basically gives me a list of tokens
        dec_input_tokens=self.tokenizer_tgt.encode(tgt_text).ids ## ids because this will give me a list of ids

        ## now i need to pad all the tokens to the sequence length
        encoder_num_padding_tokens=self.seq_len-len(enc_input_tokens)-2 ## -2 because we will be adding SOS and EOS as well
        decoder_num_padding_tokens=self.seq_len-len(dec_input_tokens)-1 ## -1 because for decoder we only add SOS and not EOS

        if encoder_num_padding_tokens<0 or decoder_num_padding_tokens<0: ## basically the length of sentence is greater than sequence length
            raise ValueError("Sentence is too long")
        
        ## build the encoder, decoder, labels by padding tokens,SOS, EOS
        encoder_input=torch.cat(                                        # input to cat --> a list of tensors
            [
            self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]*encoder_num_padding_tokens,dtype=torch.int64)
            ]
        )

        ## Adding only SOS to the decoder input
        decoder_input=torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token]*decoder_num_padding_tokens,dtype=torch.int64)
            ]
        )

        ## Adding only EOS to the label ( what we expect from the decoder output)
        label=torch.cat(
            [
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*decoder_num_padding_tokens,dtype=torch.int64)
            ]
        )
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0)== self.seq_len
        assert label.size(0)==self.seq_len

        return{
            "encoder_input":encoder_input, # (Seq_len)
            "decoder_input":decoder_input, #(Seq_len)
            "encoder_mask": (encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1,1,seq_len) --> only paying attention to the encoder_input and not the pad_tokens
            "decoder_mask": (decoder_input!=self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1,seq_len) & (1,seq_len, seq_len)
            "label":label,
            "src_text":src_text,
            "tgt_text":tgt_text,
        }   

def causal_mask(size): 
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int) ## diagnol is 1 and not 0, because in decoder we pay attention to all tokens less than current token and ITSELF AS WELL 
    return mask==0