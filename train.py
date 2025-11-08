import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset,causal_mask
from model_code import build_transformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from config import get_config,get_weights_file_path
from  tqdm import tqdm
import os

'''
** Decoding loop---> basically we compute the encoder output once
** Pass the ecnoder output to decoder for all sequences in a loop
** we compute encoder output just once
** for decoder --> predict the next token and pass it back to the decoder inputs
** start with SOS token
'''

def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
     sos_idx=tokenizer_tgt.token_to_id('[SOS]')
     eos_idx=tokenizer_tgt.token_to_id('[EOS]')

     #Precompute the encoder output and reuse it for every token we get from decoder
     encoder_output=model.encode(source,source_mask) ## mask needed for masking pad tokens

     # initialize the decoder input
     decoder_input=torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # [1,1] size
     
     while True:
          if decoder_input.size(1) == max_len: ## basically this is the maximum length output we can output
               break 
          # build the decoder mask (decoder mask) --> used in self attention --> causal mask
          decoder_mask=causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

          # output of the decoder
          out=model.decode(encoder_output,source_mask,decoder_input,decoder_mask) ## (B,seq,d_model)

          # get the next token --> which would be the last predicted token --> projecting to vocabular to take argmax
          prob=model.project(out[:,-1])  ## (B,seq,vocab_size)

          # select the next token with max probability ( greedy search)
          _,next_word=torch.max(prob,dim=1) ## this will give the index of next word in the vocabulary
          decoder_input=torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim=1)

          if next_word==eos_idx:
               break
          #print(f"this is the length of decoder input:{decoder_input.size()}") --> use to see how input is getting appended

     return decoder_input.squeeze(0) ## because we are using batch ==1 so we can remove that     

'''
validation loop
'''
def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len, device,print_msg,global_state,writer,num_examples=2):
     model.eval()
     count=0

     source_texts=[]
     expected=[]
     predicted=[]

     ## size of control window 
     console_width=80

     with torch.no_grad():
          for batch in validation_ds:
               count+=1
               encoder_input=batch['encoder_input'].to(device)
               encoder_mask=batch['encoder_mask'].to(device)

               assert encoder_input.size(0)==1 ## batch should be 1 for validation

               print(f"reached here going into the model for decoding")

               model_out=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)

               print(f"reached here going after the model for decoding")

               source_text=batch['src_text'][0]
               target_text=batch['tgt_text'][0]
               model_out_text=tokenizer_tgt.decode(model_out.detach().cpu().numpy())

               source_texts.append(source_text)
               expected.append(target_text)
               predicted.append(model_out_text)

               # printing to the console ( using print_msg because we are using tqdm)
               print_msg('-'*console_width)
               print_msg(f'SOURCE: {source_text}')
               print_msg(f'TARGET: {target_text}')
               print_msg(f'PREDICTED: {model_out_text}')


               if count==num_examples:
                    break

'''
** defining the function to get all items from the sentences(ds)
** lang is basically which language i want --> english and spanish 
** will build a tokenizer separately for both from them
'''
def get_all_sentences(ds,lang):
     for item in ds:
          yield item['translation'][lang]
          
'''
config -> configuration of our model
df--> dataset
lang --> language 
Building the tokenizer--> which will help us to map words in a sentences to tokens(ID's)
'''
def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenizer_path=Path(config['tokenizer_file'].format(lang)) ## path where i am going to save my tokenizer, we assume config['tokenizer] is the path 
    if not Path.exists(tokenizer_path):
         tokenizer=Tokenizer(WordLevel(unk_token="[UNK]")) ## build an empty tokenizer with no vocabulary 
         tokenizer.pre_tokenizer=Whitespace() # before tokenizing i want to split words in a sentence uisng white spaces
         trainer=WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2) # definining how to build the tokenizer-> id for using during tokenization 
         tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer) # now build the tokenizer, by first converting corpus to id (using trainer)
         tokenizer.save(str(tokenizer_path))
    else:
         tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer          

'''
** Loading the dataset from hugging face
** Built the source and target tokenizers
** Next split it into target and validation sets

'''

def get_ds(config):
     ds_raw=load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}',split='train') ## loading the dataset from hugging face

     # building tokenizers for source and target language
     tokenizer_src=get_or_build_tokenizer(config,ds_raw,config['lang_src'])
     tokenizer_tgt=get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

     # keep 90% for training and 10% for validation
     train_ds_size=int(0.9*len(ds_raw))
     print(f"this is train dataset size:{train_ds_size}")
     val_ds_size=len(ds_raw)-train_ds_size
     train_ds_raw,val_ds_raw=random_split(ds_raw,[train_ds_size,val_ds_size])

     train_ds=BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
     val_ds=BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])


     ## checking for maximum sequence length in the current dataset
     max_len_src=0
     max_len_tgt=0

     for item in ds_raw:
          src_ids=tokenizer_src.encode(item['translation'][config['lang_src']]).ids
          tgt_ids=tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
          max_len_src=max(max_len_src,len(src_ids))
          max_len_tgt=max(max_len_tgt,len(tgt_ids))

     print(f"Maximum length of source sentence: {max_len_src}")     
     print(f"Maximum length of target sentence: {max_len_tgt}")


     train_loader=DataLoader(train_ds,batch_size=config['batch_size'], shuffle=True)
     val_loader=DataLoader(val_ds,batch_size=1,shuffle=False)

     return train_loader,val_loader,tokenizer_src,tokenizer_tgt    


'''
Define/Instantiate the model
choose the configuration paramters with which to build the model
'''

def get_model(config,vocab_src_len,vocab_tgt_len):
     model=build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],config['d_model']) ###--> 
     return model

def train_model(config):
     ## define the device
     device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
     print(f"Using device:{device}")

     ## make the folder for saving checkpoints
     Path(config['model_folder']).mkdir(parents=True,exist_ok=True)

     ## get the data loader
     train_loader,val_loader,tokenizer_src,tokenizer_tgt=get_ds(config)
     # instantiate the model
     model=get_model(config,tokenizer_src.get_vocab_size(),tokenizer_tgt.get_vocab_size()).to(device)

     ## Tensorboard for logging
     writer=SummaryWriter(config['experiment_name'])

     optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

     inital_epoch=0
     global_step=0

     if config['preload']:
          model_filename=get_weights_file_path(config,config['preload'])
          print(f"preloading model {model_filename}")
          state=torch.load(model_filename)
          model.load_state_dict(state['model_state_dict'])
          inital_epoch=state['epoch']+1
          optimizer.load_state_dict(state['optimizer_state_dict'])
          global_step=state['global_step']
     else:
          print('No model to preload, starting from scracth')

     ## define the loss function
     loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'),label_smoothing=0.1).to(device)

     for epoch in range(inital_epoch,config['num_epochs']):
          batch_iterator=tqdm(train_loader,desc=f'Processing epoch {epoch:02d}')
          model.train()

          for batch in batch_iterator:
               encoder_input=batch['encoder_input'].to(device) # (B,seq_len)
               decoder_input=batch['decoder_input'].to(device) # (B,seq_len)
               encoder_mask=batch['encoder_mask'].to(device)   #(B,1,1,seq_len) --> used for masking the pads tokens inside attention matrix
               decoder_mask=batch['decoder_mask'].to(device)   #(B,1,seq_len,seq_len) --> Lower triangular matrix + pad tokens --> causal masking


               ## running inputs tensors through the transfomer
               encoder_output=model.encode(encoder_input,encoder_mask) # (B,seq_len,d_model)
               decoder_output=model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask) # (B,seq_len,seq_len)
               proj_output=model.project(decoder_output) # (B,seq_len,tgt_vocab_size) --> projecting back to the vocabulary

               label=batch['label'].to(device) # (B,seq_len)

               ## loss computation
               #  (B,seq_len,tgt_vocab_size) --> (B*seq_len,tgt_vocab_size) &&& (B,seq_len) --> (B*seq_len)
               # basically, consider each batch as one sentence , each sentence has seq_len token or words --> for each token prediction can be any word in vocab(vocab_size)
               # so now we compare for all tokens/words =(b*seq_len), from all words in the vocabulary , a crossentropy loss
               loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_vocab_size()),label.view(-1))
               batch_iterator.set_postfix({f"loss":f"{loss.item():6.3f}"})

               # Log the loss
               writer.add_scalar('train_loss',loss.item(),global_step)
               writer.flush()

               ## backpropagate the loss
               loss.backward()

               ## update the weights
               optimizer.step()
               optimizer.zero_grad()

               global_step+=1

          ## validating at each epoch now
          run_validation(model,val_loader,tokenizer_src,tokenizer_tgt,config['seq_len'],device,lambda msg:batch_iterator.write(msg),global_step,writer)

          ## saving the model at end of every epoch
          model_filename=get_weights_file_path(config,f'{epoch:2d}') 
          torch.save({
               'epoch':epoch,
               'model_state_dict':model.state_dict(),
               'optimizer_state_dict':optimizer.state_dict(),
               'global_step':global_step
          },model_filename)

if __name__=='__main__':
     config=get_config()
     train_model(config)