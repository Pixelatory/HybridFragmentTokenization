import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from dataset import Pretrain_Collater, load_encoders, Smiles_Bert_Dataset
from metrics import AverageMeter
from model import BertModel
from device import device

def main(random_state: int, model_save_name: str, fragmentation: bool, two_phase: bool, num_train_epoch: int = 10000, 
         max_testing_not_improved: int = 2, smiles_col: str = "smiles", arch: str = 'medium',
         dropout_rate: float = 0.1, smiles_vocab_path: str = 'vocabs/allmolgen_vocab.txt',
         smiles_frag_vocab_path: str = 'vocabs/allmolgen_frag_smiles_vocab.txt',
         frag_vocab_path: str = None, test_epoch_batch_interval: int = 5000, num_workers: int = 1):
    """
        Arguments:
        - random_state: Seed value for random number generation
        - model_save_name: Save name for model log file and weights
        - fragmentation: use hybrid fragment-SMILES tokenization (True) or not (False)
        - two_phase: two-phase pretraining (True) or one-phase pretraining (False)
        - num_train_epoch: amount of training epochs (can set to high value and training stops when testing loss does not improve)
        - dropout_rate: dropout rate for MTL-BERT model
        - smiles_frag_vocab_path: vocab path that includes SMILES linking tokens for hybrid fragmentization
        - max_testing_not_improved: number of test epochs without improvement before stopping the training process
        - smiles_col: smiles column from pretraining CSV file
        - num_workers: set number of processes for train and test dataloaders
        - test_epoch_batch_interval: number of training batches before performing test epoch
    """
    if fragmentation and frag_vocab_path is None:
        raise Exception("If using hybrid fragmentation, need to provide fragment vocabulary path")
    
    if not fragmentation:
        frag_vocab_path = None

    small = {'name': 'small', 'num_layers': 4, 'num_heads': 4, 'd_model': 128, 'path': 'small_weights'}
    medium = {'name': 'medium', 'num_layers': 8, 'num_heads': 8, 'd_model': 256, 'path': 'medium_weights'}
    large = {'name': 'large', 'num_layers': 12, 'num_heads': 12, 'd_model': 576, 'path': 'large_weights'}

    if arch.lower() == 'small':
        arch = small
    elif arch.lower() == 'medium':
        arch = medium
    elif arch.lower() == 'large':
        arch = large
    else:
        raise Exception(f"Unknown arch ({arch.lower()}). Selection is 'small', 'medium', or 'large'")

    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    dff = d_model * 4
    
    model_str2num, smiles_str2num, frag_str2num = load_encoders(fragmentation, 
                                                                smiles_vocab_path=smiles_vocab_path,
                                                                smiles_frag_vocab_path=smiles_frag_vocab_path,
                                                                frag_vocab_path=frag_vocab_path)

    if fragmentation:
        vocab_size = len(model_str2num) + len(smiles_str2num) + len(frag_str2num)
    else:
        vocab_size = len(model_str2num) + len(smiles_str2num)

    if not fragmentation:
        # can't do 2 phase if you're not fragmenting molecules
        two_phase = False

    model = BertModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, 
                      vocab_size=vocab_size, maximum_positional_encoding=200, dropout_rate=dropout_rate)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), 1e-4, betas=(0.9, 0.98))

    loss_func = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    train_loss = AverageMeter()
    train_acc = AverageMeter()
    test_loss = AverageMeter()
    test_acc = AverageMeter()

    def train_step(x, y, weights):
        model.train()
        optimizer.zero_grad()
        predictions = model(x)
        loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum() / weights.sum()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.detach().item(), x.shape[0])
        train_acc.update(((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item() / weights.cpu().sum().item(),
                         weights.cpu().sum().item())


    def test_step(x, y, weights):
        model.eval()
        with torch.no_grad():
            predictions = model(x)
            loss = (loss_func(predictions.transpose(1, 2), y) * weights).sum() / weights.sum()

            test_loss.update(loss.detach().item(), x.shape[0])
            test_acc.update(
                ((y == predictions.argmax(-1)) * weights).detach().cpu().sum().item() / weights.cpu().sum().item(),
                weights.cpu().sum().item())


    if two_phase:
        phases = 2
    else:
        phases = 1

    
    log_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    with open(f'logs/pretrain/{log_time}_{model_save_name}-pretrain-log.txt', 'a') as f:
        f.write(f"fragmentation: {fragmentation}\n")
        f.write(f"two phase: {two_phase}\n")
        f.write(f"max testing hasn't improved count: {max_testing_not_improved}\n")
        f.write(f"perform test epoch every {test_epoch_batch_interval} training batches\n")
        f.write(f"frag vocab path: {frag_vocab_path}\n")
        f.write(f"random state: {random_state}\n")
    
    
    for phase in range(phases):
        if phase == 0:
            # first phase
            if two_phase:
                tmp_frag = False
            else:
                tmp_frag = fragmentation

            full_dataset = Smiles_Bert_Dataset('allmolgen_pretrain_data_100maxlen_FIXEDCOLS.csv',
                                               smiles_col=smiles_col, fragmentation=tmp_frag,
                                               model_str2num=model_str2num, smiles_str2num=smiles_str2num,
                                               frag_str2num=frag_str2num)
        else:
            # second phase
            full_dataset = Smiles_Bert_Dataset('allmolgen_pretrain_data_100maxlen_FIXEDCOLS.csv',
                                               smiles_col=smiles_col, fragmentation=True,
                                               model_str2num=model_str2num, smiles_str2num=smiles_str2num,
                                               frag_str2num=frag_str2num)
                                               
        train_size = int(0.8 * len(full_dataset))  # 80-20 train/test split
        test_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(random_state)  # split the same way even for different phases and executions!
        train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=Pretrain_Collater(), num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=Pretrain_Collater(), num_workers=num_workers)

        early_stop_val = float('inf')
        has_not_improved_count = 0
        for epoch in range(num_train_epoch):
            start = time.time()

            test_losses = []
            test_accs = []

            train_losses = []
            train_accs = []

            for (batch, (x, y, weights)) in enumerate(tqdm(train_dataloader)):
                x = x.to(device)
                y = y.to(device)
                weights = weights.to(device)

                train_step(x, y, weights)

                if batch % 500 == 0:
                    train_losses.append(train_loss.avg)
                    train_accs.append(train_acc.avg)
                    print('Epoch {} Batch {} training Loss {:.4f}'.format(
                        epoch + 1, batch, train_loss.avg))
                    print('training Accuracy: {:.4f}'.format(train_acc.avg))

                if batch % test_epoch_batch_interval == 0:
                    for x, y, weights in tqdm(test_dataloader):
                        x = x.to(device)
                        y = y.to(device)
                        weights = weights.to(device)
                        test_step(x, y, weights)
                    print('Test loss: {:.4f}'.format(test_loss.avg))
                    print('Test Accuracy: {:.4f}'.format(test_acc.avg))
                    test_losses.append(test_loss.avg)
                    test_accs.append(test_acc.avg)
                    if (test_loss.avg < early_stop_val + 0.001):
                        early_stop_val = test_loss.avg
                        has_not_improved_count = 0
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "random_state": random_state,
                            "dropout_rate": dropout_rate,
                            "training_epoch": epoch,
                            "training_batch": batch,
                            "training_phase": phase,
                            "model_name": model_save_name,
                            "model_arch": arch,
                            "log_time": log_time,
                            "fragmentation": fragmentation,
                            "frag_vocab_path": frag_vocab_path,
                            "smiles_vocab_path": smiles_vocab_path,
                            "smiles_frag_vocab_path": smiles_frag_vocab_path,
                            "max_testing_not_improved": max_testing_not_improved,
                            "test_epoch_batch_interval": test_epoch_batch_interval,
                        }, f'weights/pretrain/{log_time}_{model_save_name}_phase{phase + 1}_best.pt')
                    else:
                        has_not_improved_count += 1

                    print(f"Testing loss hasn't improved in f{has_not_improved_count} test epochs")
                    test_acc.reset()
                    test_loss.reset()
                    train_acc.reset()
                    train_loss.reset()

                if has_not_improved_count >= max_testing_not_improved:
                    break
            print(f"Model name: {model_save_name}")
            print('Epoch {} is Done!'.format(epoch))
            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
            print('Epoch {} Training Loss {:.4f}'.format(epoch + 1, train_loss.avg))
            print('training Accuracy: {:.4f}'.format(train_acc.avg))
            print('Epoch {} Test Loss {:.4f}'.format(epoch + 1, test_loss.avg))
            print('test Accuracy: {:.4f}'.format(test_acc.avg))
            
            with open(f'logs/pretrain/{log_time}_{model_save_name}-pretrain-log.txt', 'a') as f:
                f.write(f"Phase {phase + 1}, Epoch {epoch + 1}\n")

                f.write("Train Losses\n")
                for tmp in train_losses:
                    f.write(str(tmp) + "\n")

                f.write("Train Accuracies\n")
                for tmp in train_accs:
                    f.write(str(tmp) + "\n")

                f.write("Test Losses\n")
                for tmp in test_losses:
                    f.write(str(tmp) + "\n")

                f.write("Test Accuracies\n")
                for tmp in test_accs:
                    f.write(str(tmp) + "\n")
            
            if has_not_improved_count >= max_testing_not_improved:
                break

if __name__ == "__main__":
    # NOTE: device.py needs to be changed, depending on where pretraining is performed

    model_save_name = '500freq_onephase'  # change for different file name when saving model
    num_train_epoch = 1000000  # high value that'll likely never be reached
    fragmentation = True  # True for hybrid SMILES-fragment encodings
    frag_vocab_path = 'vocabs/500freq_vocab.txt'  # change me for different fragment vocabs
    
    max_testing_not_improved = 2
    smiles_col = "smiles"
    arch = "medium"
    dropout_rate = 0.1
    smiles_vocab_path = 'vocabs/allmolgen_vocab.txt'
    smiles_frag_vocab_path = 'vocabs/allmolgen_frag_smiles_vocab.txt'
    test_epoch_batch_interval = 5000
    num_workers = 12  # for dataloader
    random_state = 42

    '''
    Two-phase (True): First trains on smiles, then hybrid fragment-smiles. 
    One-phase (False): Depends on what fragmentation is. 
                       If fragmentation is true, then train only on hybrid fragment-smiles.
                       Otherwise train on only smiles.
    '''
    two_phase = False
    
    main(random_state, model_save_name, fragmentation, two_phase,
            num_train_epoch, max_testing_not_improved, smiles_col, arch,
            dropout_rate, smiles_vocab_path, smiles_frag_vocab_path,
            frag_vocab_path, test_epoch_batch_interval, num_workers)
