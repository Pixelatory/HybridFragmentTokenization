from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim

from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DisjointDatasetBatchSampler, Prediction_Dataset, Finetune_Collater, kfolds, load_encoders, train_test_splits
from metrics import AverageMeter, Records_R2, Records_AUC, Records_RMSE, Records_Acc
from model import PredictionModel
from device import device

def main(num_folds: int, pretrain_model_weights_file_path: str,
         clf_cols: List[str], reg_cols: List[str],
         max_testing_not_improved: int = 2,
         smiles_col: str = "SMILES",
         fragmentation: bool = None,
         model_save_name: str = None,
         smiles_vocab_path: str = None,
         smiles_frag_vocab_path: str = None,
         frag_vocab_path: str = None, num_workers: int = 1,
         dropout_rate: float = None, random_state: int = None,
         train_ratio: float = None):
    """
    fragmentation: If set to None, uses same fragmentation as pretrained model
    model_save_name: If set to None, uses same model_save_name as pretrained model
    smiles_vocab_path: If set to None, uses same smiles vocab path as pretrained model
    smiles_frag_vocab_path: If set to None, uses same smiles fragment vocab path as pretrained model
    frag_vocab_path: If set to None and fragmentation is true, uses same fragment vocab path as pretrained model
    dropout_rate: If set to None, uses same dropout rate as pretrained model
    random_state: If set to None, uses same random state as pretrained model
    """

    if train_ratio is not None and num_folds is not None:
        raise Exception("train_ratio and num_folds cannot both have values")
    elif train_ratio is None and num_folds is None:
        raise Exception("train_ratio or num_folds must have a non-None value")

    pretrained_weights_file = torch.load(pretrain_model_weights_file_path, map_location='cpu')

    arch = pretrained_weights_file['model_arch']
    num_layers = arch['num_layers']
    num_heads = arch['num_heads']
    d_model = arch['d_model']
    dff = d_model * 4

    if fragmentation is None:
        fragmentation = pretrained_weights_file['fragmentation']

    if model_save_name is None:
        model_save_name = pretrained_weights_file['model_name']

    if smiles_vocab_path is None:
        smiles_vocab_path = pretrained_weights_file['smiles_vocab_path']
    
    if smiles_frag_vocab_path is None:
        smiles_frag_vocab_path = pretrained_weights_file['smiles_frag_vocab_path']
    
    if frag_vocab_path is None:
        frag_vocab_path = pretrained_weights_file['frag_vocab_path']

    if dropout_rate is None:
        dropout_rate = pretrained_weights_file['dropout_rate']
    
    if random_state is None:
        random_state = pretrained_weights_file['random_state']

    model_str2num, smiles_str2num, frag_str2num = load_encoders(fragmentation,
                                                                smiles_vocab_path=smiles_vocab_path,
                                                                smiles_frag_vocab_path=smiles_frag_vocab_path,
                                                                frag_vocab_path=frag_vocab_path)

    if fragmentation:
        vocab_size = len(model_str2num) + len(smiles_str2num) + len(frag_str2num)
    else:
        vocab_size = len(model_str2num) + len(smiles_str2num)

    generator = np.random.default_rng(random_state)

    reg_means = []
    reg_stds = []

    dfs = []
    columns = set()
    for reg_col in reg_cols:
        df = pd.read_csv('data/reg/{}.csv'.format(reg_col))
        # NORMALIZING THE REGRESSION DATA
        reg_means.append(df[reg_col].mean())
        reg_stds.append(df[reg_col].std())
        df[reg_col] = (df[reg_col] - reg_means[-1]) / (reg_stds[-1])
        df = df.sample(frac=1, random_state=generator.bit_generator).reset_index(drop=True)
        dfs.append(df)
        columns.update(df.columns.to_list())
    for clf_col in clf_cols:
        df = pd.read_csv('data/clf/{}.csv'.format(clf_col))
        df = df.sample(frac=1, random_state=generator.bit_generator).reset_index(drop=True)
        dfs.append(df)
        columns.update(df.columns.to_list())

    if num_folds is not None:
        dfs = kfolds(dfs, num_folds)
        train_epoch_loops = num_folds
    else:
        dfs = train_test_splits(dfs, train_ratio)
        train_epoch_loops = 1
    
    model = PredictionModel(num_layers=num_layers, d_model=d_model, dff=dff, num_heads=num_heads, vocab_size=vocab_size,
                            dropout_rate=dropout_rate, reg_nums=len(reg_cols), clf_nums=len(clf_cols), maximum_positional_encoding=200)

    encoder_params = {}
    for name, param in pretrained_weights_file['model_state_dict'].items():
        if name.startswith('encoder.'):
            new_name = name[len('encoder.'):]  # need to remove the prefix "encoder." so we can load_state_dict with no issues
            encoder_params[new_name] = param
    model.encoder.load_state_dict(encoder_params)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.5e-4, betas=(0.9, 0.98))

    train_loss = AverageMeter()
    test_loss = AverageMeter()

    train_aucs = Records_AUC()
    train_accs = Records_Acc()

    test_aucs = Records_AUC()
    test_accs = Records_Acc()

    train_r2 = Records_R2()
    train_rmse = Records_RMSE()

    test_r2 = Records_R2()
    test_rmse = Records_RMSE()

    loss_func1 = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_func2 = torch.nn.MSELoss(reduction='none')

    def train_step(x, properties):
        model.train()
        clf_true = properties['clf']
        reg_true = properties['reg']
        properties_pred = model(x)

        clf_pred = properties_pred['clf']
        reg_pred = properties_pred['reg']

        loss = 0

        if len(clf_cols) > 0:
            loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                    clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum() + 1e-6)

        if len(reg_cols) > 0:
            loss += (loss_func2(reg_pred, reg_true) * (reg_true != -1000).float()).sum() / (
                    (reg_true != -1000).float().sum() + 1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if len(clf_cols) > 0:
            train_aucs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
            train_accs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
        if len(reg_cols) > 0:
            train_r2.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())
            train_rmse.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())
        train_loss.update(loss.detach().cpu().item(), x.shape[0])

    def test_step(x, properties):
        model.eval()
        with torch.no_grad():
            clf_true = properties['clf']
            reg_true = properties['reg']
            properties_pred = model(x)

            clf_pred = properties_pred['clf']
            reg_pred = properties_pred['reg']

            loss = 0

            if len(clf_cols) > 0:
                loss += (loss_func1(clf_pred, clf_true * (clf_true != -1000).float()) * (
                        clf_true != -1000).float()).sum() / ((clf_true != -1000).float().sum() + 1e-6)

            if len(reg_cols) > 0:
                loss += (loss_func2(reg_pred, reg_true) * (reg_true != -1000).float()).sum() / (
                        (reg_true != -1000).sum() + 1e-6)

            if len(clf_cols) > 0:
                test_aucs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
                test_accs.update(clf_pred.detach().cpu().numpy(), clf_true.detach().cpu().numpy())
            
            if len(reg_cols) > 0:
                test_r2.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())
                test_rmse.update(reg_pred.detach().cpu().numpy(), reg_true.detach().cpu().numpy())

            test_loss.update(loss.detach().cpu().item(), x.shape[0])

    log_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    with open(f'logs/finetune/{log_time}-{model_save_name}-finetune-log.txt', 'a') as f:
        f.write(f"fragmentation: {fragmentation}\n")
        f.write(f"max testing hasn't improved count: {max_testing_not_improved}\n")
        f.write(f"frag vocab path: {frag_vocab_path}\n")
        f.write(f"random state: {random_state}\n")
        f.write(f"number of folds: {num_folds}\n")
        f.write(f"training set proportion: {train_ratio}\n")

    for i in range(train_epoch_loops):  # once for train-test split, k times for k-fold cross validation

        # GET TRAIN AND TEST DATA READY
        train_data = None
        test_data = None
        train_dataset_boundaries = []

        for entry in dfs:
            if num_folds is not None:
                train_tmp = pd.concat(entry[:i] + entry[i+1:]).reset_index(drop=True)
                test_tmp = entry[i].reset_index(drop=True)
            else:
                train_tmp = entry[0].reset_index(drop=True)
                test_tmp = entry[1].reset_index(drop=True)

            if train_data is None:
                start_train_boundary = 0
                train_data = train_tmp
            else:
                start_train_boundary = len(train_data)
                train_data = pd.concat((train_data, train_tmp), ignore_index=True).reset_index(drop=True)
            train_dataset_boundaries.append((start_train_boundary, len(train_data) - 1))

            if test_data is None:
                test_data = test_tmp
            else:
                test_data = pd.concat((test_data, test_tmp), ignore_index=True).reset_index(drop=True)

        train_dataset = Prediction_Dataset(train_data, smiles_head=smiles_col,
                                           reg_heads=reg_cols, clf_heads=clf_cols, fragmentation=fragmentation,
                                           model_str2num=model_str2num, smiles_str2num=smiles_str2num, frag_str2num=frag_str2num)
        test_dataset = Prediction_Dataset(test_data, smiles_head=smiles_col,
                                          reg_heads=reg_cols, clf_heads=clf_cols, fragmentation=fragmentation,
                                          model_str2num=model_str2num, smiles_str2num=smiles_str2num, frag_str2num=frag_str2num)

        train_dataloader = DataLoader(train_dataset,
                                      batch_sampler=DisjointDatasetBatchSampler(train_dataset_boundaries, 
                                                                                batch_size=64, 
                                                                                shuffle=True),
                                      collate_fn=Finetune_Collater(clf_cols, reg_cols),
                                      num_workers=num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                                     collate_fn=Finetune_Collater(clf_cols, reg_cols),
                                     num_workers=num_workers)

        early_stop_val = float('inf')
        has_not_improved_count = 0
        testing_aucs = [[] for _ in range(len(clf_cols))]
        testing_accs = [[] for _ in range(len(clf_cols))]
        testing_r2s = [[] for _ in range(len(reg_cols))]
        testing_rmses = [[] for _ in range(len(reg_cols))]
        test_losses = []
        test_aucs.reset()
        test_accs.reset()
        test_rmse.reset()
        test_r2.reset()
        test_loss.reset()

        for epoch in range(9999999):  # TODO: change

            # TRAINING EPOCH
            for x, properties in tqdm(train_dataloader):
                x = x.to(device)
                if len(clf_cols) > 0:
                    properties['clf'] = properties['clf'].to(device)
                if len(reg_cols) > 0:
                    properties['reg'] = properties['reg'].to(device)
                train_step(x, properties)

            print('train epoch: ', epoch, 'train loss: {:.4f}'.format(train_loss.avg))
            if len(clf_cols) > 0:
                train_auc_results = train_aucs.results()
                train_acc_results = train_accs.results()
                for num, clf_head in enumerate(clf_cols):
                    print('train auc {}: {:.4f}'.format(clf_head, train_auc_results[num]))
                    print('train acc {}: {:.4f}'.format(clf_head, train_acc_results[num]))
            if len(reg_cols) > 0:
                train_r2_results = train_r2.results()
                train_rmse_results = train_rmse.results()
                for num, reg_head in enumerate(reg_cols):
                    print('train r2 {}: {:.4f}'.format(reg_head, train_r2_results[num]))
                    print('train rmse {}: {:.4f}'.format(reg_head, train_rmse_results[num]))
            train_aucs.reset()
            train_accs.reset()
            train_r2.reset()
            train_rmse.reset()
            train_loss.reset()

            # TESTING EPOCH
            for x, properties in tqdm(test_dataloader):
                x = x.to(device)
                if len(clf_cols) > 0:
                    properties['clf'] = properties['clf'].to(device)
                if len(reg_cols) > 0:
                    properties['reg'] = properties['reg'].to(device)
                test_step(x, properties)
            
            print('testing epoch: ', epoch, 'test loss: {:.4f}'.format(test_loss.avg))
            if len(clf_cols) > 0:
                test_auc_results = test_aucs.results()
                test_acc_results = test_accs.results()
                for num, clf_head in enumerate(clf_cols):
                    print('test auc {}: {:.4f}'.format(clf_head, test_auc_results[num]))
                    print('test accs {}: {:.4f}'.format(clf_head, test_acc_results[num]))
                    testing_aucs[num].append(test_auc_results[num])
                    testing_accs[num].append(test_acc_results[num])
            if len(reg_cols) > 0:
                test_r2_results = test_r2.results()
                test_rmse_results = test_rmse.results()
                for num, reg_head in enumerate(reg_cols):
                    print('test r2 {}: {:.4f}'.format(reg_head, test_r2_results[num]))
                    print('test rmse {}: {:.4f}'.format(reg_head, test_rmse_results[num]))
                    testing_r2s[num].append(test_r2_results[num])
                    testing_rmses[num].append(test_rmse_results[num])
            test_losses.append(test_loss.avg)

            if (test_loss.avg < early_stop_val + 0.001):
                early_stop_val = test_loss.avg
                has_not_improved_count = 0

                save_data = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "random_state": random_state,
                    "dropout_rate": dropout_rate,
                    "training_epoch": epoch,
                    "model_name": model_save_name,
                    "model_arch": arch,
                    "log_time": log_time,
                    "fragmentation": fragmentation,
                    "frag_vocab_path": frag_vocab_path,
                    "smiles_vocab_path": smiles_vocab_path,
                    "smiles_frag_vocab_path": smiles_frag_vocab_path,
                    "max_testing_not_improved": max_testing_not_improved,
                    "reg_cols": reg_cols,
                    "clf_cols": clf_cols,
                    "reg_means": reg_means,
                    "reg_stds": reg_stds,
                }

                if num_folds is not None:
                    save_name = f'weights/finetune/{log_time}_{model_save_name}_fold{i}_best.pt'
                    save_data['fold'] = i
                else:
                    save_name = f'weights/finetune/{log_time}_{model_save_name}_best.pt'
                    save_data['train_ratio'] = train_ratio

                torch.save(save_data, save_name)
            else:
                has_not_improved_count += 1

            print(f"Testing loss hasn't improved in f{has_not_improved_count} test epochs")

            test_aucs.reset()
            test_accs.reset()
            test_r2.reset()
            test_rmse.reset()
            test_loss.reset()

            if has_not_improved_count >= max_testing_not_improved:
                print("Threshold of testing non-improvement reached.")
                break
        
        with open(f'logs/finetune/{log_time}-{model_save_name}-finetune-log.txt', 'a') as f:
            if num_folds is not None:
                f.write(f'############## FOLD {i} ##############\n')
            for j in range(len(clf_cols)):
                f.write(clf_cols[j] + ' (AUC)\n')
                for val in testing_aucs[j]:
                    f.write(str(val) + '\n')
                f.write(clf_cols[j] + ' (Acc)\n')
                for val in testing_accs[j]:
                    f.write(str(val) + '\n')

            for j in range(len(reg_cols)):
                f.write(reg_cols[j] + ' (R^2)\n')
                for val in testing_r2s[j]:
                    f.write(str(val) + '\n')
                f.write(reg_cols[j] + ' (RMSE)\n')
                for val in testing_rmses[j]:
                    f.write(str(val) + '\n')

            f.write('test_epoch_losses\n')
            for val in test_losses:
                f.write(str(val) + '\n')


if __name__ == '__main__':
    # NOTE: device.py needs to be changed, depending on where finetuning is performed
    # NOTE: Read description of main() to understand why some arguments are set to None

    num_folds = 5  # k for k-fold cross-validation (set None for train-test split)
    train_ratio = None  # train ratio for train-test split (set to None for cross-validation)

    pretrain_model_weights_file_path = 'weights/pretrain/FILLTHIS!'
    model_save_name = None
    fragmentation = None

    max_testing_not_improved = 2
    smiles_vocab_path = None
    smiles_frag_vocab_path = None
    frag_vocab_path = None
    num_workers = 12
    dropout_rate = None
    random_state = None

    clf_cols = ['PAMPA_NCATS', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma', 'BBB_Martins',
                'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith',
                'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels',
                'CYP3A4_Substrate_CarbonMangels', 'AMES', 'DILI', 'skin_reaction', 'Carcinogens_Lagunin',
                'ClinTox', 'hERG']
    reg_cols = ['Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB',
                'HydrationFreeEnergy_FreeSolv', 'PPBR_AZ', 'VDss_Lombardo',
                'Half_Life_Obach', 'Clearance_Hepatocyte_AZ',
                'Clearance_Microsome_AZ', 'LD50_Zhu']
    smiles_col = "SMILES"

    main(num_folds, pretrain_model_weights_file_path, clf_cols, reg_cols,
         max_testing_not_improved, smiles_col, fragmentation, model_save_name, 
         smiles_vocab_path, smiles_frag_vocab_path, frag_vocab_path, num_workers, 
         dropout_rate, random_state, train_ratio)
