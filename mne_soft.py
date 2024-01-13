import mne
import pandas as pd
import scipy.signal
import numpy as np



# epoch文件名需要设置为subN-epo.fif

def get_amp_ROIS(sub,ROIS, conditions,t_min=0.35, t_max=0.45, name='N400', csv_file_path='get_amp_ROI.csv'):
    try:    # 首先尝试读取现有的 CSV 文件到 DataFrame，如果文件不存在，则初始化一个空的 DataFrame
        amp_df = pd.read_csv(csv_file_path)
        expected_columns = {'subject', 'condition'}  # 预期的基本列集合
        if not expected_columns.issubset(amp_df.columns):
            raise ValueError(f"CSV 文件缺少必需的列: {expected_columns - set(amp_df.columns)}")
    except (FileNotFoundError, ValueError):
        amp_df = pd.DataFrame(columns=['subject', 'condition'])  # 添加基本列        
    epo = mne.read_epochs(f'sub{sub}-epo.fif')
    epo_crop= epo.copy().crop(t_min, t_max)  # 裁剪到 N400 或 P600 等时间段
    for condition in conditions:
        for ROI_key, ROI_value in ROIS.items():        
            existing_records = amp_df[(amp_df['subject'] == f'sub{sub}') & (amp_df['condition'] == condition)]# 检查是否存在当前主题和条件的记录  
            epo_roi = epo_crop.copy().pick_channels(ROI_value)  # 选定感兴趣区
            evoked = epo_roi[condition].average()
            mean_amp = evoked.data.mean(axis=0).mean()*1000000 #换算成uV
            column_name = f"{ROI_key}_{name}"
            if not existing_records.empty:# 如果存在，更新记录              
                amp_df.loc[(amp_df['subject'] == f'sub{sub}') & (amp_df['condition'] == condition), f"{ROI_key}_{name}"] = mean_amp
            else:# 如果不存在，创建新记录
                amp_df = amp_df._append({'subject': f'sub{sub}', 'condition': condition, column_name: mean_amp}, ignore_index=True)
    amp_df.to_csv(csv_file_path, header=True, index=False)# 写入 CSV 文件
    return amp_df

def get_amp_channel(sub,ROIS, conditions,t_min=0.35, t_max=0.45, name='N400', csv_file_path='get_amp_channel.csv'):
    try:
        amp_df = pd.read_csv(csv_file_path)
        expected_columns = {'subject', 'condition'}  # 预期的基本列集合
        if not expected_columns.issubset(amp_df.columns):
            raise ValueError(f"CSV 文件缺少必需的列: {expected_columns - set(amp_df.columns)}")
    except (FileNotFoundError, ValueError):
        amp_df = pd.DataFrame(columns=['subject', 'condition'])  # 添加基本列        
    epo = mne.read_epochs(f'sub{sub}-epo.fif')
    epo_crop = epo.copy().crop(t_min, t_max)
    for condition in conditions:
        for ROI_key, ROI_value in ROIS.items():
            epo_roi = epo_crop.copy().pick_channels(ROI_value)
            evoked = epo_roi[condition].average()
            for ch in ROI_value:# 对每个电极计算平均幅度
                ch_index = epo_roi.ch_names.index(ch)
                mean_amp = evoked.data[ch_index].mean() * 1000000  # V转换为uV
                column_name = f"{ROI_key}_{ch}_{name}"
                if amp_df[(amp_df['subject'] == f'sub{sub}') & (amp_df['condition'] == condition)].empty:#创建/更新
                    amp_df = amp_df._append({'subject': f'sub{sub}', 'condition': condition, column_name: mean_amp}, ignore_index=True)
                else:
                    amp_df.loc[(amp_df['subject'] == f'sub{sub}') & (amp_df['condition'] == condition), column_name] = mean_amp
    amp_df.to_csv(csv_file_path, header=True, index=False)# 写入 CSV 文件
    return amp_df

def get_amp_epoch(sub, ROIS, conditions, t_min=0.35, t_max=0.45, name='N400', csv_file_path='get_amp_epoch.csv'):
    try:
        amp_df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, ValueError):
        column_names = ['subject', 'condition', 'epoch'] + [f"{ROI}_{ch}_{name}" for ROI in ROIS for ch in ROIS[ROI]]
        amp_df = pd.DataFrame(columns=column_names)

    epo = mne.read_epochs(f'sub{sub}-epo.fif')
    epo_crop = epo.copy().crop(t_min, t_max)

    for condition in conditions:
        for epoch_idx, epoch_data in enumerate(epo_crop[condition].get_data()):
            row = {'subject': f'sub{sub}', 'condition': condition, 'epoch': epoch_idx}
            for ROI_key, ROI_value in ROIS.items():
                for ch in ROI_value:
                    ch_index = epo_crop.ch_names.index(ch)
                    mean_amp = np.mean(epoch_data[ch_index]) * 1e6  # V转换为uV
                    column_name = f"{ROI_key}_{ch}_{name}"
                    row[column_name] = mean_amp
            existing_row = amp_df[(amp_df['subject'] == row['subject']) & (amp_df['condition'] == row['condition']) & (amp_df['epoch'] == row['epoch'])]
            if existing_row.empty:
                amp_df = amp_df._append(row, ignore_index=True)
            else:
                idx = existing_row.index[0]
                for col, val in row.items():
                    amp_df.at[idx, col] = val

    amp_df.to_csv(csv_file_path, header=True, index=False)
    return amp_df

def find_close(arr, e):
    idx = (np.abs(arr - e)).argmin()
    return idx

def auc(freqs, psd, x1, x2):
    idx1 = find_close(freqs, x1)
    idx2 = find_close(freqs, x2)
    return np.trapz(psd[idx1:idx2 + 1], freqs[idx1:idx2 + 1])

def get_psd_epoch(sub, ROIS, conditions, x1, x2, csv_file_path='get_psd_epoch.csv', sfreq=512, n_per_seg=256, n_overlap=192):
    try:
        psd_df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, ValueError):
        column_names = ['subject', 'condition', 'epoch'] + [f"{ROI}_{ch}_{x1}-{x2}Hz" for ROI in ROIS for ch in ROIS[ROI]]
        psd_df = pd.DataFrame(columns=column_names)
    epo = mne.read_epochs(f'sub{sub}-epo.fif')
    for condition in conditions:
        for epoch_idx, epoch_data in enumerate(epo[condition].get_data()):
            row = {'subject': f'sub{sub}', 'condition': condition, 'epoch': epoch_idx}
            for ROI_key, ROI_value in ROIS.items():
                for ch in ROI_value:
                    ch_index = epo.ch_names.index(ch)
                    data_uV = epoch_data[ch_index] * 1e6 # 将数据从V转换为uV
                    f, Pxx = scipy.signal.welch(data_uV, fs=sfreq, nperseg=n_per_seg, noverlap=n_overlap)
                    area = auc(f, Pxx, x1, x2)
                    column_name = f"{ROI_key}_{ch}_{x1}-{x2}Hz"
                    row[column_name] = area
            if not ((psd_df['subject'] == row['subject']) & (psd_df['condition'] == row['condition']) & (psd_df['epoch'] == row['epoch'])).any():
                psd_df = psd_df._append(row, ignore_index=True)
            else:
                idx = psd_df[(psd_df['subject'] == row['subject']) & (psd_df['condition'] == row['condition']) & (psd_df['epoch'] == row['epoch'])].index[0]
                for col, val in row.items():
                    psd_df.at[idx, col] = val
    psd_df.to_csv(csv_file_path, header=True, index=False)
    return psd_df

def get_psd__window(sub, ROIS, conditions, x1, x2, csv_file_path='get_psd_window.csv', sfreq=512, n_per_seg=256, n_overlap=192):
    try:
        psd_df = pd.read_csv(csv_file_path)
    except (FileNotFoundError, ValueError):
        psd_df = pd.DataFrame()
    epo = mne.read_epochs(f'sub{sub}-epo.fif')
    epoch_length = len(epo.times) / sfreq  # 计算每个 epoch 的时长（秒）
    num_windows = int(np.ceil((epoch_length * sfreq - n_overlap) / (n_per_seg - n_overlap)))  # 计算窗的数量
    # 如果是新文件就初始化 DataFrame 的列名称
    if psd_df.empty:
        column_names = ['subject', 'condition', 'epoch'] + [f"{ROI}_{ch}_window{w}_{x1}-{x2}Hz" for w in range(num_windows) for ROI in ROIS for ch in ROIS[ROI]]
        psd_df = pd.DataFrame(columns=column_names)
    for condition in conditions:
        for epoch_idx, epoch_data in enumerate(epo[condition].get_data()):
            row = {'subject': f'sub{sub}', 'condition': condition, 'epoch': epoch_idx}
            for window_idx in range(num_windows):
                start_idx = window_idx * (n_per_seg - n_overlap)
                end_idx = start_idx + n_per_seg
                window_data = epoch_data[:, start_idx:end_idx] * 1e6  # 将数据单位从V转换为uV
                for ROI_key, ROI_value in ROIS.items():
                    for ch in ROI_value:
                        ch_index = epo.ch_names.index(ch)
                        f, Pxx = scipy.signal.welch(window_data[ch_index], fs=sfreq, nperseg=n_per_seg, noverlap=0)
                        area = auc(f, Pxx, x1, x2)
                        column_name = f"{ROI_key}_{ch}_window{window_idx}_{x1}-{x2}Hz"
                        row[column_name] = area
            existing_row = psd_df[(psd_df['subject'] == row['subject']) & 
                                  (psd_df['condition'] == row['condition']) &
                                  (psd_df['epoch'] == row['epoch'])]
            if existing_row.empty:
                psd_df = psd_df._append(row, ignore_index=True)
            else:
                idx = existing_row.index[0]
                for col, val in row.items():
                    psd_df.at[idx, col] = val
    psd_df.to_csv(csv_file_path, header=True, index=False)
    return psd_df
