a EEG feature extraction tool based on PYTHON-MNE,numpy,pandas,scipy
5 function:
1. get_amp_channel(sub,ROIS, conditions,t_min=0.35, t_max=0.45, name='N400', csv_file_path='get_amp_channel.csv')
2. get_amp_epoch(sub,ROIS, conditions,t_min=0.35,t_max=0.5,name='P200',csv_file_path='get_amp_epoch.csv')
3. get_amp_ROIS(sub,ROIS, conditions,t_min=0.35, t_max=0.45, name='N400', csv_file_path='get_amp_ROI.csv')
4. get_psd_epoch(sub, ROIS, conditions, x1, x2, csv_file_path='get_psd_epoch.csv', sfreq=512, n_per_seg=256, n_overlap=192)
5. get_psd__window(sub, ROIS, conditions, x1, x2, csv_file_path='get_psd_window.csv', sfreq=512, n_per_seg=256, n_overlap=192)

for deepleaning(suchas1D-CNN) or dataanalyse(suchas comparation of N400 in different condition in a typical ERPstudy)
