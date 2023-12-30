import pandas as pd
import numpy as np
import ast

def read_exp(path_to_csv):   
    if path_to_csv.__contains__('oracle_oracle'):
        annotation_time, metric = rank_policy(pd.read_csv(path_to_csv))
    elif path_to_csv.__contains__('eva_vos'):
        annotation_time, metric = rank_policy(pd.read_csv(path_to_csv), rl_agent=True)
    else :
        df = pd.read_csv(path_to_csv).groupby('round')
        metric = df['mu_metric'].mean().to_numpy()
        annotation_time = df['annotation_time'].sum().to_numpy()
        annotation_time = np.cumsum(annotation_time)/3600

    return annotation_time, metric

PLOT_DATA = {
    # mask only experiment
    'qnet_mask': ['black', '-'],
    'oracle_mask': [(1.0,0.0, 0.16, 1.0), '--'],
    'rand_mask': [(0.36036036036036034, 1.0, 0.0, 1.0), '-'],
    'l2_mask_dino_large': ['cyan', '-'],
    'l2_mask_resnet101': ['#B2BEB5', '-'],
    'l2_mask_vit_large': ['#FFEF00', '-'],

}

PLOT_DATA_ANNOTATIONS = {
    # multiple annotations experiment
    'eva_vos': ['black', '-', 'EVA-VOS'],
    'rand_rand_3clicks_mask': [(0.36036036036036034, 1.0, 0.0, 1.0), '-', 'Random'],
    'oracle_oracle_3clicks_mask': [(1.0,0.0, 0.16, 1.0), '--', 'Oracle'],
    'rand_mask': ['magenta', '-', 'Mask-only'],
    'rand_type_3clicks': ['cyan', '-', 'Clicks-only'],

}


def rank_policy(df, gamma=0.6, rl_agent=False):
    """Video ranking Eq. 3 in the paper"""
    policy_data = {}
    videos = set()
   
    """Read daas"""
    for _, row in df.iterrows():
        video_name = row['video']
        videos.add(video_name)
        if video_name not in policy_data.keys():
            policy_data[video_name] = {}
            
        curr_round = row['round']
        round_metrics = ast.literal_eval(row['round_metrics'])
        round_mu = row['mu_metric']
        next_round_df = df[(df['video'] == video_name) & (df['round'] == curr_round + 1)].reset_index()
        
        if len(next_round_df) == 0:
            continue

        next_frame = next_round_df['annotated_frames'][0]

        policy_data[video_name][curr_round] = {
            'metric' : round_metrics,
            'mu_metric' : round_mu,
            'next_frame': next_frame,
            'next_metric':  ast.literal_eval(next_round_df['round_metrics'][0]),
            'annotatation_time': row['annotation_time'],
            'next_annotation_time': next_round_df['annotation_time'][0]
        }

        if rl_agent:
            policy_data[video_name][curr_round]['rl_value'] = next_round_df['rl_values'][0]

    videos_max_round = df.groupby('video')['round'].max().to_dict()
  
    """Rank videos"""
    #Initialize with mask at all videos
    initial_mus = []
    initial_times = []
    round_pointers = {}
    for vid in videos:
        mu = np.mean(policy_data[vid][0]['mu_metric'])
        t = policy_data[vid][0]['annotatation_time']

        initial_mus.append(mu)
        initial_times.append(t)
        round_pointers[vid] = 0 # first round all videos
    
    points = [np.mean(initial_mus)]
    times = [np.sum(initial_times)]
    videos = list(videos)

    while True:
        r = {}

        for vid in videos:
            try:
                pointer = round_pointers[vid]
                curr_round_metric = policy_data[vid][pointer]['metric']
                next_round_metric = policy_data[vid][pointer+1]['metric']

                frame_num = policy_data[vid][pointer]['next_frame']
                cost = policy_data[vid][pointer]['next_annotation_time']

                init_iou = curr_round_metric[frame_num]
                iou = next_round_metric[frame_num]
                    
               
                if rl_agent:
                    value = policy_data[vid][pointer]['rl_value'] + 0.04
                    if value == -2:
                        value = 0
                    
                    f_v = value* (gamma**pointer)/cost                  
                    r_ii =f_v
                else:
                   r_ii = (iou - init_iou)/cost
                        
                
            except KeyError: # max rounds
               r_ii = -1e10
            
            r[vid] = r_ii

        assert len(r) == len(videos)

        sorted_r = sorted(r.items(), key=lambda x: x[1], reverse=True) # sort in descending order
        selected_video = -1
        for vid, r_ii in sorted_r:
            if round_pointers[vid] != videos_max_round[vid] -1 :
                selected_video = vid
                break            

        if selected_video == -1:
            break
        
        pointer = round_pointers[selected_video]
        cost = policy_data[selected_video][pointer]['next_annotation_time']
        round_pointers[selected_video] += 1

        new_ious = []
        for vid in videos:
            new_ious.append(policy_data[vid][round_pointers[vid]]['mu_metric'])
        
        points.append(np.mean(new_ious))
        times.append(times[-1] + cost)

    times = np.array(times)/3600

    return times, points 
