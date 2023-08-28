############# Converts the raw brekel data from the .txt files into .csv ###############
# conda install fastparquet

import math
import os
import logging
import re
import shutil

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R



def save_csv( df, filepath, filename ):
    """Save dataframe (table) as CSV.

    Parameters
    ----------
    df : DataFrame
        The data to save.
    filepath : str
        Path to where the data should be saved.
    filename : str
        Name of the output file (without extension).

    """
    df.to_csv(Path(filepath, f'{filename}.csv'), float_format='%.7f')
    # df.to_csv(Path(filepath, f'{filename}.raw.csv.gz'), float_format='%.7f', compression="gzip")
    df.to_parquet(Path(filepath, f'{filename}.parquet'), engine='fastparquet')


def main():    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # pd.set_option('display.float_format', lambda x: '%.8f' % x)

    logging.info('**************** Brekel raw data import to csv ************************')

    # Prepare parsing state tracking
    device_types = [
        'HMD',                  # devicetype = 0
        'Controllers',          # devicetype = 1
        'Generic Trackers',     # devicetype = 2  Tundra
        'Tracking references',  # devicetype = 3  Base station
    ]
    current_device_type = -1
    
    parsing_state = 0
    # 0 = beginning of file, ignore everything until (first) device type is found
    # 1 = device type found (like HMD, controller, ...) -> do nothing until "deviceid timestamp ..." is found
    # 2 = parse deviced,timestamp,location,rotation
    #       until an empty line is found, back to 0
    # 3 =  post-process and save parsed data

    input = Path('.') / 'analysis' / 'Data'
    output = Path('.') / 'analysis' / 'Data-output'

    # File parameters:
    subdirs = ['2023-05-15','2023-05-16','2023-05-19','2023-05-20','2023-05-22','2023-05-23','2023-05-24','2023-05-26','2023-06-05','2023-06-06','2023-06-09','2023-06-10','2023-06-21',]
    savesubdirs = [
        # '2023-03-31 Test run Alex(HL+Wrench) and Sam (PW)',
        # '2023-05-12 (5 Calibration with outages recordings)',
        '2023-05-15',
        '2023-05-16',
        '2023-05-19',
        '2023-05-20',
        '2023-05-22',
        '2023-05-23',
        '2023-05-24',
        '2023-05-26',
        '2023-06-05',
        '2023-06-06',
        '2023-06-09',
        '2023-06-10',
        '2023-06-21',
    ]

    filesummary = output / 'recordings_debug.csv'
    csvrecordings = output / 'recordings.csv'
    if (not filesummary.exists()):
        sum_df = pd.DataFrame(columns=['filein','file','start','time','stop',
                               'HMD_quality', 'Contr_quality', 'Tundra_quality',
                               'HMD_samples', 'HMD_duplicates', 'HMD_maxstep', 'HMD_maxv',
                               'Contr1_samples','Contr1_duplicates', 'Contr1_maxstep', 'Contr1_maxv',
                               'Contr2_samples','Contr2_duplicates', 'Contr2_maxstep', 'Contr2_maxv',
                               'Tundra_samples','Tundra_duplicates', 'Tundra_maxstep', 'Tundra_maxv',
                               'Bases','Base_samples','Base_duplicates','timeanalysed',
                               ])
        sum_df.set_index(['filein'])
    else:
        sum_df = pd.read_csv( filesummary, index_col=[0], compression='infer')
    df_label = pd.read_excel(input / 'Data label.xlsx')

    filemeta = 'first'
    for subdir in subdirs:
        inputfolder = input / subdir
        file_filter = f'*Brekel*_*.txt'
        logging.info(inputfolder)

        # Iterate over all Brekel files in datafolter
        files = Path(inputfolder).rglob(file_filter)
        for f in files:
            filemeta = getfilemeta(f, filemeta)
            logging.info('%s < %s %s', filemeta['filename_out'], f.parent.relative_to(inputfolder), f.name)
            if not subdir in savesubdirs:
                continue
            outputfolder = output / 'recordings' # / filemeta['userid']
            Path(outputfolder).mkdir(parents=True, exist_ok=True)
            shutil.copy(f.with_suffix('.fbx'), outputfolder / (filemeta['filename_out'] + '.fbx'))

            txt_file = open(f)

            date = {}
            sumrow = { 'file': filemeta['filename_out'], 'filein': f.name }

            # Go over file line by line
            data = []
            for line in txt_file:
                # Check parsing_state for which file section we're parsing atm

                if parsing_state == 0:              # Look for start of next data section
                    if line.startswith('Recording start'):
                        l = line.replace('\n', '').split('\t')
                        date[l[1].replace(':','')] = int(l[-1])
                        if 'millisecond' in line:
                            logging.info("Recording start: %s", date)

                    if line == ' Head Mounted Display(s)\n':
                        current_device_type = device_types.index('HMD')
                    elif line == ' Controllers\n' :
                        current_device_type = device_types.index('Controllers')
                    elif line == ' Generic Trackers\n' :
                        current_device_type = device_types.index('Generic Trackers')
                    elif line == ' Tracking Reference(s)\n' :
                        current_device_type = device_types.index('Tracking references')

                    if current_device_type >= 0:
                        logging.info("Parsing %s %s", current_device_type, device_types[current_device_type])
                        parsing_state = 1

                elif parsing_state == 1:            # column headers found, data section is starting
                    if 'deviceID	timestamp		posX' in line:
                        parsing_state = 2
                        # logging.info("Parsing")

                elif parsing_state == 2:            # parsing data by
                    # logging.info(line)

                    if line != '\n':                # split by tab character, then typecasting it to dataline array
                        l = list(filter(len, line.replace('\n', '').split('\t')))
                        l = np.array(l, dtype = 'float')
                        dataline = [
                            float(l[1]),       # timestamp
                            int(l[0]),  # deviceid
                            int(current_device_type),        # device_type
                            l[2],       # posX
                            l[3],       # posY
                            l[4],       # posZ
                            l[5],       # rotX
                            l[6],       # rotY
                            l[7],       # rotZ
                        ]
                        if current_device_type == device_types.index('Controllers'):
                            dataline.extend(l[8:])
                        data.append(dataline)
                        # logging.info(dataline)
                    else:                           # if line is empty, stop parsing
                        # parsing_state = 0
                        parsing_state = 3
                        # current_device_type = -1
                        logging.info("Stopped parsing, rows: %s", len(data))

                elif parsing_state == 3:

                    # Create dataframe from data array
                    columns = ['timestamp_s', 'deviceid', 'devicetype', 'tx_m', 'ty_m', 'tz_m', 'rx_deg', 'ry_deg', 'rz_deg']
                    if current_device_type == device_types.index('Controllers'):
                        columns.extend(['System', 'ApplicationMenu', 'Grip', 'DPad_Left', 'DPad_Up', 'DPad_Right', 'DPad_Down', 'A', 'ProximitySensor', 'Axis0', 'Axis1', 'Axis2', 'Axis3', 'Axis4', 'axis0_X', 'axis0_Y', 'axis1_X', 'axis1_Y', 'axis2_X', 'axis2_Y', 'axis3_X', 'axis3_Y', 'axis4_X', 'axis4_Y'])
                    df = pd.DataFrame(
                        columns=columns,
                        data=data
                    )#.set_index('timestamp_s')
                    # logging.info('Parsed')

                    if current_device_type != device_types.index('Tracking references'): 
                        df.insert(1,'time_step',np.nan)
                        cc = ['v_ms','vx_ms','vy_ms','vz_ms','wx_degs','wy_degs','wz_degs','q2x','q2y','q2z','q2w','q1x','q1y','q1z','q1w','wq_degs','wqx_degs','wqy_degs','wqz_degs']
                        cc.reverse()
                        for c in cc:
                            df.insert(10,c,np.nan)
                    
                    dflen1 = len (df.index)
                    df.drop_duplicates(['timestamp_s','deviceid'],keep='first',inplace=True)
                    dflen2 = len (df.index)
                    # logging.info('\n%s\n%s\n', df, df.dtypes)
                    if dflen1 != dflen2:
                        logging.info('Dropped %s - %s = %s duplicates', dflen1, dflen2, dflen1 - dflen2)

                    df.insert(0,'timestamp',np.nan)
                    df['timestamp'] = df['timestamp_s']*(10**9)
                    # logging.info('\n%s\n\n', df[['timestamp_s','timestamp']])
                    df['timestamp'] = df['timestamp'].astype('int64')
                    # logging.info('\n%s\n\n', df[['timestamp_s','timestamp']])
                    df['timestamp'] = pd.to_timedelta(df['timestamp'], 'ns')
                    # logging.info('\n%s\n\n', df[['timestamp_s','timestamp']])
                    df['timestamp'] += pd.Timestamp(year=date['year'], month=date['month'], day=date['day'], hour=date['hour'], minute=date['minute'], second=date['second'], microsecond=1000*date['millisecond'])
                    # logging.info('\n%s\n\n', df[['timestamp_s','timestamp']])

                    id_list = np.unique(df['deviceid'].values)
                    id_count = 1
                    for id in id_list:

                        b=df['deviceid'] == id
                        # logging.info('\n%s', df)
                        # logging.info('\n%s', df.loc[b])

                        #####################################
                        # Converting XZY Euler to ZYX Euler #
                        #####################################
                        # x = pitch
                        # y = yaw
                        # z = roll

                        # Extract N Euler rotation vectors - size is (N,3)
                        rot_vec = df.loc[b,['rx_deg', 'ry_deg', 'rz_deg']].to_numpy()
                        # Apply correct sequence (w.r.t. SteamVR, ZYX - roll, yaw, pitch)
                        rot_vec_tran = np.c_[rot_vec[:, 2], rot_vec[:, 1], rot_vec[:, 0]]

                        r = R.from_euler(
                            'XZY',  # Roll, yaw, pitch in Scipy notation
                            rot_vec_tran,  # Sorted Euler rotation vectors
                            degrees=True
                        )

                        # rot_vec_new = r.as_euler('ZYX', degrees=True)  # Same rotations represented as yaw, pitch, roll Scipy Euler sequences
                        # # new ZYX (yaw 0, pitch 1, roll 2) euler data to SteamVR axes (rx=pitch, ry=yaw, rz=roll)
                        # df.loc[b,'rx_deg'] = rot_vec_new[:, 1].tolist()
                        # df.loc[b,'ry_deg'] = rot_vec_new[:, 0].tolist()
                        # df.loc[b,'rz_deg'] = rot_vec_new[:, 2].tolist()

                        # convert to quaternions
                        rot_vecQ_new = r.as_quat()  # The returned value is in scalar-last (x, y, z, w) format.

                        df.loc[b,'q2x'] = rot_vecQ_new[:, 0].tolist()
                        df.loc[b,'q2y'] = rot_vecQ_new[:, 1].tolist()
                        df.loc[b,'q2z'] = rot_vecQ_new[:, 2].tolist()
                        df.loc[b,'q2w'] = rot_vecQ_new[:, 3].tolist()

                        # Calculate velocities 'v_ms','vx_ms','vy_ms','vz_ms','wx_degs','wy_degs','wz_degs'
                        if current_device_type != device_types.index('Tracking references'): # Not a base station
                            # Calculate time_step, needed for velocity calculations
                            df.loc[b,'time_step'] = df.loc[b,'timestamp_s'].diff()

                            # Calculate total and ax translation velocity
                            total_v_col = 'v_ms'
                            df.loc[b,total_v_col] = 0
                            for ax in ['x', 'y', 'z']:
                                v_ax_col = f'v{ax}_ms'
                                # Velocity(ax) = diff t(ax) / time_step
                                df.loc[b,v_ax_col] = df.loc[b,f't{ax}_m'].diff() / df.loc[b,'time_step']
                                # Total velocity^2 += Ax velocity^2
                                df.loc[b,total_v_col] += df.loc[b,v_ax_col].pow(2)
                            # Total velocity = sqrt(sum axes^2)
                            df.loc[b,total_v_col] = df.loc[b,total_v_col].pow(1. / 2)

                            # https://mariogc.com/post/angular-velocity-quaternions/
                            # ωx = ​Δt2​(qw​(t)qx​(t+Δt)−qx​(t)qw​(t+Δt)−qy​(t)qz​(t+Δt)+qz​(t)qy​(t+Δt))
                            # ωy = Δt2​(qw​(t)qy​(t+Δt)+qx​(t)qz​(t+Δt)−qy​(t)qw​(t+Δt)−qz​(t)qx​(t+Δt))
                            # ​ωz​​ = Δt2​(qw​(t)qz​(t+Δt)−qx​(t)qy​(t+Δt)+qy​(t)qx​(t+Δt)−qz​(t)qw​(t+Δt))​
                            # ωx = ​Δt2​(q1w*q2x−q1x*q2w−q1y*q2z+q1z*q2y)
                            # ωy = Δt2​(q1w*q2y+q1x*q2z−q1y*q2w−q1z*q2x)
                            # ​ωz​​ = Δt2​(q1w*q2z−q1x*q2y+q1y*q2x−q1z*q2w)​
                            # def angular_velocities(q1, q2, dt):
                            #     return (2 / dt) * np.array([
                            #         q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
                            #         q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
                            #         q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

                            df.loc[b,'q1x'] = df.loc[b,'q2x'].shift(1)
                            df.loc[b,'q1y'] = df.loc[b,'q2y'].shift(1)
                            df.loc[b,'q1z'] = df.loc[b,'q2z'].shift(1)
                            df.loc[b,'q1w'] = df.loc[b,'q2w'].shift(1)

                            df.loc[b,'wqx_degs'] = np.rad2deg(
                                2 / df.time_step * ((df.q1w * df.q2x) - (df.q1x * df.q2w) - (df.q1y * df.q2z) + (df.q1z * df.q2y)))
                            df.loc[b,'wqy_degs'] = np.rad2deg(
                                2 / df.time_step * ((df.q1w * df.q2y) + (df.q1x * df.q2z) - (df.q1y * df.q2w) - (df.q1z * df.q2x)))
                            df.loc[b,'wqz_degs'] = np.rad2deg(
                                2 / df.time_step * ((df.q1w * df.q2z) - (df.q1x * df.q2y) + (df.q1y * df.q2x) - (df.q1z * df.q2w)))

                            # https://math.stackexchange.com/questions/90081/quaternion-distance
                            # \theta \;=\; \cos^{-1}\bigl(2\langle q_1,q_2\rangle^2 -1\bigr)
                            df.loc[b,'wq_degs'] = np.rad2deg( np.arccos(2*(df.q1w*df.q2w + df.q1x*df.q2x + df.q1y*df.q2y + df.q1z*df.q2z)**2 - 1) ) / df.time_step

                        # Summary stats
                        iddf = df[df['deviceid'] == id]
                        # logging.info(iddf)
                        samples = len(iddf.index)
                        pre = ['HMD','Contr','Tundra', 'Base'][current_device_type]
                        match current_device_type:
                            case 3:
                                sumrow['Bases'] = id_count
                            case 1:
                                pre = f'{pre}{id_count}'
                        if id_count==1 or current_device_type==1:
                            sumrow[pre+'_samples'] = samples
                        else:
                            sumrow[pre+'_samples'] += samples
                        sumrow[pre+'_duplicates'] = dflen1 - dflen2
                        if current_device_type != 3:
                            sumrow[pre+'_maxstep'] = round(iddf['time_step'].max(),2)
                            sumrow[pre+'_maxv'] = round(iddf['v_ms'].abs().max(),2)

                        id_count += 1
                    
                    start = df['timestamp'].min()
                    time = round(df['timestamp_s'].max(),1)
                    stop = df['timestamp'].max()
                    sumrow['start'] = min(start, sumrow['start']) if 'start' in sumrow else start
                    sumrow['time']  = max(time, sumrow['time'])   if 'time' in sumrow  else time
                    sumrow['stop']  = max(stop, sumrow['stop'])   if 'stop' in sumrow  else stop
                    sumrow['timeanalysed'] = pd.Timestamp.now()

                    labels = df_label[df_label['file'] == f.name].iloc[0]
                    sumrow['HMD_quality'] = labels['HMD_rating']
                    sumrow['Contr_quality'] = labels['Contr_rating']
                    sumrow['Tundra_quality'] = labels['Tundra_rating']

                    df.rename(columns={"q2x": "qx","q2y": "qy","q2z": "qz","q2w": "qw"},inplace=True)
                    df.drop(columns=['rx_deg','ry_deg','rz_deg','wx_degs','wy_degs','wz_degs','q1x','q1y','q1z','q1w'],inplace=True, errors='ignore')
                    df.drop(columns=['timestamp', 'time_step'],inplace=True, errors='ignore')   # cleanup useless columns
                    df.dropna(inplace=True)
                    df.set_index(['timestamp_s','deviceid'],inplace=True)

                    save_csv( df, outputfolder, f"{filemeta['filename_out']}_[{current_device_type}]" )
                    logging.info("Saved parsing for %s", current_device_type )

                    data = []
                    parsing_state = 0
                    current_device_type = -1

            txt_file.close()
            
            sum_df.loc[sumrow['filein']] = sumrow
            sum_df.to_csv(filesummary)

            rsum_df_copy = sum_df.copy()
            rsum_df_copy.drop(columns=[
                'start',
                'filein',
                'stop',
                'HMD_duplicates',
                'Contr1_duplicates',
                'Contr2_duplicates',
                'Tundra_duplicates',
                'Base_duplicates',
                'timeanalysed',
            ], inplace=True)
            rsum_df_copy.to_csv(csvrecordings, index=False)
    
    generateVolunteerData(input, output)

userids = []
recordings = {}
pattern = re.compile(   r"""Brekel_
                        (?P<date>[0-9\-]{10})_                      # date
                        S(?P<slot>\d\d)_                            # slot
                        (?P<volunteer>[^_]*)_                       # volunteer
                        (?P<chapter>\d)(\.(?P<counter>\d))?\.       # chapter (+ counter)
                        (?P<game>[^#_]*)(_(?P<scenario>.*).*)?      # gamename (+ scenario name)
                        \#(?P<datetime>.*)                          # recording datetime 
                        """, re.VERBOSE)
def getfilemeta(f, volunteeriter):
    # Parse filenames for section
    # BrekelRaw_2023-05-26_S15_MariaSO_1.Calibration1#26_05_2023_14_47_55_[0]
    # BrekelRaw_2023-05-26_S15_MariaSO_2.Tpose#26_05_2023_15_08_55_[0]
    # BrekelRaw_2023-05-26_S15_MariaSO_3.1.Alyx_intro#26_05_2023_15_14_06_[0]
    # BrekelRaw_2023-05-26_S15_MariaSO_5.PW#26_05_2023_16_35_43_[0]
    match = pattern.match(f.name)
    filemeta = {'file': f.name}
    for i in ['date','slot','volunteer','chapter','counter','game','scenario','datetime']:
        filemeta[i] = match.group(i)
    if not filemeta['volunteer'] in userids:
        userids.append(filemeta['volunteer'])
        recordings[filemeta['volunteer']] = 0
    filemeta['recording nr'] = recordings[filemeta['volunteer']] + 1
    recordings[filemeta['volunteer']] = filemeta['recording nr']
    filemeta['userid'] = userids.index(filemeta['volunteer']) + 1
    if filemeta['chapter'] == '1':  # calibrationX to counter
        filemeta['counter'] = filemeta['game'][-1:]
        filemeta['scenario'] = filemeta['game'][-1:]
        filemeta['game'] = 'Calibration'
    if filemeta['counter'] is None:
        filemeta['counter'] = '1'
        filemeta['scenario'] = '1'
    # logging.info('%s', filemeta)
    filemeta['title'] = f"{filemeta['volunteer']} {filemeta['game']}{(' ' + filemeta['scenario'])}"
    s=f.name.split('#')
    filemeta['filename'] = '_'.join(np.array(s[1].split('_'))[[2,1,0,3,4,5]]) + ' ' + '_'.join(s[0].split('_')[2:])
    filemeta['userid'] = f"user{'{:02d}'.format(filemeta['userid'])}"
    filemeta['filename_out'] = f"cacerumd.{filemeta['userid']}.{'record{:02d}'.format(filemeta['recording nr'])}_{filemeta['chapter']}.{filemeta['counter']}"

    return filemeta

def generateVolunteerData(input, output):
    df_form = pd.read_excel(input / 'Questionaire.xlsx')
    df_notes = pd.read_excel(input / 'Notes.xlsx')

    df = pd.merge( left_on="Volunteer", right_on="UserID",
        left = df_notes[['Volunteer','Nr','Medium']],
        right = df_form[['UserID', 'Your age\n', 'Your gender\n','Your body height\n','Experience','Sportive']]
    )
    df.drop(columns=[
        'Volunteer',
        'UserID',
    ], inplace=True)
    df.rename(inplace=True, columns={
        'Nr': 'userid',
        'Medium': 'medium',
        'Your age\n': 'age',
        'Your gender\n': 'gender',
        'Your body height\n': 'height',
        'Experience': 'xr_experience',
        'Sportive': 'sportive',
    })
    df['userid'] = df['userid'].map("user{:02d}".format)
    df.to_csv(output / 'volunteers.csv', index=False)

    generateSSQScore(df_form, output)

questions = ['1. General discomfort', '2. Fatigue',
       '3. Headache', '4. Eyestrain', '5. Difficulty focusing',
       '6. Increased salivation', '7. Sweating', '8. Nausea',
       '9. Difficulty concentrating', '10. Fullness of head',
       '11. Blurred vision', '12. Dizziness (eyes open)',
       '13. Dizziness (eyes closed)', '14. Vertigo*',
       '15. Stomach awareness**', '16. Burping']
weights_SSQ = {
    'N': [1,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1],
    'O': [1,1,1,1,1,0,0,0,1,0,1,0,0,0,0,0],
    'D': [0,0,0,0,1,0,0,1,0,1,1,1,1,1,0,0]
}
def generateSSQScore(df, output):
    ssq_df = pd.DataFrame(columns=['userid','gender','seq',"app",'sick','nausea','oculomotor','disorientation',"total",
                                    '1. General discomfort', '2. Fatigue',
                                    '3. Headache', '4. Eyestrain', '5. Difficulty focusing',
                                    '6. Increased salivation', '7. Sweating', '8. Nausea',
                                    '9. Difficulty concentrating', '10. Fullness of head',
                                    '11. Blurred vision', '12. Dizziness (eyes open)',
                                    '13. Dizziness (eyes closed)', '14. Vertigo*',
                                    '15. Stomach awareness**', '16. Burping'])
    i = 0
    for index, row in df.iterrows():
        df[df['UserID'] == row['UserID']].fillna('')
        for x in range(1,10):
            suffix = str(x) if x>1 else ''
            data = row[[q + suffix for q in questions]]
            N = sum(data*weights_SSQ['N'])
            O = sum(data*weights_SSQ['O'])
            D = sum(data*weights_SSQ['D'])
            sick = row['Are you motion sick now?'+suffix] == "Yes"
            comments = row['Supervisor: Comments about this section?' + suffix]
            if type(comments) != str or 'Pistol Whip' not in comments:
                app = ['Begin','HL: Alyx','HL: Alyx','HL: Alyx','Wrench','Wrench','Wrench','Pistol Whip','Pistol Whip','Pistol Whip'][x-1]
            else:
                app = 'Pistol Whip' # exception for user07 who changed order due to sickness
            cont = row['Continue the experiment?\n'+suffix]
            ssqrow = {
                'userid': "user{:02d}".format(index+1),
                'gender': row['Your gender\n'],
                'seq': x,
                'app': app,
                'sick': sick,
                'nausea' : np.round(N * 9.54,2),
                'oculomotor' : np.round(O * 7.58,2),
                'disorientation' : np.round(D * 13.92,2),
                'total' : np.round((N+O+D) * 3.74,2),
            }
            for q in questions:
                ssqrow[q] = row[q + suffix]
            ssq_df.loc[i] = ssqrow
            i += 1
            if x<10 and 'No > ' in cont:
                break
    ssq_df.to_csv(output / 'ssq_scores.csv', index=False)

if __name__ == '__main__':
    main()