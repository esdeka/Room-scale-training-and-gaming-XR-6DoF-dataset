########### Reads all raw Brekel csv from Data subfolders and generates plots in figures/ ########
###### Current state: can only import HMD0 data, generates several plots for each scenario and for each game's combined data 

import re
import shutil
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib as mpl
from matplotlib import markers, colors, ticker, pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('****************xr analysis************************')

# pd.set_option('display.float_format', lambda x: '%.2f' % x)
# plt.rcParams.update({'font.size': 5})
# default plt.rcParams['font.size'] = 10
defaultfontsize = 8

# File parameters:
input = Path('.') / 'analysis' / 'Data-output'
df_volunteers = pd.read_csv(input / 'volunteers.csv')
df_label = pd.read_csv(input / 'recordings.csv')
df_ssq = pd.read_csv(input / 'ssq_scores.csv')
output = Path('.') / 'analysis' / 'Data-figures'

output_overview = Path(output / 'Overview plots')
output_volunteer = Path(output / 'Volunteer plots')
output_single = Path(output / 'Recording plots')
output_thesispdf = Path(output / 'Thesis pdf')
output_thesispng = Path(output / 'Thesis png')
path_hdf = input / 'alldata.h5'

# load dataset files
do_singleplots, do_singleplots_dupes, do_calib, do_tpose, do_overviewplots, do_writehdf, do_readnplothdf,do_onlythesis = False, False, False, False, False, False, False, False

# This section can be used to select specific plot types, comment out plots or edit users that shouldn't be executed

do_singleplots = True
# do_singleplots_dupes = True
do_calib = True
do_tpose = True
do_overviewplots = True

# do_writehdf = True        # Create a HDF database with a dataframe combining all tracking samples, takes an hour at least
# do_readnplothdf = True

do_onlythesis = True        # Only do the plots that are used in the thesis

do_plots = {
    'test' : True,
    'users' : 33,
    # "userlist": np.arange(1,33+1), 
    "userlist": [
                    # 1,3,17,18,27,33,
                    3, 4, #volunteercalib
                    14, 22, 29, #update_delay_boxplot
                    2, #devices_velocities_CCDF_thesis
                    4,7,8, #ssq
                    17,18, #hmd_relativedev
                ], 
    'single': [     # needs do_singleplots = True
        'update_delay_boxplot',     # 1-CDF of pose update delay
        'hmd_relativedev',          # Position of devices compared to HMD
        'devices_velocities_CCDF_thesis',  # 1-CDF of velocities
        
        # 'xz_position',              # Simple plot of HMD x,z position (top-down)
        # 'xz_heatmap',               # Heatmap of HMD x,z position (top-down)
        # 'boxplot V',                # Boxplot of the HMD height and total velocity
        # 'devices_velocities_CCDF',  # 1-CDF of velocities
        # ## Quaternion plots:
        # 'quat_plot',            # quat plot
        # 'quat_vel',             # quat ang vel
        # 'quat_vel_box',         # quat ang vel box
        # 'quat_vel_hist',        # quat ang vel hist
        # 'quat_angvel_ccdf',     # quat ang vel ccdf
    ],
    'user': [
        'ssq',
        'ssqo',
        'volunteerCalib', 
        'calibrmse',
        'tpose',
    ],
    'overview': [
        'delay_scatter_meanstd',
        'plotQualityLabel',
        'metric_cat',
    ],
    'hdf': [
        'hmd_xz_app_heatmap',       # HMD floor position heatmap per app
        'hmd_y_app_boxplot',       # HMD y boxplot per app
        'hmd_app_yawpitchroll_boxplot',       # HMD rot boxplot per app
        'hmd_vms_app_ccdf',         # 1-CDF comparison of HMD lat velocity per app
        'hmd_wq_app_ccdf',          # 1-CDF comparison of HMD ang velocity per app
    ],
}
thesis_list = {
    'delay_scatter_meanstd_': 'H5',
    'cacerumd.user14.record14_4.1_update_delay_boxplot': 'H5',
    'cacerumd.user22.record14_4.1_update_delay_boxplot': 'H5',
    'cacerumd.user29.record15_4.2_update_delay_boxplot': 'H5',
    'cacerumd.user29.record16_4.3_update_delay_boxplot': 'H5',
    'calibrmse': 'H5',
    'caliby': 'H5',
    'calib user03': 'H5',
    'calib user04': 'H5',
    # 'calib user27': 'H5',
    'plotQualityLabel': 'H5',
    'metric_cat': 'H5',
    'metric_cat_volunteers': 'H5',
    'metric_cat_ld_std': 'H5',
    'metric_cat_lv_med': 'H5',
    'metric_cat_ad_std': 'H5',
    'metric_cat_av_med': 'H5',
    # 'ssq user01_ssq': 'H5',
    'ssq user04_ssq': 'H5',
    'ssq user07_ssq': 'H5',
    'ssq app_': 'H5',
    'ssq gender_': 'H5',
    'hmd_xz_app_heatmaps': 'H5',
    'hmd_y_app_boxplot': 'H5',
    'hmd_app_yawpitchroll_boxplot': 'H5',
    'hmd_vms_app_ccdf': 'H5',
    'hmd_wq_app_ccdf': 'H5',
    'cacerumd.user17.record13_3.3_hmd_relativedev': 'H5',
    'cacerumd.user17.record15_4.2_hmd_relativedev': 'H5',
    'cacerumd.user18.record18_5.1_hmd_relativedev': 'H5',
    'tposes': 'H5',
    'cacerumd.user02.record14_4.1_devices_velocities_CCDF_thesis': 'H4',
    'cacerumd.user02.record16_4.3_devices_velocities_CCDF_thesis': 'H4',
    'ssq user08_ssq': 'H3',
}

chapters = ['Calibration', 'Tpose', 'HL: Alyx', 'Wrench', 'Pistol Whip']
device_types = [
    'HMD',                  # devicetype = 0
    'Controllers',          # devicetype = 1
    'Generic Trackers',     # devicetype = 2  Tundra
    'Tracking references',  # devicetype = 3  Base station
]

device_shorttypes = [
    'HMD',
    'Contr',
    'Tundra',
    'BaseSt',
]


def main():
    # variables to store data from multiple files
    games = []
    plt.rcParams['font.size'] = defaultfontsize

    if do_writehdf:
        # df_all.to_hdf(path_hdf, 'df_all')
        hdf = pd.HDFStore(path_hdf, mode='w')
    else:
        hdf = False
    
    for u in range(1,do_plots['users']+1):
        inputfolder = input / 'recordings'
        file_filter = f'cacerumd.user{"{:02d}".format(u)}.record*[[]0[]].parquet'   # filter for files ending with [0].parquet
        logging.info(inputfolder / f'cacerumd.user{"{:02d}".format(u)}')
        # Iterate over all Brekel files in datafolter
        files = Path(inputfolder).rglob(file_filter)
        for f in files:
            # logging.info('%s %s', f.parent.relative_to(inputfolder), f.name)
            # Path(outputfolder / f.parent.relative_to(inputfolder)).mkdir(parents=True, exist_ok=True)
            
            filemeta = openFile(f,u, hdf)

            # deep dive single plots, each plotted per scenario per file
            if filemeta['do_single']:
                output_single.mkdir(parents=True, exist_ok=True)
                for i in [
                    #[<function to generate plot>, <plot filename>]

                    ## translation plots
                    [xz_plot,'xz_position'],                            # Simple plot of HMD x,z position (top-down)
                    [xz_heatmap,'xz_heatmap'],                          # Heatmap of HMD x,z position (top-down)
                    [posbox,'boxplot V'],                               # Boxplot of the HMD height and total velocity
                    [velccdf, 'devices_velocities_CCDF',(2,1)],         # 1-CDF of velocities
                    [velccdfthesis, 'devices_velocities_CCDF_thesis'],         # 1-CDF of velocities
                    [delayboxplot, 'update_delay_boxplot', (2,1)],      # 1-CDF of pose update delay

                    ## Quaternion plots:
                    [quatplot, 'quat_plot'],                  # quat plot
                    [quatvel, 'quat_vel'],                    # quat ang vel
                    [quatvelbox, 'quat_vel_box'],             # quat ang vel box
                    [quatvelhist, 'quat_vel_hist'],           # quat ang vel hist
                    [angvelccdf, 'quat_angvel_ccdf',(4,2)],   # quat ang vel ccdf

                ]:
                    plotname = i[1]
                    if plotname in do_plots['single']:
                        plotfunction = i[0]
                        nrows = i[2][0] if len(i) > 2 else 1
                        ncols = i[2][1] if len(i) > 2 else 1
                        singlefig(plotfunction, filemeta['df'], filemeta['title'], plotname, filemeta, filemeta['devices'], output_single, nrows=nrows, ncols=ncols )
                if 'hmd_relativedev' in do_plots['single']:
                    hmd_relativedev (filemeta, plotfilename='hmd_relativedev', output=output_single)


        
        # do plots for volunteer
        volunteerplots(filemeta, output_volunteer, dfcalib)

    if do_writehdf:
        hdf.close()
        
    overviewplots(output_overview, dfdelaymeanstd, dfmetrics)

    if do_readnplothdf:
        hdf = pd.HDFStore(path_hdf, mode='r')
        hdfplots(hdf, output_overview)

    return
    # After collecting data from all csv files, make some plots comparing use cases
    if True:
        logging.info('Comparing:', games)
        do_gameplots=True
        if do_gameplots:
            for (func, title) in [
                #[<function to generate plot>, <plot filename>]
                [gameslocxz, 'gameslocxz'],
                [gamesangvel, 'gamesangvel'],
                [gamesangvel1cdf, 'gamesangvel1cdf'],
            ]:
                singlefig(func, combinedgamedata, title, 'all_scenarios', 'devices')

def openFile(f,u, hdf):
    # Parse filenames for section
    # cacerumd.user01.record01_1.1_[0]
    # cacerumd.user01.record10_2.1_[0]
    # cacerumd.user01.record11_3.1_[0]
    # cacerumd.user01.record17_5.1_[0]
    pattern = re.compile(   r"""cacerumd\.
                            user(?P<volunteer>\d\d)\.                    # volunteer
                            record(?P<record>\d\d)_                    # volunteer
                            (?P<chapter>\d)\.(?P<counter>\d)_     # chapter (+ counter)
                            \[(?P<devicetype>\d)]\.parquet  #  and devicetype
                            """, re.VERBOSE)
    match = pattern.match(f.name)
    filemeta = {'file': f.name, 'userid': match.group('volunteer')}
    for i in ['volunteer','record','chapter','counter','devicetype']:
        filemeta[i] = match.group(i)
    filemeta['volunteer'] = 'user' + filemeta['volunteer']
    filemeta['game'] = chapters[int(filemeta['chapter'])-1]
    filemeta['scenario'] = f"{filemeta['record']}-{filemeta['counter']}"
    # logging.info('%s', filemeta)
    filemeta['title'] = f"{filemeta['volunteer']}({filemeta['record']}) {filemeta['game']}({filemeta['counter']})"
    filemeta['filename'] = f.name.split('[')[0][:-1]
    filemeta['recording nr'] = filemeta['record']

    chapter = int(filemeta['chapter'])
    filemeta['do_overview'] = chapter >= 3 and do_overviewplots
    filemeta['do_single'] = chapter >= 3 and do_singleplots and (u in do_plots['userlist'])
    filemeta['do_calib'] = chapter == 1 and do_calib
    filemeta['do_tpose'] = chapter == 2 and do_tpose
    filemeta['do_hdf'] = do_writehdf

    if not (filemeta['do_overview'] or filemeta['do_single'] or filemeta['do_calib'] or filemeta['do_tpose'] or filemeta['do_hdf']):
        return filemeta
    
    logging.info('%s', filemeta['filename'])
    readCSVdevices(f, filemeta)

    if filemeta['do_overview']:
        for label, dev in filemeta['devices'].items():
            if dev['devicetype'] >= 3:
                continue
            processDeviceStats(f, filemeta, dev)
    if filemeta['do_calib']:
        processCalibrationFile(f, filemeta, filemeta['devices'])
    if filemeta['do_tpose']:
        processTposeFile(f, filemeta, filemeta['devices'])
    if filemeta['do_hdf']:
        processHDF(f, filemeta, hdf)
    return filemeta


def readCSVdevices(f, filemeta):
    # Read (compressed) csv
    df = pd.read_parquet( f ) 
    devices = {}
    for devicetype in [2,1,0,3]:
        devicefilename = f.with_name(f.name.replace('[0]', f'[{devicetype}]'))
        if devicefilename.is_file():
            df_i =  pd.read_parquet (devicefilename)
            deviceids = df_i.index.get_level_values('deviceid').unique().array
            if devicetype != 3:
                for id in deviceids:
                    name = device_shorttypes[devicetype]
                    if len(deviceids) > 1:
                        name = name + f'{id}'
                    df_ = df_i.loc[(slice(None), id), :].reset_index()
                    df_['time_step'] = df_['timestamp_s'].diff()
                    df_.at[0,'time_step'] = df_.at[0,'timestamp_s']
                    df_.set_index(['timestamp_s','deviceid'], inplace=True)
                    devices[name] = {"devicetype": devicetype, "deviceid": id, "name": name, "df": df_}
            else: #don't split basestations
                devices[device_shorttypes[devicetype]] = {"devicetype": devicetype, "df": df_i}
    filemeta['df'] = df
    filemeta['devices'] = devices

# summary data from all recordings 
dfdelaymeanstd = pd.DataFrame(columns=['file','room','mean', 'std', 'devicetype', 'game'])
dfmetrics = pd.DataFrame(columns=['file','ld_std','lv_med','ad_std','av_med','count', 'devicetype', 'game', 'exp', 'mw', 'age', 'sport', 'ssq'])
# dfvel = pd.DataFrame(columns=['v_ms','App', 'UserID'])
def processDeviceStats(f, filemeta, device):
    df = device["df"]
    form = df_volunteers[df_volunteers['userid'] == filemeta['volunteer']]
    ssq = df_ssq[(df_ssq["userid"] == filemeta['volunteer']) & (df_ssq['app'] == chapters[int(filemeta['chapter'])-1])]
    dfdelaymeanstd.loc[len(dfdelaymeanstd)] = {
        'file': f.name, 
        'room': form.iloc[0]['medium'], 
        'devicetype': device["devicetype"],
        'game': filemeta['game'],
        'mean': df['time_step'].mean(), 
        'std': df['time_step'].std()
    }
    # mean std avg lateral distance from center
    Xc, Zc = df["tx_m"].mean(), df["tz_m"].mean()   # average center Xc, Zc
    latdist = np.sqrt((Xc-df["tx_m"])**2 + (Zc-df["tz_m"])**2)
    # average quaternion https://stackoverflow.com/questions/12374087/average-of-multiple-quaternions
    eig = np.linalg.eig(df[['qx','qy','qz','qw']].transpose().dot(df[['qx','qy','qz','qw']]))
    qx, qy, qz, qw = eig[1][:,np.argmax(eig[0])]
    # quaternion distance from avg quaternion https://math.stackexchange.com/questions/90081/quaternion-distance
    angdist = np.rad2deg( np.arccos(2*(qw*df.qw + qx*df.qx + qy*df.qy + qz*df.qz)**2 - 1) )
    dfmetrics.loc[len(dfmetrics)] = {
        'file': f.name, 
        'ld_std': latdist.std()*100,
        'lv_med': df['v_ms'].median()*100,
        'ad_std': angdist.std(),
        'av_med': df['wq_degs'].median(),
        'count': len(df.index),
        'devicetype': device["devicetype"],
        'game': filemeta['game'],
        'exp': form.iloc[0]['xr_experience'],
        'mw': form.iloc[0]['gender'],
        'age': form.iloc[0]['age'],
        'sport': int(form.iloc[0]['sportive']),
        'ssq': ssq['total'].mean(),
    }

# allc1 = ['deviceid', 'timestamp_s', 'devicetype', 'tx_m', 'ty_m', 'tz_m', 'v_ms', 'vx_ms', 'vy_ms', 'vz_ms','qx', 'qy', 'qz', 'qw', 'wq_degs', 'wqx_degs', 'wqy_degs', 'wqz_degs'] 
allcc = ['System', 'ApplicationMenu', 'Grip', 'DPad_Left', 'DPad_Up', 'DPad_Right', 'DPad_Down', 'A', 'ProximitySensor', 'Axis0', 'Axis1', 'Axis2', 'Axis3', 'Axis4', 'axis0_X', 'axis0_Y', 'axis1_X', 'axis1_Y', 'axis2_X', 'axis2_Y', 'axis3_X', 'axis3_Y', 'axis4_X', 'axis4_Y']
allcbs = ['v_ms', 'vx_ms', 'vy_ms', 'vz_ms','wq_degs', 'wqx_degs', 'wqy_degs', 'wqz_degs']
# allc2 = ['userid', 'record', 'chapter', 'counter']
# allc = allc1.copy()
# allc.extend(allc2)
# df_all = pd.DataFrame(columns=allc)
def processHDF(f, filemeta, hdf):
    for label, dev in filemeta['devices'].items():
        if dev['devicetype'] <= 3:
            # df = dev['df'].reset_index(drop=True)[allc1]
            df = dev['df'].reset_index() #[allc1]
            if dev['devicetype'] == 1:
            #     df = df.reindex(columns = df.columns.tolist() + allcc)
                df = df.drop(columns=allcc)
            if dev['devicetype'] == 3:
                df = df.reindex(columns = df.columns.tolist() + allcbs)
            else:
                df = df.drop(columns=['time_step'])
            df['userid'] = int(filemeta['userid'])
            df['record'] = int(filemeta['record'])
            df['chapter'] = int(filemeta['chapter'])
            df['counter'] = int(filemeta['counter'])
            df.set_index(['userid', 'record', 'deviceid', 'timestamp_s'])
            df.describe()
            hdf.append('df_all', df, format='table',  data_columns=True )

dfcalib = pd.DataFrame(columns=['userid','seq','dev','devicetype','deviceid','point', 'x','y','z','stddev','mean','mse_','len_'])
calibpoints = [
    [0,0],
    [-1,0],
    [-1,1],
    [0,1],
    [1,1],
    [1,0],
    [1,-1],
    [0,-1],
    [-1,-1],
    [29,29]     #fake point
]
def processCalibrationFile(f, fm, devices):
    for label, dev in devices.items():
        if dev['devicetype'] >= 3:
            continue
        df = dev['df'][['tx_m','ty_m','tz_m']]
        for point, c in enumerate(calibpoints):
            if (c[0] == int(df.tx_m.mean().round())) and (c[1] == int(df.tz_m.mean().round())):
                dist = ((c[0] - df.tx_m)**2 + (c[1] - df.tz_m)**2)**0.5
                dfcalib.loc[len(dfcalib)] = {
                    'userid': fm['userid'],
                    'seq': fm['recording nr'],
                    'point':point+1,
                    'dev': label,
                    'devicetype': dev['devicetype'],
                    'deviceid': dev['deviceid'],
                    'stddev': dist.std(),
                    'mean':dist.mean(),
                    'mse_':dist.sum(),
                    'len_':len(df),
                    'x':df.tx_m.mean(),
                    'y':df.ty_m.mean(),
                    'z':df.tz_m.mean()
                }
                break
            if point > 8:
                print('No close calibration point found, this should not happen!')

dftpose = pd.DataFrame(columns=['userid','devicetype','deviceid', 'x','y','z'])
def processTposeFile(f, fm, devices):
    for label, dev in devices.items():
        if dev['devicetype'] >= 3:
            continue
        df = dev['df'][['tx_m','ty_m','tz_m']]
        means = {
            'userid': int(fm['userid']),
            'devicetype': dev['devicetype'],
            'deviceid': dev['deviceid'],
            'x':df.tx_m.mean(),
            'y':df.ty_m.mean(),
            'z':df.tz_m.mean()
        }
        dftpose.loc[len(dftpose)] = means
        logging.info("processTposeFile %s", means)


def savefig(fig, suptitle='title', plotfilename='plotfilename', type='overview', outputfig=output, dosubaxplots=True, tight_layout=True, ax_title = False, pdfdpi=200):
    logging.info("%s %s", type, plotfilename)
    fig.suptitle(suptitle)
    if tight_layout:
        fig.tight_layout()

    outputfig.mkdir(parents=True, exist_ok=True)
    figsavepath = outputfig / f'{plotfilename}.png'
    fig.savefig( figsavepath, dpi=600 )

    if plotfilename in thesis_list:
        logging.info("--- Thesis")
        output_thesispdf.mkdir(parents=True, exist_ok=True)
        output_thesispng.mkdir(parents=True, exist_ok=True)
        fig.suptitle("")
        for i,ax in enumerate(fig.axes):
            ax.title.set_visible(ax_title)
            if not dosubaxplots:
                continue
            logging.info("--- Thesis ax %s png file: %s", i, plotfilename)
            fig.savefig( output_thesispng / f'{plotfilename}_{i}.png', bbox_inches=ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted() ),pad_inches = 0.1) #https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib/26432947#26432947
            logging.info("--- Thesis ax %s pdf file: %s", i, plotfilename)
            fig.savefig( output_thesispdf / f'{plotfilename}_{i}.pdf', format="pdf", bbox_inches=ax.get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted() ),pad_inches = 0.1, dpi=pdfdpi) #https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib/26432947#26432947
            # fig.savefig( output_thesispdf / f'{plotfilename}_{i}.pdf', format="pdf", bbox_inches=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.1, 1.2) )
        logging.info("--- Thesis png file: %s", plotfilename)
        fig.savefig( output_thesispng / f'{plotfilename}.png', bbox_inches='tight',pad_inches = 0.1, dpi=600  )
        logging.info("--- Thesis pdf file: %s", plotfilename)
        fig.savefig( output_thesispdf / f'{plotfilename}.pdf', format="pdf",bbox_inches='tight',pad_inches = 0.1, dpi=pdfdpi )
        
    plt.rcParams.update({'font.size': defaultfontsize})
    plt.close()

# singlefig utility function: saves a figure to a .png file
def singlefig(func, df, figtitle, plotfilename, filemeta, devices, output, nrows=1, ncols=1, single=True, fontsize=defaultfontsize):
    deffilename = filemeta['filename'] + f'_{plotfilename}'
    if do_onlythesis and not deffilename in thesis_list:
        return
    
    logging.info("%s %s", figtitle, plotfilename)
    plt.rcParams.update({'font.size': fontsize})
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    func (df, fig, ax, figtitle, devices)
    fig.tight_layout()
    output_title = Path(output / plotfilename)
    output_title.mkdir(parents=True, exist_ok=True)
    figsavepath = output_title / f'{deffilename}.png'
    fig.savefig( figsavepath, dpi=600 )

    if deffilename in thesis_list:
        output_thesispdf.mkdir(parents=True, exist_ok=True)
        logging.info("--- Thesis pdf file: %s", deffilename)
        shutil.copy(figsavepath, output_thesispng)
        for ax in fig.axes:
            ax.title.set_visible(False)
        fig.suptitle("")
        fig.savefig( output_thesispdf / f'{deffilename}.pdf', format="pdf" )

    plt.close()
    plt.rcParams['font.size'] = defaultfontsize
    plt.rcParams.update({'font.size': defaultfontsize})

    if single and do_singleplots_dupes:
        output_abpt = Path(output / 'all by plot type')
        output_abpt.mkdir(parents=True, exist_ok=True)
        output_abpt_path = output_abpt / (f"{plotfilename}_{filemeta['filename']}.png")
        shutil.copy(figsavepath, output_abpt_path)

        output_abpt = Path(output / 'all by plot type and chapter.counter')
        output_abpt.mkdir(parents=True, exist_ok=True)
        output_abpt_path = output_abpt / (f"{plotfilename}_{filemeta['chapter']}.{filemeta['counter']}.{filemeta['game']}.{filemeta['scenario']}_{filemeta['volunteer']}.png")
        shutil.copy(figsavepath, output_abpt_path)

        output_abgc = Path(output / 'all')
        output_abgc.mkdir(parents=True, exist_ok=True)
        output_abgc_path = output_abgc / (figsavepath.name)
        shutil.copy(figsavepath, output_abgc_path)

        if plotfilename in ['devices_velocities_CCDF','hmd_relativedev','quat_angvel_ccdf','update_delay_boxplot','xz_position']:
            output_abr = Path(output / 'all for rating')
            output_abr.mkdir(parents=True, exist_ok=True)
            output_abr_path = output_abr / (figsavepath.name)
            shutil.copy(figsavepath, output_abr_path)


prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
hmd_color, contr1_color, tundra_color, contr2_color = 6, 9, 8, 5
dcolors = {
    'HMD': prop_colors[hmd_color],
    'HMD0': prop_colors[hmd_color],
    'Contr': prop_colors[contr1_color],
    'Contr3': prop_colors[contr1_color],
    'Contr5': prop_colors[contr1_color],
    'Contr7': prop_colors[contr1_color],
    'Contr9': prop_colors[contr1_color],
    'Tundra': prop_colors[tundra_color],
    'Tundra0': prop_colors[tundra_color],
    'Contr2': prop_colors[contr2_color],
    'Contr4': prop_colors[contr2_color],
    'Contr6': prop_colors[contr2_color],
    'Contr8': prop_colors[contr2_color],
}

# Root overview plots
def hdfplots (hdf, output_overview):
    # if do_readnplothdf:
    output_overview.mkdir(parents=True, exist_ok=True)
    if 'hmd_xz_app_heatmap' in do_plots['hdf']:
        hmd_xz_app_heatmap(hdf, output_overview)
    if 'hmd_y_app_boxplot' in do_plots['hdf']:
        hmd_y_app_boxplot(hdf, output_overview)
    if 'hmd_app_yawpitchroll_boxplot' in do_plots['hdf']:
        hmd_app_yawpitchroll_boxplot(hdf, output_overview)
    if 'hmd_wq_app_ccdf' in do_plots['hdf']:
        hmd_wq_app_ccdf(hdf, output_overview)
    if 'hmd_vms_app_ccdf' in do_plots['hdf']:
        hmd_vms_app_ccdf(hdf, output_overview)

def hmd_wq_app_ccdf(hdf, output):
    suptitle = f"CCDF HMD angular velocity per app"
    logging.info("%s", suptitle)

    fig, ax = plt.subplots()

    for i in [3,4,5]:
        logging.info("Select HDF chapter %s", i)
        df = hdf.select('df_all', f"chapter = {i} & devicetype = 0 & columns=['wq_degs']")
        for col in ['wq_degs']:
            ax.hist(df[col], bins=400000, density=True, histtype='step', cumulative=-1,label=f'{chapters[i-1]}', linewidth=2)

    ax.set(xlabel='HMD wq_degs (deg/s)')
    ax.set(ylabel=r'CCDF( HMD $w_{deg/s}$ )')

    ax.set_xscale('log')
    # ax.set_xticks([0.01, 0.1,1,10,25, 50, 100, 400])
    ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.set_xlim(left=1, right=2000) 

    ax.set_yscale('log')
    ax.legend(fontsize='small')
    # ax.grid(which='minor', color='#dddddd', linestyle = '--', lw=0.25)
    ax.grid(which='major', color='#bbbbbb', linestyle = '-.', lw=0.5)
    ax.set_axisbelow(True)

    savefig(fig=fig, suptitle=suptitle, plotfilename=f"hmd_wq_app_ccdf", outputfig=output, type='overview', dosubaxplots=False)


def hmd_vms_app_ccdf(hdf, output):
    suptitle = f"CCDF HMD lateral velocity per app"
    logging.info("%s", suptitle)

    fig, ax = plt.subplots()

    for i in [3,4,5]:
        logging.info("Select HDF chapter %s", i)
        df = hdf.select('df_all', f"chapter = {i} & devicetype = 0 & columns=['v_ms']")
        for col in ['v_ms']:
            ax.hist(df[col], bins=400000, density=True, histtype='step', cumulative=-1,label=f'{chapters[i-1]}', linewidth=2)

    ax.set(xlabel='HMD v_ms (m/s)')
    ax.set(ylabel=r'CCDF( HMD $v_{s}$ )')

    ax.set_xscale('log')
    # ax.set_xticks([0.01, 0.1,1,10,25, 50, 100, 400])
    ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.set_xlim(left=0.01, right=100) 

    ax.set_yscale('log')
    ax.legend(fontsize='small')
    # ax.grid(which='minor', color='#dddddd', linestyle = '--', lw=0.25)
    ax.grid(which='major', color='#bbbbbb', linestyle = '-.', lw=0.5)
    ax.set_axisbelow(True)

    savefig(fig=fig, suptitle=suptitle, plotfilename=f"hmd_vms_app_ccdf", outputfig=output, type='overview', dosubaxplots=False)

def hmd_xz_app_heatmap(hdf,output):
    suptitle = f"HMD X-Z normalised heatmap"
    plt.rcParams.update({'font.size': 5})

    # https://stackoverflow.com/questions/23270445/adding-a-colorbar-to-two-subplots-with-equal-aspect-ratios/66987579#66987579
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(1, len(chapters)-2, wspace=0.05)        
    axes = [fig.add_subplot(gs[0, col], aspect="equal") for col in range(len(chapters)-2)]

    for col, ax in enumerate(axes):
        suptitle_ = f"{chapters[col+2]}"
        ax.set_title(suptitle_)

        df = hdf.select('df_all', f"chapter = {col+3} & devicetype = 0 & columns=['tx_m','tz_m']")
        logging.info("%s %s", suptitle_, len(df))

        norm = colors.LogNorm()
        # norm = colors.LogNorm(vmin=1, vmax=3*(10**4), clip=False)
        hh = ax.hist2d( df['tx_m'], df['tz_m'],bins=[np.arange(-2,2,0.05),np.arange(-2,2,0.05)], cmap = 'BuPu', norm = norm)

        if col == 0:
            ax.set(ylabel='Z (m)')
        else:
            ax.set_yticklabels([])
            ax.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False, # labels along the bottom edge are of
                labelright=False # labels along the bottom edge are of
            )
        ax.set(xlabel='X (m)')

        ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.set_xticks([-2,-1,0,1, 2 ])
        ax.set_yticks([-2,-1,0,1, 2 ])
        ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
        ax.grid(which='minor', color='#dddddd', linestyle = '-.', lw=0.25)
        ax.set_axisbelow(True)

    cb = fig.colorbar(hh[3], ax=axes, pad=0.01, shrink=0.4)
    cb.set_ticklabels([])
    cb.set_label('# of samples in 2D grid')

    savefig(fig=fig, suptitle=suptitle, plotfilename=f"hmd_xz_app_heatmaps", outputfig=output, type='overview', dosubaxplots=False, tight_layout=False, ax_title = True)

def hmd_y_app_boxplot(hdf,output):
    suptitle = f"HMD tx_y boxplot"
    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots()

    xtickslabels = []
    for i,app in enumerate(['HL: Alyx', 'Wrench', 'Pistol Whip']):
        df = hdf.select('df_all', f"chapter = {i+3} & devicetype = 0 & columns=['ty_m']")
        logging.info("%s %s %s", suptitle, app, len(df))
        xtickslabels.append(app)
        ax.boxplot(df['ty_m']*100, positions=[len(xtickslabels)], notch = True, showmeans = True, meanline = True, showfliers=False)
    ax.set_xticks(np.arange(1, len(xtickslabels)+1), xtickslabels)

    ax.set(ylabel='HMD ty_m (cm)')
    ax.grid(which='major', color='#dddddd', linestyle = '--', lw=0.5, axis='y')
    ax.set_axisbelow(True)

    savefig(fig=fig, suptitle=suptitle, plotfilename=f"hmd_y_app_boxplot", outputfig=output, type='overview', dosubaxplots=False)

def hmd_app_yawpitchroll_boxplot(hdf,output):
    suptitle = f"HMD yaw, pitch, roll boxplot"
    # plt.rcParams.update({'font.size': 15})
    logging.info("%s ", suptitle )

    fig, axs = plt.subplots(1,3)
    xtickslabelscols = {
        'Yaw': [],
        'Pitch': [],
        'Roll': [],
    }
    for i,app in enumerate(['HL: Alyx', 'Wrench', 'Pistol Whip']):

        eulers = (Rotation.from_quat((hdf.select('df_all', f"chapter = {i+3} & devicetype = 0 & columns=['qx','qy','qz','qw']")[['qx','qy','qz','qw']]).values)).as_euler('ZYX', degrees=True)
        # eulers = (Rotation.from_quat((hdf.select('df_all', f"chapter = {i+3} & devicetype = 0 & columns=['qx','qy','qz','qw']")[['qx','qy','qz','qw']]).values) * Rotation.from_euler('Z', 180, degrees=True)).as_euler('ZYX', degrees=True)
        logging.info("-%s %s", app, len(eulers) )
        # df = hdf.select('df_all', f"chapter = {i+3} & devicetype = 0 & columns=['qx','qy','qz','qw']")[['qx','qy','qz','qw']]
        # r = Rotation.from_quat(df.values)
        # vals_aligned = r * Rotation.from_euler('Z', 180, degrees=True)
        # eulers = vals_aligned.as_euler('ZYX', degrees=True)

        for x, col in enumerate(['Yaw', 'Pitch', 'Roll']):
            ax = axs[x]
            xtickslabels = xtickslabelscols[col]

            logging.info("--%s ", col )
            ax.boxplot(eulers[:, x].tolist(), positions=[len(xtickslabels)], notch = True, showmeans = True, meanline = True, showfliers=False)
            xtickslabels.append(f'{app}')
            ax.set_xticks(np.arange(0, len(xtickslabels)), xtickslabels, rotation='vertical')
            ax.set(ylabel=f'HMD {col} (degrees)')
            ax.grid(which='major', color='#dddddd', linestyle = '--', lw=0.75, axis='y')
            ax.set_axisbelow(True)
            ax.set_title(col)

    savefig(fig=fig, suptitle=suptitle, plotfilename=f"hmd_app_yawpitchroll_boxplot", outputfig=output, type='overview', dosubaxplots=False)




def overviewplots(output_overview, dfdelaymeanstd, dfmetrics):
    if do_overviewplots:
        dfdelaymeanstd.to_csv(output / 'dfdelaymeanstd.csv')
        dfmetrics.to_csv(output / 'dfmetrics.csv')
    else:
        dfdelaymeanstd = pd.read_csv(output / 'dfdelaymeanstd.csv')
        dfmetrics = pd.read_csv(output / 'dfmetrics.csv')
    output_overview.mkdir(parents=True, exist_ok=True)
    if 'plotQualityLabel' in do_plots['overview']:
        plotQualityLabel(df_label, output_overview)
    if 'delay_scatter_meanstd' in do_plots['overview']:
        singlefig(delayscatter, dfdelaymeanstd, 'Delay mean/std/room', '', {'filename': 'delay_scatter_meanstd'}, {}, output_overview, single=False )
    if 'metric_cat' in do_plots['overview']:
        metric_cat (dfmetrics, plotfilename='metric_cat', output=output_overview)

categories = ['exp', 'age', 'sport', 'ssq']
cat_titles = ['Experience in VR rated from 0-10', 'Age in years', 'Sportive rated from 0-5', 'SSQ total motion sickness per app']
cat_ylabel = ['VR experience [0-10]', 'Age (years)', 'Sportive [0-5]', 'Total SSQ score']
cat_ymajloc = [2, 5, 1, 25]
cat_df = [df_volunteers, df_volunteers, df_volunteers, df_ssq]
cat_df_col = ['xr_experience','age','sportive', 'total']
# cat_bins = [12,5, 5, 10]
# cat_bins = [range(11),range(15,45,5), range(1,6), range(0,150,25)]
cat_xlabel = ['Number of volunteers','Number of volunteers','Number of volunteers','Number of SSQ forms']
cat_yticks = [1,2,2,25]
cat_bins = [np.arange(0,10+2)-0.5,np.arange(15,45, 5), np.arange(1,7)-0.5,None]
metrics = ['ld_std','lv_med','ad_std','av_med']
met_xlabel = [r'$\sigma$ lat. dist. (cm)', r'$P_{50}$ lat. vel. (cm/s)', r'$\sigma$ ang. dist. ($\degree$)', r'$P_{50}$ ang. vel. ($\degree$\s)']
met_suptitle = [r"$\sigma$ lat. dist. from average center on floor", r'Median $P_{50}$ lateral velocity', r"$\sigma$ ang. dist. from average quaternion", r'Median $P_{50}$ angular velocity']
def metric_cat(df, plotfilename, output):
    markers = ['|','_', 'x','.', '*']
    plt.rcParams['font.size'] = 9

    # Metrics
    fig, axs = plt.subplots(2,2)
    suptitle = 'Metrics per app'
    for i, metric in enumerate(metrics):
        ax = axs[int(np.floor(i/2))][i%2]
        xtickslabels = []
        for game in ['HL: Alyx','Wrench','Pistol Whip']:
            xtickslabels.append(game)
            d = df[(df.game==game)&(df.devicetype==0)][metric]
            ax.boxplot(d, positions=[len(xtickslabels)], notch = True, showmeans = True, meanline = True)
        ax.set_xticks(np.arange(1, len(xtickslabels)+1), xtickslabels)
        ax.set_title(met_suptitle[i])
        ax.set(ylabel=met_xlabel[i])
        ax.grid(which='major', color='#dddddd', linestyle = '--', lw=0.5, axis='y')
        ax.set_axisbelow(True)
    # plt.show()
    savefig(fig=fig, suptitle=suptitle, plotfilename=f'{plotfilename}', outputfig=output, type='overview')

    # Categories
    fig, axs = plt.subplots(2,2)
    suptitle = 'Categories volunteers'
    for i, cat in enumerate(categories):
        ax = axs[int(np.floor(i/2))][i%2]
        ax.hist(cat_df[i][cat_df_col[i]], edgecolor='black', bins=cat_bins[i])
        ax.set(xlabel=cat_ylabel[i], ylabel = cat_xlabel[i])
        ax.yaxis.set_major_locator(plt.MultipleLocator(cat_yticks[i]))
        ax.grid(which='major', color='#dddddd', linestyle = '--', lw=0.5, axis='y')
        ax.set_axisbelow(True)
    # plt.show()
    savefig(fig=fig, suptitle=suptitle, plotfilename=f'{plotfilename}_volunteers', outputfig=output, type='overview')

    # Metrics x categories
    for k, metric in enumerate(metrics):
        suptitle = met_suptitle[k]
        fig, axs = plt.subplots(2,2)

        for i, cat in enumerate(categories):
            ax = axs[int(np.floor(i/2))][i%2]
            for j, game in enumerate(df['game'].unique()):
                df_game = df[(df['game'] == game) & (df['devicetype']==0)]
                ax.scatter(df_game[metric],df_game[cat], label=game, marker=markers[j], linewidth=0.8, alpha = 0.9)
            ax.legend(fontsize='xx-small')
            ax.set_title(cat_titles[i])
            ax.set(xlabel=met_xlabel[k])
            ax.set(ylabel=cat_ylabel[i])
            ax.grid(which='minor', color='#dddddd', linestyle = '--', lw=0.25)
            ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_locator(plt.MultipleLocator(cat_ymajloc[i]))
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        savefig(fig=fig, suptitle=suptitle, plotfilename=f'{plotfilename}_{metric}', outputfig=output, type='overview')

# Root overview of tracking delays
def delayscatter(df, fig, ax, title, devices):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['|','_', 'x','.', '*']

    legendm = []
    legendc = []
    for i,room in enumerate(df['room'].unique()):
        for j,game in enumerate(df['game'].unique()):
            df_room = df[(df['room'] == room) & (df['game'] == game) & (df['devicetype']==0)].sample(frac=1)
            a = ax.scatter(df_room['mean']*1000, df_room['std'], c=colors[i], label=room, marker=markers[j], linewidth=1, alpha = 0.9, s = mpl.rcParams['lines.markersize'] ** 2.3)
            if i==0:
                legendm.append(a)
        legendc.append(a)
    
    ax.legend()
    ax.set_title('Timestep delay HMD deviation')
    ax.set(xlabel='Mean (ms)')
    ax.set(ylabel='Standard deviation (ms)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1,2,3,4,5,6,7,8,9,10,20,30,40])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.grid(which='minor', color='#dddddd', linestyle = '--', lw=0.25)
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
    ax.set_axisbelow(True)

    legend1 = ax.legend(legendm,df['game'].unique(), loc= 'lower right') 
    ax.add_artist(legend1)
    for h in legend1.legend_handles:
        h.set_color('black')
    ax.legend(legendc,df['room'].unique(), loc= 'upper right') 

def plotQualityLabel(df,output):
    plt.rcParams['font.size'] = 9
    suptitle = "Quality label of recordings"
    fig, axs = plt.subplots(2,1)
    df_ = df.dropna(subset=['HMD_quality'])
    col_rating = ['Contr_quality','Tundra_quality','HMD_quality']
    col_rating_label = ['Contr quality','Tundra quality','HMD quality']
    col_rating_color = [dcolors['Contr'],dcolors['Tundra'],dcolors['HMD']]
    ax = axs[0]
    for col in col_rating:
        ax.step(df_.index, df_[col], label=col.replace('_', ' '), color=dcolors[col.replace('_quality','')])
    ax.legend()
    ax.set_title('Quality of recordings over time')
    ax.set(xlabel='Recording index [i]')
    ax.set(ylabel='Quality rating [0-5]')
    ax.set_yticks(np.arange(0,6))
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5, axis='y')
    ax.set_axisbelow(True)


    ax = axs[1]
    data = []
    col_rating.reverse()
    col_rating_color.reverse()
    col_rating_label.reverse()
    for col in col_rating:
        data.append(df[col])
    ax.hist(data, label=col_rating_label, bins = np.arange(7)-0.5, color=col_rating_color, edgecolor='black')
    ax.legend()
    ax.set_title('Quality of recordings')
    ax.set(xlabel='Quality rating from 0-5')
    ax.set(ylabel='Number of recordings')
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5, axis='y')
    ax.set_axisbelow(True)

    
    savefig(fig=fig, suptitle=suptitle, plotfilename=f'plotQualityLabel', outputfig=output, type='overview')




# Plots per volunteer
def volunteerplots(filemeta, output_volunteer, dfcalib):
    output_volunteer.mkdir(parents=True, exist_ok=True)
    if 'ssq' in do_plots['user'] and (int(filemeta['userid']) in do_plots['userlist']):
        singlefig(volunteerSSQplot, df_ssq, filemeta['volunteer'], 'ssq', {'filename': f"ssq {filemeta['volunteer']}"}, {}, output_volunteer, single=False, fontsize=13 )
    if 'volunteerCalib' in do_plots['user'] and (int(filemeta['userid']) in do_plots['userlist']):
        volunteerCalib (dfcalib, filemeta, output=output_volunteer)
    if int(filemeta['userid']) >= do_plots['users']:
        if do_calib:
            dfcalib.to_csv(output / 'dfcalib.csv')
        else:
            dfcalib = pd.read_csv(output / 'dfcalib.csv')
        if 'ssqo' in do_plots['user']:
            singlefig(ssqapp, df_ssq, 'SSQ app', '', {'filename': f"ssq app"}, {}, output_overview, single=False, fontsize=13 )
            singlefig(ssqgender, df_ssq, 'SSQ gender', '', {'filename': f"ssq gender"}, {}, output_overview, single=False, fontsize=13)
        if 'calibrmse' in do_plots['user']:
            caliby (dfcalib, plotfilename='caliby', output=output_overview)
            calibrmse (dfcalib, plotfilename='calibrmse', output=output_overview)
        if 'tpose' in do_plots['user']:
            tposes (dftpose, plotfilename='tposes', output=output_overview)

def ssqapp(df, fig, ax, title, devices):
    ax.set_title('Total SSQ score boxplots per app')
    # ax.set_ylim([0,200]) 
    ax.set(xlabel='Experiment stage', ylabel='Total SSQ score')
    xtickslabels = []
    for app in ['Begin', 'HL: Alyx', 'Wrench', 'Pistol Whip']:
        xtickslabels.append(app)
        ax.boxplot(df[df['app']==app]['total'], positions=[len(xtickslabels)], notch = True, showmeans = True, meanline = True)
    ax.set_xticks(np.arange(1, len(xtickslabels)+1), xtickslabels)
    ax.grid()
    ax.set_axisbelow(True)
    fig.set_size_inches(fig.get_figwidth()*2*4/6, 4.8)

    set_fontsize=18
    plt.rcParams.update({'font.size': set_fontsize})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(set_fontsize)

def ssqgender(df, fig, ax, title, devices):
    ax.set_title('Total SSQ score boxplots by gender')
    # ax.set_ylim([0,200]) 
    ax.set(xlabel='Gender', ylabel='Total SSQ score')
    xtickslabels = []
    for gender in ['Man', 'Woman']:
        xtickslabels.append(gender)
        ax.boxplot(df[df['gender']==gender]['total'], positions=[len(xtickslabels)], notch = True, showmeans = True, meanline = True)
    ax.set_xticks(np.arange(1, len(xtickslabels)+1), xtickslabels)
    ax.grid()
    ax.set_axisbelow(True)
    fig.set_size_inches(fig.get_figwidth()*2*2/6, 4.8)

    set_fontsize=18
    plt.rcParams.update({'font.size': set_fontsize})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(set_fontsize)

def volunteerSSQplot(df, fig, ax, title, devices):
    ax.set_title('SSQ form scores for ' + title)
    ax.set_ylim([0,200]) 
    ax.set(xlabel='SSQ form #ID', ylabel='SS score')
    for score in ['nausea','oculomotor','disorientation','total']:
        ax.plot(df[df['userid']==title][score], label=score)
    ax.legend(fontsize='small', loc='upper left')
    ax.grid(which='major', color='#dddddd', linestyle = '-', lw=0.25, axis='y')
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5, axis='x')
    ax.set_axisbelow(True)

    set_fontsize=18
    plt.rcParams.update({'font.size': set_fontsize})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(set_fontsize)

def volunteerCalib(dfc, filemeta, output):
    markers = ['x','_', '.','|', '*']
    plt.rcParams['font.size'] = 13
    fig, axs = plt.subplots()
    suptitle = f"Distance error to calibration truth per recording(9) and device(4) ({filemeta['volunteer']})"
    axs.set(xlabel='Mean (mm)', ylabel='Standard deviation (mm)')

    df = dfc[dfc['userid'] == filemeta['userid']]
    ax = axs
    # axs.set_xlim(right=0.125*1000)
    axs.set_xlim(right=0.175*1000)
    # axs.set_xlim(right=0.2*1000)
    # axs.set_ylim(top=0.004*1000)
    axs.set_ylim(top=0.002*1000)
    # axs.set_ylim(top=0.0008*1000)
    for dev in df['dev'].unique():
        dfd = df[df['dev'] == dev]
        ax.scatter(dfd['mean']*1000, dfd['stddev']*1000, label=dev, color=dcolors[dev], marker=markers[dfd['devicetype'].unique()[0]], s = mpl.rcParams['lines.markersize'] ** 3)
    ax.legend(loc='upper left')
    ax.grid(which='minor', color='#dddddd', linestyle = '--', lw=0.25)
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
    ax.set_axisbelow(True)

    savefig(fig=fig, suptitle=suptitle, plotfilename=f"calib {filemeta['volunteer']}", outputfig=output, type='volunteer')

def calibrmse(dfc, plotfilename, output):
    dfr=dfc.drop(['seq'],axis=1).groupby(['userid','devicetype','deviceid','dev']).sum()
    dfr['rmse'] = ((dfr.mse_/dfr.len_) ** 0.5)

    fig, ax = plt.subplots()
    suptitle = f"RMSE from 90s of calibration data per user/device"
    ax.set(ylabel='RMSE (cm)', xlabel='User ID [1 to 33]')
    markers = ['x','_', '.','|', '*']
    ax.set_xticks(np.arange(1,do_plots['users']+1, 2))
    ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.set_ylim(bottom=0, top=40)

    for type in [1,2,0]:
        df = dfr[dfr.index.get_level_values('devicetype')==type]
        ax.scatter(df.index.get_level_values('userid').astype(int), df.rmse * 100, label=device_shorttypes[type], marker=markers[type], color=dcolors[device_shorttypes[type]])

    ax.legend(loc='lower right')
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
    ax.set_axisbelow(True)

    # plt.show()
    savefig(fig=fig, suptitle=suptitle, plotfilename=plotfilename, outputfig=output, type='overview')

def caliby(dfc, plotfilename, output):
    fig, ax = plt.subplots()
    suptitle = f"Heigth averages from 90s of calibration data per user/device"
    ax.set(ylabel='Y (cm)', xlabel='User ID [1 to 33]')
    ax.set_xticks(np.arange(1,do_plots['users']+1, 2))
    ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))

    for type in [1,2,0]:
        df = dfc[dfc.devicetype==type]
        ax.scatter(df.userid.astype(int), df.y * 100, label=device_shorttypes[type], marker='_', color=dcolors[device_shorttypes[type]], alpha=0.8)

    ax.legend(loc='lower right')
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
    ax.set_axisbelow(True)

    # plt.show()
    savefig(fig=fig, suptitle=suptitle, plotfilename=plotfilename, outputfig=output, type='overview')

def tposes (dftpose, plotfilename='tposes', output=output_overview):
    fig, ax = plt.subplots()
    suptitle = f"T-pose height"
    ax.set(ylabel='T-pose Y (cm)', xlabel='User ID [1 to 33]')
    ax.set_xticks(np.arange(1,do_plots['users']+1, 2))
    ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter('%g'))

    userids = np.arange(1,do_plots['users']+1)
    ax.scatter(userids, df_volunteers.height, label='Form height input', marker='*')
    labels = [
        'HMD = eye height',
        'Controllers = armspan',
        'Tundra = hip',
    ]
    for type in [0,1,2]:
        df = dftpose[dftpose.devicetype==type]
        data = df.y
        if (type == 1):
            l = df.groupby('userid').first()
            r = df.groupby('userid').last()
            data = ((l.x-r.x)**2+(l.y-r.y)**2+(l.z-r.z)**2)**0.5
        ax.scatter(userids, data * 100, label=labels[type], marker='_', color=dcolors[device_shorttypes[type]])

    ax.legend()
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.75)
    ax.grid(which='minor', color='#dddddd', linestyle = '-.', lw=0.5, axis='y')
    ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax.set_axisbelow(True)

    savefig(fig=fig, suptitle=suptitle, plotfilename=plotfilename, outputfig=output, type='overview')


# Single figure plots for one csv file
def xz_plot(df, fig, ax, title, devices):
    ax.set_title('X.Z position ' + title)
    ax.set_xlim([-200,200]) 
    ax.set_ylim([-200,200]) 
    for label, dev in devices.items():
        df = dev['df']
        ax.plot( df['tx_m']*100, df['tz_m']*100, alpha=0.5, label=label, linewidth=0.5)
    ax.legend()

# def hmd_relativedev(df, fig, axs, title, devices):
def hmd_relativedev (filemeta, plotfilename, output):
    fig, axs = plt.subplots(3,1)
    df = filemeta['df']
    title = filemeta['title']
    devices = filemeta['devices']
    deffilename = filemeta['filename'] + f'_{plotfilename}'

    mergecolumns = ['tx_m','ty_m','tz_m'
                    # ,'v_ms','vx_ms','vy_ms','vz_ms'
                    # ,'qx','qy','qz','qw'
                    # ,'wqx_degs','wqy_degs','wqz_degs'
                    ]
    # df_hmd = df.reset_index(level=1)[mergecolumns]
    df_combined_ = pd.DataFrame()
    # df_combined_ = []
    devicesbytype = [0,0,0]     #counter for HMD, Controllers, Tundra
    otherdev = []
    firstdf = True
    hmd_index = None
    for label, dev in devices.items():
        if dev['devicetype'] >= 3:
            continue
        devicetype = dev['devicetype']
        df_ = dev['df'].reset_index(level=1).tail(-1)[mergecolumns]
        tag = f'{device_shorttypes[devicetype]}{devicesbytype[devicetype]}'
        df_ = df_.add_prefix(tag+'_')
        if firstdf:
            firstdf = False
            df_combined_ = df_
        else:
            df_combined_ = df_.join(df_combined_,how='outer')
        if devicetype == 0:
            hmd_index = df_.index
        else:
            otherdev.append(tag)
        # df_combined_.insert(0,df_)
        devicesbytype[devicetype] += 1
        # print(df)

    df_combined = df_combined_.sort_index().interpolate('nearest')      # interpolate NaN values with nearest values
    if (devicesbytype[0] > 0):
        df_combined = df_combined[df_combined.index.isin(hmd_index)]        # only keep original HMD timestamps+ original HMD values + interpolated device values
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')].dropna()  # drop duplicates and NaN values that weren't interpolated
    # print(df_combined)


    fig.suptitle(title)
    
    for i in range(3):
        ax = axs[i]
        if i==0:
            ax.set_title('Device height over time')
            ax.set(ylabel='ty_m (cm)')
            ax.set_ylim([0,225])
            if (len(otherdev) > 0):
                ax.plot(df_combined.index.get_level_values('timestamp_s'), 
                        df_combined[[x + '_ty_m' for x in otherdev]]*100, 
                        label=otherdev, 
                        linewidth=0.5, alpha=0.5)
            if (devicesbytype[0] > 0):
                ax.plot(df_combined.index.get_level_values('timestamp_s'), 
                        df_combined[['HMD0_ty_m']]*100, 
                        label='HMD0', 
                        linewidth=0.5, alpha=0.5)
        elif i==1:
            ax.set_title('Device height relative to HMD')
            ax.set(ylabel=r'$\Delta$ty_m (cm)')
            ax.set_ylim([-125,75])
            if (devicesbytype[0] > 0 and len(otherdev) > 0):
                for d in otherdev:
                    ax.plot(df_combined.index.get_level_values('timestamp_s'), 
                        (df_combined[d + '_ty_m'] - df_combined['HMD0_ty_m'])*100, 
                        label=d, 
                        linewidth=0.5, alpha=0.5)
        elif i==2:
            ax.set_title('Device distance to HMD')
            ax.set(ylabel=r'$\Delta$txyz_m (cm)')
            ax.set_ylim([0,111])
            if (devicesbytype[0] > 0 and len(otherdev) > 0):
                for d in otherdev:
                    ax.plot(df_combined.index.get_level_values('timestamp_s'), 
                        np.sqrt( (df_combined[d + '_tx_m'] - df_combined['HMD0_tx_m'])**2 + (df_combined[d + '_ty_m'] - df_combined['HMD0_ty_m'])**2 + (df_combined[d + '_tz_m'] - df_combined['HMD0_tz_m'])**2 )*100, 
                        label=d, 
                        linewidth=0.5, alpha=0.5)

        if (len(df_combined) > 0):
            ax.set_xlim([0,df_combined.index.get_level_values('timestamp_s').max()])
        ax.set(xlabel='Time (s)')
        ax.xaxis.set_major_locator(plt.MultipleLocator(60))
        ax.legend(fontsize="xx-small", loc="upper right")
        ax.grid()
        ax.set_axisbelow(True)

    return savefig(fig=fig, suptitle=title, plotfilename=deffilename, outputfig=output, type='single')



def xz_heatmap(df, fig, ax, title, devices):
    ax.set_title('Floor HMD heatmap ' + title)
    ax.set(ylabel='X [cm]')
    ax.set(ylabel='X [cm]')
    ax.set_box_aspect(1)
    ax.hist2d( df['tx_m']*100, df['tz_m']*100,bins=[np.arange(-200,200,5),np.arange(-200,200,5)], cmap = 'BuPu', norm = colors.LogNorm())

def posbox(df, fig, ax, title, devices):
    ax.set_title('Boxplot ' + title)
    # ax.set(ylabel='cm cm/s')
    ax.set_ylim([0,275]) 
    # print(df[['ty_m','v_ms']].tail(-1)*100)
    ax.boxplot(df[['ty_m','v_ms']].tail(-1)*100)
    ax.set_xticks([1,2], ['Height [cm]','Velocity[cm/s]'])

def velccdf(df, fig, axs, title, devices):
    ax = axs[0]
    ax.set_title('Velocity ' + title)
    ax.set(xlabel='V [cm/s]')
    ax.set(ylabel='CCDF')

    ax.set_xscale('log')
    ax.set_xlim([0.1,2500]) 

    ax.set_yscale('log')
    ax.set_ylim([10**-6,10**0])

    # for label, dev in sorted(list(devices.items()), key=lambda x:x[0].lower(), reverse=True):
    for label, dev in devices.items():
        if dev['devicetype'] >= 3:
            continue

        data = dev['df']['v_ms']*100
        x = np.sort(data)
        y = np.arange(len(data)) / len(data)
        ax.plot(x,1-y,label = label, marker='+', markersize=2)
        axs[1].plot(data.index.get_level_values('timestamp_s'), data, label=label,linewidth=0.5, alpha=0.5)
    ax.legend()

    axs[1].set(ylabel='V [cm/s]')
    axs[1].set(xlabel='Time [s]')
    axs[1].xaxis.set_minor_locator(plt.MultipleLocator(60))
    axs[1].legend()

def velccdfthesis(df, fig, axs, title, devices):
    ax = axs
    ax.set_title('Velocity ' + title)
    ax.set(xlabel=r'$V_{ms}$ (cm/s)')
    ax.set(ylabel=r'CCDF ( $v_{ms}$ )')

    ax.set_xscale('log')
    ax.set_xlim([0.1,2500]) 

    ax.set_yscale('log')
    ax.set_ylim([10**-6,10**0])

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    i = 0
    # for label, dev in sorted(list(devices.items()), key=lambda x:x[0].lower(), reverse=True):
    for label, dev in devices.items():
        if dev['devicetype'] >= 3:
            continue

        data = dev['df']['v_ms']*100
        n,bins,patches = ax.hist(data, bins=4000, density=True, histtype='step', cumulative=-1, color = colors[i])
        # x = np.sort(data)
        # y = np.arange(len(data)) / len(data)
        # ax.scatter(x,1-y,label = label, marker='+', s=5, color = colors[i])
        ax.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n,label = label, marker='_', s=5, color = colors[i])
        i += 1
        # axs[1].plot(data.index.get_level_values('timestamp_s'), data, label=label,linewidth=0.5, alpha=0.5)
    ax.legend()
    plt.rcParams.update({'font.size': 12})
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
    ax.set_axisbelow(True)

    # axs[1].set(ylabel='V [cm/s]')
    # axs[1].set(xlabel='Time [s]')
    # axs[1].xaxis.set_minor_locator(plt.MultipleLocator(60))
    # axs[1].legend()

def delayboxplot(df, fig, axs, title, devices):
    ax = axs[0]
    ax.set_title('Pose update delay ' + title)
    ax.set(ylabel='Sampling delay (ms)')
    ax.set_ylim([0,16])

    xtickslabels = []
    for label, dev in devices.items():
        if dev['devicetype'] >= 3:
            continue
        df = dev['df']
        xtickslabels.append(label)
        ax.boxplot(df[['time_step']].tail(-1)*1000, positions=[len(xtickslabels)], notch = True, showmeans = True, meanline = True)        

        axs[1].plot(df.index.get_level_values('timestamp_s'), df['time_step']*1000, label=label,linewidth=0.5, alpha=0.9, color=dcolors[label])
    ax.set_xticks(np.arange(1, len(xtickslabels)+1), xtickslabels)
    ax.grid(which='minor', color='#dddddd', linestyle = '--', lw=0.25)
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))

    ax = axs[1]
    ax.set_title('Pose update delay ' + title)
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(60))
    ax.xaxis.set_major_locator(plt.MultipleLocator(60))
    ax.set(xlabel='Time (s)')
    ax.set(ylabel='Sampling delay (ms)')
    # ax.set_ylim([0,10])
    ax.grid(which='minor', color='#dddddd', linestyle = '--', lw=0.25)
    ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
    ax.set_axisbelow(True)
    ax.legend()



axsmap_col = {
    'qz': 'Yaw (qz)',
    'qy': 'Pitch (qy)',
    'qx': 'Roll (qx)',
    'qw': 'qw',
    'wqz_degs': 'Yaw (wqz)',
    'wqy_degs': 'Pitch (wqy)',
    'wqx_degs': 'Roll (wqx)',
}
qcolumns = ['qw','qx','qy','qz']
qwcolumns = ['wqx_degs','wqy_degs','wqz_degs']
qwlabels = list(map(lambda x: axsmap_col[x],qwcolumns))
def quatplot (df, fig, ax, title, devices):
    ax.set_title('HMD Quaternions ' + title)
    ax.set(xlabel='Timestamp [seconds]')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(60))
    ax.set_ylim([-1,1]) 

    for col in qcolumns:
        ax.plot(df.index.get_level_values('timestamp_s'), df[col], label = axsmap_col[col], linewidth=0.5, alpha=0.75)
    ax.legend()

def quatvel (df, fig, ax, title, devices):
    ax.set_title('HMD Quat ang vel ' + title)
    # ax.set_ylim([-600,600]) 
    ax.set(xlabel='Timestamp [seconds]')
    ax.xaxis.set_minor_locator(plt.MultipleLocator(60))
    ax.set(ylabel='W [degr/s]')
    ax.set_ylim([-400,400]) 
    for col in qwcolumns:
        ax.plot(df.index.get_level_values('timestamp_s'), df[col], label = axsmap_col[col], linewidth=0.5, alpha=0.75)
    ax.legend()

def quatvelhist (df, fig, ax, title, devices):
    ax.set_title('HMD Quat Vel hist ' + title)
    ax.set(xlabel='W [degr/s]')

    col = ['wqx_degs','wqy_degs','wqz_degs']
    
    # #density=true
    # ax.set_ylim([10e-8,10e-2]) 
    # ax.hist(df[col].abs(), label=col, density=True, log=True, bins=range(00,450,50))
    # ax.yaxis.set_major_formatter(PercentFormatter(xmax=0.1,decimals=3))
    
    ax.hist(df[col].abs(), label=qwlabels, log=True, bins=range(00,450,50))
    ax.set_ylim([.1,10e5]) 
    ax.legend()

def quatvelbox (df, fig, ax, title, devices):
    ax.set_title('HMD Quat Vel box ' + title)
    ax.set(ylabel='W [degr/s]')
    # ax.set_ylim([0,400]) 
    col = ['wqx_degs','wqy_degs','wqz_degs']
    ax.boxplot(df[col].tail(-1).abs())
    ax.set_xticks([1,2,3], labels=qwlabels)

def angvelccdf(df, fig, axs, title, devices):
    fig.suptitle('Ang Vel CCDF/plot ' + title)
    # for label, dev in devices.items():
    figrow = 0
    for label, dev in sorted(list(devices.items()), key=lambda x:x[0].lower(), reverse=True):
        if dev['devicetype'] >= 3:
            continue
        df = dev['df']
        ax = axs[figrow][0]
        ax2 = axs[figrow][1]
        for col in qwcolumns:
            data = df[col].abs()
            x = np.sort(data)
            y = np.arange(len(data)) / len(data)
            ax.plot(x,1-y,label = f'{axsmap_col[col]}'.split(' ')[0], marker='+', markersize=2)
            ax2.plot(df.index.get_level_values('timestamp_s'), df[col].abs(), label=f'{axsmap_col[col]}'.split(' ')[0],linewidth=0.5, alpha=0.5)
        ax.set_title(label)
        ax.set(xlabel='W [degr/s]')
        ax.set(ylabel='CCDF')
        if dev['devicetype'] == 1:
            ax.set_xlim([0,2000]) 
        else:
            ax.set_xlim([0,1000]) 
        ax.set_yscale('log')
        ax.set_ylim([10**-6,10**0])
        ax.legend(fontsize='xx-small')
        ax2.set_title(label)
        ax2.set(xlabel='Time [s]')
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(60))
        ax2.set(ylabel='W [degr/s]')
        ax2.legend(fontsize='xx-small')
        figrow += 1



# ### Plots comparing all data per app

# # Floor location per application
# def gameslocxz(combinedgamedata, ax, title, devices):
#     ax.set_title('User mobility per application')
#     # ax.set_xlim([-120,120]) 
#     # ax.set_ylim([-100,100]) 
#     ax.set(xlabel='X (cm)')
#     ax.set(ylabel='Z (cm)')
#     alpha = 1
#     linewidth = 0.8
#     for d in combinedgamedata.keys():
#         df = combinedgamedata[d]
#         ax.plot( df[f'tx_m']*100, df[f'tz_m']*100, alpha=alpha, label=d.replace('Alyx', 'Half Life: Alyx'), linewidth=linewidth)
#         linewidth -= 0.05
#         alpha -= 0.2
#     #get handles and labels
#     handles, labels = plt.gca().get_legend_handles_labels()
#     #specify order of items in legend
#     order = [1,0,2]
#     #add legend to plot
#     leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc= 'upper left') 
#     # change the line width in the legend
#     for line in leg.get_lines():
#         line.set_linewidth(3.0)
#     ax.grid()
#     ax.set_axisbelow(True)

# # Histogram: Absolute combined (Pitch+Yaw+Roll) angular velocity density per application
# def gamesangvel(combinedgamedata, ax, title, devices):
#     ax.set_title('Absolute combined (Pitch+Yaw+Roll) Angular velocity density per app ')
#     ax.set(xlabel=r'|ω| [$^{\circ}$/s]')
#     ax.set(ylabel='Percentage')
#     bins = range(0,440,40)
#     absangvel = list(map(lambda g: abs(combinedgamedata[g]), combinedgamedata.keys()))
#     angvelconcat = list(map(lambda df: pd.concat([df['qwy'],df['qwx'],df['qwz']]), absangvel))
#     ax.hist(angvelconcat, label=list(combinedgamedata.keys()), density=True, log=True, bins=bins)
#     ax.set_xticks(bins)
#     # ax.hist([gamesangvel['Alyx'], gamesangvel['Wrench'], gamesangvel['Pistol Whip']], label=games, log=True, bins=range(00,450,50))
#     # ax.set_ylim([.1,10e5]) 
#     ax.legend(loc= 'upper right')
#     ax.grid()
#     ax.set_axisbelow(True)

# # gamesangvel1cdf
# def gamesangvel1cdf(combinedgamedata, ax, title, devices):
#     ax.set_title('Angular velocity per application, per axis')
#     ax.set(xlabel=r'$|ω_{axis}|$ [$^{\circ}$/s]')
#     ax.set(ylabel='1-CDF')
#     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#     colors.reverse()
#     # HMD angular velocity: qwz = Yaw, qwy = Pitch, qwx = Roll
#     axsmap_col = {
#         'qwz': 'Yaw',
#         'qwy': 'Pitch',
#         'qwx': 'Roll',
#     }
#     ax.set_yscale('log')
#     l = []
#     for g in combinedgamedata.keys():
#         # logging.info(g)
#         df = abs(combinedgamedata[g])
#         # logging.info(df.describe())        
#         color = colors.pop()
#         linesstyles = ['dashdot','dotted','dashed','solid']
#         line = False
#         for col in axsmap_col.keys():
#             # ax.hist(df[col], n_bins, density=True, log=True, histtype='step', cumulative=-1, label=f'{g} {axsmap_col[col]}', linestyle=linesstyles.pop(), color=color)
#             data = df[col]
#             x = np.sort(data)
#             y = np.arange(len(data)) / len(data)
#             line = plt.plot(x,1-y,label=(g + r' $_{' + axsmap_col[col] + r'}$').replace('Alyx', 'Half Life: Alyx'), linestyle=linesstyles.pop(), color=color)
#             l.append(line)

#     # get handles and labels
#     handles, labels = plt.gca().get_legend_handles_labels()
#     #specify order of items in legend
#     order = [6,0,3]
#     #add legend to plot
#     legend1 = ax.legend([handles[idx] for idx in order],['Pistol Whip', 'Half Life: ALyx', 'Wrench'], loc= 'upper right') 
#     ax.add_artist(legend1)
#     order2 = [6,7,8]
#     legend2 = ax.legend([handles[idx] for idx in order2],['Yaw', 'Pitch', 'Roll'], loc= 'lower left') 
#     for h in legend2.legend_handles:
#         h.set_color('grey')
#     # legend1 = ax.legend([0,1,2], loc=1)
#     # ax.add_artist(legend1)
#     # ax.legend([l[0],l[3],l[6]], loc=4)

#     ax.grid()
#     ax.set_axisbelow(True)







# C:\Users\dks>mklink /j "C:\XRmobility\analysis\Data" "C:\Users\dks\OneDrive - KU Leuven\VR Experiment\Data"
# Junction created for C:\XRmobility\analysis\Data <<===>> C:\Users\dks\OneDrive - KU Leuven\VR Experiment\Data

# C:\Users\dks>mklink /j "C:\XRmobility\analysis\Data-output" "C:\Users\dks\OneDrive - KU Leuven\VR Experiment\Data-output"
# Junction created for C:\XRmobility\analysis\Data-output <<===>> C:\Users\dks\OneDrive - KU Leuven\VR Experiment\Data-output

# C:\Users\dks>mklink /j "C:\XRmobility\analysis\Data-figures" "C:\Users\dks\OneDrive - KU Leuven\VR Experiment\Data-figures"
# Junction created for C:\XRmobility\analysis\Data-figures <<===>> C:\Users\dks\OneDrive - KU Leuven\VR Experiment\Data-figures


# ax template
    # ax.legend(fontsize='small')
    # ax.set_title('Timestep delay HMD deviation')
    # ax.set(xlabel='Mean (ms)')
    # ax.set(ylabel='Standard deviation (ms)')
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xticks([1,2,3,4,5,6,7,8,9,10,20,30,40])
    # ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    # ax.xaxis.set_minor_locator(plt.MultipleLocator(60))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(60))
    # ax.grid(which='minor', color='#dddddd', linestyle = '-.', lw=0.25, axis='y')
    # ax.grid(which='major', color='#bbbbbb', linestyle = '--', lw=0.5)
    # ax.set_axisbelow(True)
    # , color=dcolors[device_shorttypes[type]] , color=dcolors[label]
    # plt.rcParams['font.size'] = 13
    # plt.rcParams.update({'font.size': 11})
    # fig.set_figwidth(fig.get_figwidth()*2*4/6)
    # fig.set_size_inches(fig.get_figwidth()*2*2/6, 4.8)

main()