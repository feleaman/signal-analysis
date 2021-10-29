import os

mypath = 'M:\\Betriebsmessungen\\VO_Reibung_AE_Sandbeladung\\Stationare_Phasen'


#features
myname = 'Features_AE1_hp5khz_MAX'
os.system('python P_AE_Sand_Water.py --mode features_per_file --channel AE1 --fs 1.e6 --save ON --mypath ' + mypath + ' --name ' + myname)


myname = 'Features_AE2_hp5khz_MAX'
os.system('python P_AE_Sand_Water.py --mode features_per_file --channel AE2 --fs 1.e6 --save ON --mypath ' + mypath + ' --name ' + myname)



# #burst 1
# myname = 'Burst_AE1_hp5khz_thr12rms'
# os.system('python P_AE_Sand_Water.py --mode bursts_per_file --channel AE1 --fs 1.e6 --save ON --mypath ' + mypath + ' --name ' + myname + ' --thr_mode factor_rms --thr_value 12.')

# myname = 'Burst_AE2_hp5khz_thr12rms'
# os.system('python P_AE_Sand_Water.py --mode bursts_per_file --channel AE2 --fs 1.e6 --save ON --mypath ' + mypath + ' --name ' + myname + ' --thr_mode factor_rms --thr_value 12.')


# #burst 2
# myname = 'Burst_AE1_hp5khz_thr0p2fix'
# os.system('python P_AE_Sand_Water.py --mode bursts_per_file --channel AE1 --fs 1.e6 --save ON --mypath ' + mypath + ' --name ' + myname + ' --thr_mode fixed_value --thr_value 0.2')

# myname = 'Burst_AE2_hp5khz_thr0p2fix'
# os.system('python P_AE_Sand_Water.py --mode bursts_per_file --channel AE2 --fs 1.e6 --save ON --mypath ' + mypath + ' --name ' + myname + ' --thr_mode fixed_value --thr_value 0.2')