#+++import modules and functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk, Button, filedialog
from os.path import join, abspath, basename
from os import getcwd
import sys
import pywt
from argparse import ArgumentParser

#+++import user-defined functions
sys.path.insert(0, './lib') 
from p_open_extension import *
from p_bursts import *
from p_fft import *
from p_denois import *
from W_Wavelet import *

#+++options
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'Times New Roman'
fslabel = 16
fstitle = 16
fstick = 14

#+++inputs
Inputs = ['mode', 'channel', 'fs']
InputsOpt_Defaults = {'mypath':None, 'range':None, 'save':'OFF', 'window':'boxcar', 'filter':['highpass', 5.e3, 3], 'thr_value':12.0, 'thr_mode':'factor_rms', 'widths_wavelet':[1, 31], 'mother_wavelet':'morl', 'level_contour':20, 'cmap':'viridis', 'window_time':0.001, 'stella':300, 'lockout':300, 'preview':'OFF', 'name':'auto', 'level_wavelet':3, 'mother_wavelet':'db6'}

#YlGnBu
def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	mydir = abspath(getcwd())
	

	if config['mode'] == 'bursts_per_file':
		#+++Load data
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
		Filenames = [basename(filepath) for filepath in Filepaths]
		
		#+++Calculations
		mydict = {}
		mydict['bursts'] = []
		rownames = []
		count = 0
		myrows = {}
		for filepath, filename in zip(Filepaths, Filenames):
			signal = load_signal(filepath, config['channel'], extension='tdm')
			time = [i/config['fs'] for i in range(len(signal))]

			#+++filter
			if config['filter'][0] != 'OFF':
				print('with filter! ', config['filter'])
				signal = butter_filter(signal, config['fs'], config['filter'])
	
			threshold = read_threshold(config['thr_mode'], config['thr_value'], signal)
			
			t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1 = thr_burst_detector_stella_lockout(signal, config, count=0, threshold=threshold)

			if config['preview'] == 'ON':
				# from THR_Burst_Detection import plot_burst_rev
				fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
				plot_burst_rev(fig_0, ax_0, 0, time, signal, config, t_burst_corr1, amp_burst_corr1, t_burst_corr_rev1, amp_burst_corr_rev1)
				ax_0.set_title(filename)
				ax_0.set_ylabel('Amplitude [mV]', fontsize=13)
				ax_0.set_xlabel('Time [s]', fontsize=13)
				ax_0.tick_params(axis='both', labelsize=12)
				plt.show()
			
			num = len(t_burst_corr1)

			mydict['bursts'].append(num)
			rownames.append(filename)
		
		
		#+++output
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Bursts')
			writer.close()
	
	
	elif config['mode'] == 'freq_band_wpd':
		#+++Load data
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
		Filenames = [basename(filepath) for filepath in Filepaths]
		
		#+++Calculations
		mydict = {}
		rownames = []
		count = 0
		for filepath, filename in zip(Filepaths, Filenames):
			wfm = load_signal(filepath, config['channel'], extension='tdm')
			time = [i/config['fs'] for i in range(len(wfm))]
			
			#+++filter
			if config['filter'][0] != 'OFF':
				print('with filter! ', config['filter'])
				wfm = butter_filter(wfm, config['fs'], config['filter'])
			
			
			#+++config wpd spectrum	
			wpdmatr, wpdmatr_sqr, freq = wav_packet_deco(wfm, config)
			print('Frequencies = ', freq)
			
			Energy = np.zeros(2**config['level_wavelet'])
			sum = np.zeros(len(wpdmatr_sqr))
			for i in range(len(wpdmatr_sqr)):
				sum[i] = np.sum((wpdmatr_sqr[i]))
				Energy += sum
			
			for k in range(len(freq)):
				if str(freq[k]) in mydict:
					mydict[str(freq[k])].append(Energy[k])
				else:
					print('first time!')
					mydict[str(freq[k])] = []
					mydict[str(freq[k])].append(Energy[k])
				
			
			rownames.append(filename)
		
		
		#+++output
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='WPD')
			writer.close()
	
	elif config['mode'] == 'features_per_file':
		#+++Load data
		if config['mypath'] == None:
			root = Tk()
			root.withdraw()
			root.update()
			Filepaths = filedialog.askopenfilenames()			
			root.destroy()
		else:		
			Filepaths = [join(config['mypath'], f) for f in listdir(config['mypath']) if isfile(join(config['mypath'], f)) if f[-4:] == 'tdms']
		Filenames = [basename(filepath) for filepath in Filepaths]
		
		#+++Calculations
		mydict = {}
		rownames = []
		# RMS = []
		# KURT = []
		MAX = []
		count = 0
		for filepath, filename in zip(Filepaths, Filenames):
			wfm = load_signal(filepath, config['channel'], extension='tdm')
			time = [i/config['fs'] for i in range(len(wfm))]
			
			#+++filter
			if config['filter'][0] != 'OFF':
				print('with filter! ', config['filter'])
				wfm = butter_filter(wfm, config['fs'], config['filter'])
			
			# RMS.append(np.sqrt(np.sum(wfm**2.0)/len(wfm)))
			# KURT.append(stats.kurtosis(wfm, fisher=True))
			MAX.append(np.max(np.abs(wfm)))
				
			
			rownames.append(filename)
		
		# mydict['KURT'] = KURT
		# mydict['RMS'] = RMS
		mydict['MAX'] = MAX
		#+++output
		if config['save'] == 'ON':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='WPD')
			writer.close()	
	
	
		
	else:
		print('unknown mode')

	
	return


def read_parser(argv, Inputs, InputsOpt_Defaults):
	try:
		Inputs_opt = [key for key in InputsOpt_Defaults]
		Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		parser = ArgumentParser()
		for element in (Inputs + Inputs_opt):
			print(element)
			if element == 'files' or element == 'range' or element == 'filter' or element == 'widths_wavelet':
				parser.add_argument('--' + element, nargs='+')
			else:
				parser.add_argument('--' + element, nargs='?')
		print(parser.parse_args())
		args = parser.parse_args()
		
	except:
		# args = argv
		arguments = [element for element in argv if element[0:2] == '--']
		values = [element for element in argv if element[0:2] != '--']

		# from argparse import ArgumentParser
		# from ArgumentParser import Namespace
		parser = ArgumentParser()
		for element in arguments:
			parser.add_argument(element)

		args = parser.parse_args(argv)

		# print(test)
		# sys.exit()
		
	config = {}	
		
	for element in Inputs:
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Required:', element)

	for element, value in zip(Inputs_opt, Defaults):
		if getattr(args, element) != None:
			config[element] = getattr(args, element)
		else:
			print('Default ' + element + ' = ', value)
			config[element] = value
	

		
	config['fs'] = float(config['fs'])
	config['thr_value'] = float(config['thr_value'])
	config['window_time'] = float(config['window_time'])
	
	config['stella'] = int(config['stella'])
	config['lockout'] = int(config['lockout'])
	config['level_wavelet'] = int(config['level_wavelet'])
	
	
	if config['filter'][0] != 'OFF':
		if config['filter'][0] == 'bandpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2]), float(config['filter'][3])]
		elif config['filter'][0] == 'highpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		elif config['filter'][0] == 'lowpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		else:
			print('error filter 87965')
			sys.exit()
	
		
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config


if __name__ == '__main__':
	main(sys.argv)

   
#plt.rcParams['agg.path.chunksize'] = 1000
#plt.rcParams['savefig.directory'] = os.chdir(os.path.dirname('D:'))
# plt.rcParams['savefig.dpi'] = 1500
# plt.rcParams['savefig.format'] = 'jpeg'
