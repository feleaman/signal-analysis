#+++import modules and functions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk, Button, filedialog
from os.path import join, abspath
from os import getcwd
import sys
import pywt
from argparse import ArgumentParser

#+++import user-defined functions
sys.path.insert(0, './lib') 
from p_open_extension import *
from p_fft import *
from p_denois import *


#+++inputs
Inputs = ['mode', 'channel', 'fs']
InputsOpt_Defaults = {'mypath':None, 'range':None, 'output':'plot', 'window':'hanning', 'filter':['highpass', 5.e3, 3], 'mode_stft':('window', 0.001), '3d_plot':'mesh'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	mydir = abspath(getcwd())
	
	if config['mode'] == 'wfm':
		#+++read data
		if config['mypath'] == None:
			print('select signal file: ')
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()
		else:
			filepath = join(mydir, config['mypath'])		
		
		wfm = load_signal(filepath, config['channel'])
		#wfm = np.abs(wfm)
		if config['range'] != None:
			wfm = wfm[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		
		time = np.arange(len(wfm))/config['fs']
		
		if config['filter'][0] != 'OFF':
			print(config['filter'])
			wfm = butter_filter(wfm, config['fs'], config['filter'])
		
		
		
		plt.plot(time, wfm)
		plt.show()
	
	elif config['mode'] == 'fft':
		#+++read data
		if config['mypath'] == None:
			print('select signal file: ')
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()
		else:
			filepath = join(mydir, config['mypath'])		
		
		wfm = load_signal(filepath, config['channel'])
		#wfm = np.abs(wfm)
		if config['range'] != None:
			wfm = wfm[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		
		time = np.arange(len(wfm))/config['fs']
		
		if config['filter'][0] != 'OFF':
			print(config['filter'])
			wfm = butter_filter(wfm, config['fs'], config['filter'])
		
		
		
		plt.plot(time, wfm)
		plt.show()
		
		if config['window'] == 'boxcar':
			magX, freq, df = mag_fft(wfm, config['fs'])
		elif config['window'] == 'hanning':
			magX, freq, df = mag_fft_hanning(wfm, config['fs'])
		else:
			print('error window 88416')
			sys.exit()

		#+++plot
		fig, ax = plt.subplots()
		ax.plot(freq, magX)
		ax.set_xlabel('Frequency [Hz]')
		ax.set_ylabel('Magnitude')
		ax.set_title('FFT')
		
		if config['output'] == 'plot':
			plt.show()
		elif config['output'] == 'save':
			import datetime
			stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			plt.savefig(stamp+'.png')
			save_pickle(stamp+'.pkl', config)
	
	elif config['mode'] == 'stft':
		#+++read data
		if config['mypath'] == None:
			print('select signal file: ')
			root = Tk()
			root.withdraw()
			root.update()
			filepath = filedialog.askopenfilename()
			root.destroy()
		else:
			filepath = join(mydir, config['mypath'])		
		
		wfm = load_signal(filepath, config['channel'])
		
		if config['range'] != None:
			wfm = wfm[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		
		time = np.arange(len(wfm))/config['fs']
		
		# plt.plot(time, wfm)
		# plt.show()
		
		if config['filter'][0] != 'OFF':
			print(config['filter'])
			wfm = butter_filter(wfm, config['fs'], config['filter'])
	
		stftX, f_stft, df_stft, t_stft = shortFFT(wfm, config['fs'], config['mode_stft'], config['window'])
		
		print('dt ****', t_stft[1]-t_stft[0])
		print('df ****', df_stft)
		
		# #+++plot
		if config['3d_plot'] == 'mesh':
			fig, ax = plt.subplots()
			fig.set_size_inches(9,6)
			mesh = ax.pcolormesh(t_stft, f_stft/1000., stftX, shading='auto')
			ax.set_xlabel('Time [s]')
			ax.set_ylabel('Frequency [kHz]')
			ax.set_title('STFT')
			fig.colorbar(mesh)
		elif config['3d_plot'] == 'surface':
			from matplotlib import cm
			fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
			tgato, fgato = np.meshgrid(t_stft, f_stft/1000.)
			surf = ax.plot_surface(tgato, fgato, stftX, cmap=cm.coolwarm)
			ax.set_xlabel('Time [s]')
			ax.set_ylabel('Frequency [kHz]')
			fig.colorbar(surf, shrink=0.5, aspect=5)
		
		
		if config['output'] == 'plot':
			plt.show()
		elif config['output'] == 'save':
			import datetime
			stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			plt.savefig(stamp+'.png')
			save_pickle(stamp+'.pkl', config)
	
	
		
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
			if element == 'files' or element == 'range' or element == 'filter':
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
	# config['segments'] = int(config['segments'])
	
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

def shortFFT(x, fs, time_mode, window):
	if time_mode[0] == 'segments':
		nperseg = int(len(x)/segments)
	elif time_mode[0] == 'window':
		length = time_mode[1]*fs
		#nperseg = int(len(x)/length)
		nperseg = int(length)
	f, t, stftX = signal.spectrogram(x, fs, nperseg=nperseg, window=window, mode='magnitude')
	stftX = stftX/nperseg
	df = f[2] - f[1]
	return stftX, f, df, t

if __name__ == '__main__':
	main(sys.argv)

#plt.rcParams['agg.path.chunksize'] = 1000
#plt.rcParams['savefig.directory'] = os.chdir(os.path.dirname('D:'))
# plt.rcParams['savefig.dpi'] = 1500
# plt.rcParams['savefig.format'] = 'jpeg'
