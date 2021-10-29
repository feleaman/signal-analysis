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
from scipy import signal

#+++import user-defined functions
sys.path.insert(0, './lib') 
from p_open_extension import *
from p_fft import *
from p_denois import *

from Kurtogram3 import Fast_Kurtogram_filters


#+++options
fslabel = 16
fstitle = 16
fstick = 14



#+++inputs
Inputs = ['mode', 'channel', 'fs']
InputsOpt_Defaults = {'mypath':None, 'range':None, 'output':'plot', 'window':'boxcar', 'filter':['OFF', 5.e3, 3], 'demodulation':'hilbert', 'name':'auto'}

#YlGnBu
def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	mydir = abspath(getcwd())
	
	if config['mode'] == 'fir_filter':
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
		
		
		#+++load signal
		wfm = load_signal(filepath, config['channel'], extension='tdm')
		if config['range'] != None:
			wfm = wfm[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		time = np.arange(len(wfm))/config['fs']


		#+++filter
		# if config['filter'][0] != 'OFF':
			# print(config['filter'])
			# wfm = butter_filter(wfm, config['fs'], config['filter'])
		
		# numtaps = 3
		# f0 = 90.e3/1.e6
		# coeff = signal.firwin(numtaps, f0, pass_zero=False)
		# w, h = signal.freqz(coeff)
		# plt.plot(w, h)
		# plt.show()
		
		
		desired = (0, 0, 1, 1, 0, 0)
		bands = (0, 1, 2, 4, 4.5, 5)
		coeff = signal.firls(73, bands, desired, fs=config['fs'])
		print(coeff)
		w, h = signal.freqz(coeff)
		plt.plot(0.5*config['fs']*w/np.pi, np.abs(h))
		plt.show()
		
		
		
		
		sys.exit()
		# w, h = signal.freqz(b)
		

		#+++plot
		fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.15, right=0.9, bottom=0.175, top=0.9)
		fig.set_size_inches(6, 3.5)
		ax.plot(time, wfm)
		# ax.set_xlim(left=0, right=0.315*1000)
		# ax.set_ylim(bottom=-6.5, top=6.5)
		# ax.set_ylim(bottom=-2.5, top=2.5)
		ax.set_xlabel('Time [s]', fontsize=fslabel)
		ax.set_ylabel('Amplitude [V]', fontsize=fslabel)
		# ax.set_title('Waveform ' + materials, fontsize=fstitle)
		ax.tick_params(axis='both', labelsize=fstick)
		ax.grid(axis='both')
		
		
		#+++output
		if config['output'] == 'plot':
			plt.show()
		elif config['output'] == 'save':
			# import datetime
			# stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			# plt.savefig(stamp+'.png')
			plt.savefig('WFM_' + filename[:-5]+ '.svg')
			
	
	elif config['mode'] == 'obtain_freq_response':
		print('select imf...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		mydict = read_pickle(filepath)
		magX_imf = mydict['fft']
		f_imf = mydict['f']
		
		
		print('select raw signal...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		mydict = read_pickle(filepath)
		magX_raw = mydict['fft']
		f_raw = mydict['f']
		
		
		freq_resp = magX_imf/magX_raw
		
		fig, ax = plt.subplots(ncols=3)
		ax[0].plot(f_imf, magX_imf)
		ax[1].plot(f_raw, magX_raw)
		ax[2].plot(f_raw, freq_resp)
		plt.show()
	
	
	elif config['mode'] == 'freq_response_filter':
		print('select imf...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		mydict = read_pickle(filepath)
		magX_imf = mydict['fft']
		f_imf = mydict['f']
		
		
		print('select raw signal...')
		root = Tk()
		root.withdraw()
		root.update()
		filepath = filedialog.askopenfilename()
		root.destroy()
		
		mydict = read_pickle(filepath)
		magX_raw = mydict['fft']
		f_raw = mydict['f']
		
		
		freq_resp = magX_imf/magX_raw	
		
		
		desired = freq_resp
		bands = f_raw
		coeff = signal.firls(73, bands, desired, fs=config['fs'])
		w, h = signal.freqz(coeff)
		plt.plot(0.5*config['fs']*w/np.pi, np.abs(h))
		plt.show()
	
	
	elif config['mode'] == '3plot_freq_response_filter':
		print('select 3 imfs...')
		root = Tk()
		root.withdraw()
		root.update()
		filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print(filepaths)
		
		mydict = read_pickle(filepaths[0])
		magX_imf_1 = mydict['fft']
		f_imf_1 = mydict['f']
		
		mydict = read_pickle(filepaths[1])
		magX_imf_2 = mydict['fft']
		f_imf_2 = mydict['f']
		
		mydict = read_pickle(filepaths[2])
		magX_imf_3 = mydict['fft']
		f_imf_3 = mydict['f']
		
		
		
		print('select 3 raw signals...')
		root = Tk()
		root.withdraw()
		root.update()
		filepaths = filedialog.askopenfilenames()
		root.destroy()
		
		print(filepaths)
		
		mydict = read_pickle(filepaths[0])
		magX_raw_1 = mydict['fft']
		f_raw_1 = mydict['f']
		
		mydict = read_pickle(filepaths[1])
		magX_raw_2 = mydict['fft']
		f_raw_2 = mydict['f']
		
		mydict = read_pickle(filepaths[2])
		magX_raw_3 = mydict['fft']
		f_raw_3 = mydict['f']
		
		
		
		freq_resp_1 = magX_imf_1/magX_raw_1
		freq_resp_2 = magX_imf_2/magX_raw_2
		freq_resp_3 = magX_imf_3/magX_raw_3
		
		desired_1 = freq_resp_1
		desired_2 = freq_resp_2
		desired_3 = freq_resp_3
		
		bands_1 = f_raw_1
		coeff_1 = signal.firls(73, bands_1, desired_1, fs=config['fs'])		
		bands_2 = f_raw_2
		coeff_2 = signal.firls(73, bands_2, desired_2, fs=config['fs'])		
		bands_3 = f_raw_3
		coeff_3 = signal.firls(73, bands_3, desired_3, fs=config['fs'])
		
		

		
		w1, h1 = signal.freqz(coeff_1)
		w2, h2 = signal.freqz(coeff_2)
		w3, h3 = signal.freqz(coeff_3)
		
		
		# plt.plot(0.5*config['fs']*w/np.pi, np.abs(h))
		# plt.show()
		
		
		
		#+++plot
		myfont = 12.5
		fig, ax = plt.subplots(ncols=3)
		
		
		fig.set_size_inches(12,4)
		plt.subplots_adjust(left=0.05, right=0.985, bottom=0.15, top=0.9, wspace=0.19)
		
		ax[0].plot(0.5*config['fs']*w1/np.pi/1000., np.abs(h1), label='IMF-3')
		ax[0].legend(loc='upper center', fontsize=10, handletextpad=0, handlelength=0, labelspacing=.3)
		ax[0].set_xlim(left=0, right=500)
		ax[0].set_ylim(bottom=0, top=1.2)
		ax[0].set_title('No Fault', fontsize=myfont)
		ax[0].set_xlabel('Frequency [kHz]', fontsize=myfont)
		ax[0].set_ylabel('Response [-]', fontsize=myfont)
		# ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
		
		ax[1].plot(0.5*config['fs']*w2/np.pi/1000., np.abs(h2), label='IMF-3')
		ax[1].legend(loc='upper center', fontsize=10, handletextpad=0, handlelength=0, labelspacing=.3)
		ax[1].set_xlim(left=0, right=500)
		ax[1].set_ylim(bottom=0, top=1.2)
		ax[1].set_title('Initial Faulty Condition', fontsize=myfont)
		ax[1].set_xlabel('Frequency [kHz]', fontsize=myfont)
		ax[1].set_ylabel('Response [-]', fontsize=myfont)
		# ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
		
		ax[2].plot(0.5*config['fs']*w3/np.pi/1000., np.abs(h3), label='IMF-3')
		ax[2].legend(loc='upper center', fontsize=10, handletextpad=0, handlelength=0, labelspacing=.3)
		ax[2].set_xlim(left=0, right=500)
		ax[2].set_ylim(bottom=0, top=1.2)
		ax[2].set_title('Developed Faulty Condition', fontsize=myfont)
		ax[2].set_xlabel('Frequency [kHz]', fontsize=myfont)
		ax[2].set_ylabel('Response [-]', fontsize=myfont)		
		# ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
		# ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		# ax.set_title('Waveform ' + materials, fontsize=fstitle)
		# ax.tick_params(axis='both', labelsize=fstick)
		# plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
		
		

		


		#+++output
		if config['output'] == 'plot':
			plt.show()
		elif config['output'] == 'save':
			# import datetime
			# stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			# plt.savefig(stamp+'.png')
			plt.savefig('Plot_' + config['name'] + '.svg')
		
		
		
		
		

		
	
	elif config['mode'] == '3avg_env_spectrum':
		#+++read data
		if config['mypath'] == None:
			print('select signal files: ')
			root = Tk()
			root.withdraw()
			root.update()
			filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			filepaths = join(mydir, config['mypath'])		
		
		count = 1
		for filepath in filepaths:
			wfm = load_signal(filepath, config['channel'])
			wfm = wfm*1000.

			if config['filter'][0] != 'OFF':
				if config['filter'][0] != 'kurtogram':
					print(config['filter'])
					wfm = butter_filter(wfm, config['fs'], config['filter'])
				elif config['filter'][0] == 'kurtogram':
					print('Kurtogram filter order 4!')
					lp, hp, max_kurt = Fast_Kurtogram_filters(wfm, 4, config['fs'])
					if hp >= config['fs']/2:
						hp = config['fs']/2 - 1.0
					wfm = butter_bandpass(x=wfm, fs=config['fs'], freqs=[lp, hp], order=3, warm_points=None)
			
			if config['demodulation'] == 'hilbert':
				print('hilbert demodulation!')
				wfm = hilbert_demodulation(wfm)
			
			magX, frec, df = mag_fft(wfm, config['fs'])
			
			if count == 1:
				AvgEnv = magX
			else:
				AvgEnv += AvgEnv
			count +=1
		AvgEnv = AvgEnv/count
		
		mydic = {'fft':AvgEnv, 'f':frec}
		save_pickle(config['name'] + '.pkl', mydic)


	elif config['mode'] == 'plot_3avg_env_spectrum':
		#+++read data
		if config['mypath'] == None:
			print('select signal files: ')
			root = Tk()
			root.withdraw()
			root.update()
			filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			filepaths = join(mydir, config['mypath'])		
		print(filepaths)
		
		mydict = read_pickle(filepaths[0])
		mag1 = mydict['fft']
		f1 = mydict['f']
		
		mydict = read_pickle(filepaths[1])
		mag2 = mydict['fft']
		f2 = mydict['f']
		
		mydict = read_pickle(filepaths[2])
		mag3 = mydict['fft']
		f3 = mydict['f']

		#+++plot
		myfont = 12.5
		fig, ax = plt.subplots(ncols=3)
		fig.set_size_inches(12,4)
		plt.subplots_adjust(left=0.05, right=0.985, bottom=0.15, top=0.9, wspace=0.19)
		# fig.set_size_inches(6, 3.5)
		ax[0].plot(f1, mag1)
		ax[0].set_xlim(left=0, right=50)
		ax[0].set_ylim(bottom=0, top=0.0012)
		# ax[0].set_ylim(bottom=0, top=0.6)
		ax[0].set_title('No Fault', fontsize=myfont)
		ax[0].set_xlabel('Frequency [Hz]', fontsize=myfont)
		ax[0].set_ylabel('Magnitude [mV]', fontsize=myfont)
		ax[0].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
		
		ax[1].plot(f2, mag2)
		ax[1].set_xlim(left=0, right=50)
		ax[1].set_ylim(bottom=0, top=0.0012)
		# ax[1].set_ylim(bottom=0, top=0.6)
		ax[1].set_title('Initial Faulty Condition', fontsize=myfont)
		ax[1].set_xlabel('Frequency [Hz]', fontsize=myfont)
		ax[1].set_ylabel('Magnitude [mV]', fontsize=myfont)
		ax[1].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
		
		ax[2].plot(f3, mag3)
		ax[2].set_xlim(left=0, right=50)
		ax[2].set_ylim(bottom=0, top=0.0012)
		# ax[2].set_ylim(bottom=0, top=0.6)
		ax[2].set_title('Developed Faulty Condition', fontsize=myfont)
		ax[2].set_xlabel('Frequency [Hz]', fontsize=myfont)
		ax[2].set_ylabel('Magnitude [mV]', fontsize=myfont)		
		ax[2].ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
		# ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
		# ax.set_title('Waveform ' + materials, fontsize=fstitle)
		# ax.tick_params(axis='both', labelsize=fstick)
		# plt.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))

		
		#+++output
		if config['output'] == 'plot':
			plt.show()
		elif config['output'] == 'save':
			# import datetime
			# stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			# plt.savefig(stamp+'.png')
			plt.savefig('Plot_' + config['name'] + '.svg')
	
	
	elif config['mode'] == 'kurtogram':
		#+++read data
		if config['mypath'] == None:
			print('select signal files: ')
			root = Tk()
			root.withdraw()
			root.update()
			filepaths = filedialog.askopenfilenames()
			root.destroy()
		else:
			filepaths = join(mydir, config['mypath'])		
		
		LP = []
		HP = []
		KURT = []
		rownames = []
		for filepath in filepaths:
			filename = basename(filepath)
			rownames.append(filename)
			
			
			wfm = load_signal(filepath, config['channel'])
			wfm = wfm*1000.
			
			# lp, hp, max_kurt = Fast_Kurtogram_filters(wfm, config['level'], config['fs'])
			lp, hp, max_kurt = Fast_Kurtogram_filters(wfm, 4, config['fs'])
			print(lp, hp, max_kurt)
			LP.append(lp)
			HP.append(hp)
			KURT.append(max_kurt)
		mydict = {'LP':LP, 'HP':HP, 'KURT':KURT}
		
		writer = pd.ExcelWriter('all_kurtosis' + '.xlsx')
		DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
		DataFr_max.to_excel(writer, sheet_name='Bursts')
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

	
	if config['filter'][0] != 'OFF':
		if config['filter'][0] == 'bandpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2]), float(config['filter'][3])]
		elif config['filter'][0] == 'highpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		elif config['filter'][0] == 'lowpass':
			config['filter'] = [config['filter'][0], float(config['filter'][1]), float(config['filter'][2])]
		else:
			print('warning filter 87965')
			# sys.exit()
	
		
	if config['range'] != None:
		config['range'][0] = float(config['range'][0])
		config['range'][1] = float(config['range'][1])
	
	# config['fscore_min'] = float(config['fscore_min'])
	#Type conversion to int	
	# Variable conversion
	return config

#+++functions
def hilbert_demodulation(x, rect=None):
	#Rectification
	n = len(x)
	if rect == 'only_positives':
		x_rect = np.zeros(n)
		for i in range(n):
			if x[i] > 0:
				x_rect[i] = x[i]
	elif rect == 'absolute_value':
		x_rect = np.abs(x)
	else:
		# print('Info: Demodulation without rectification')
		x_rect = x
	
	x_ana = signal.hilbert(x_rect)
	x_demod = np.abs(x_ana)	
	
	return x_demod

if __name__ == '__main__':
	main(sys.argv)

   
#plt.rcParams['agg.path.chunksize'] = 1000
#plt.rcParams['savefig.directory'] = os.chdir(os.path.dirname('D:'))
# plt.rcParams['savefig.dpi'] = 1500
# plt.rcParams['savefig.format'] = 'jpeg'
