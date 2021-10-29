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
from p_denois import *
plt.rcParams['agg.path.chunksize'] = 1000

#+++inputs
Inputs = ['mode', 'channel', 'fs']
InputsOpt_Defaults = {'mypath':None, 'range':None, 'mother_wavelet':'morl', 'widths':[1,21], 'cmap':'Greys', 'level_contour':10, 'output':'plot', 'level_wavelet':10, 'filter':['highpass', 5.e3, 3], 'absolute':'OFF'}

def main(argv):
	config = read_parser(argv, Inputs, InputsOpt_Defaults)
	mydir = abspath(getcwd())
	
	if config['mode'] == 'cwt_scalogram':
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
		
		if config['filter'][0] != 'OFF':
			print(config['filter'])
			wfm = butter_filter(wfm, config['fs'], config['filter'])
		
		# plt.plot(time, wfm)
		# plt.show()
		#+++config cwt
		mother_wv = config['mother_wavelet']
		min_width = config['widths'][0]
		max_width = config['widths'][1]
		widths = np.arange(min_width, max_width)
		
		cwtmatr, freq = pywt.cwt(data=wfm, scales=widths, wavelet=mother_wv, sampling_period=1./config['fs'])
		print('Frequencies = ', freq)
		
		if config['absolute'] == 'ON':
			cwtmatr = np.abs(cwtmatr)
		
		
		#+++plot cwt
		maxxx = np.max(cwtmatr)
		minnn = np.min(cwtmatr)
		levels_plot = list(np.linspace(minnn, maxxx, num=config['level_contour']))
		fig, ax = plt.subplots()
		ax.set_xlabel('Time [s]')
		ax.set_ylabel('Frequency [Hz]')
		ax.set_title('Scalogram CWT')
		
		cwt = ax.contourf(time, freq, cwtmatr, levels=levels_plot, cmap=config['cmap'])
		
		ax.set_ylim(bottom=0 , top=config['fs']/2.)
		cbar = fig.colorbar(cwt)
		cbar.set_label('Magnitude [V]')
		if config['output'] == 'plot':
			plt.show()
		elif config['output'] == 'save':
			import datetime
			stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			plt.savefig(stamp+'.png')
			save_pickle(stamp+'.pkl', config)
	
	
	elif config['mode'] == 'wpd_scalogram':
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
		
		if config['filter'][0] != 'OFF':
			print(config['filter'])
			wfm = butter_filter(wfm, config['fs'], config['filter'])
			
		plt.plot(time, wfm)
		plt.show()
		#+++config cwt
		
		wpdmatr, wpdmatr_sqr, freq = wav_packet_deco(wfm, config)
		print('Frequencies = ', freq)
		
		#+++plot cwt
		maxxx = np.max(wpdmatr)
		minnn = np.min(wpdmatr)
		levels_plot = list(np.linspace(minnn, maxxx, num=config['level_contour']))
		fig, ax = plt.subplots()
		ax.set_xlabel('Time [s]')
		ax.set_ylabel('Frequency [Hz]')
		ax.set_title('Scalogram WPD')
		
		#extent_ = [0, np.max(time), 0, config['fs']/2]
		#cwt = ax.contourf(wpdmatr, extent=extent_, cmap=config['cmap'], levels=levels_plot)
		
		cwt = ax.contourf(time, freq, wpdmatr, levels=levels_plot, cmap=config['cmap'])
		
		ax.set_ylim(bottom=0 , top=config['fs']/2.)
		cbar = fig.colorbar(cwt)
		cbar.set_label('Magnitude [V]')
		if config['output'] == 'plot':
			plt.show()
		elif config['output'] == 'save':
			import datetime
			stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			plt.savefig(stamp+'.png')
			save_pickle(stamp+'.pkl', config)
	
	
	elif config['mode'] == 'wpd_spectrum':
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
		time = np.arange(len(wfm))/config['fs']
		
		#+++config wpd spectrum	
		wpdmatr, wpdmatr_sqr, freq = wav_packet_deco(wfm, config)
		print('Frequencies = ', freq)
		
		Energy = np.zeros(2**config['level_wavelet'])
		sum = np.zeros(len(wpdmatr_sqr))
		for i in range(len(wpdmatr_sqr)):
			sum[i] = np.sum((wpdmatr_sqr[i]))
			Energy += sum
		
		#+++plot wpd spectrum
		fig, ax = plt.subplots()
		ax.plot(freq, Energy)
		ax.set_xlabel('Frequency [Hz]')
		ax.set_ylabel('Magnitude [V]')
		ax.set_title('WPD Energy Spectrum')

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

def wav_packet_deco(wfm, config):
    wpd = pywt.WaveletPacket(data=wfm, wavelet=config['mother_wavelet'], maxlevel=config['level_wavelet'])
    
    mylevels = [node.path for node in wpd.get_level(config['level_wavelet'], 'freq')] 
    myfreq = np.arange(len(mylevels))*config['fs']/(2**(1+config['level_wavelet']))
    myfreq += config['fs']/(2**(2+config['level_wavelet']))
    
    wpdmatr = []
    wpdmatr_sqr = []
    count = 0
    for level in mylevels:
        mywav = wpd[level].data
        xold = np.linspace(0., 1., len(mywav))
        xnew = np.linspace(0., 1., len(wfm))
        mywav_int = np.interp(x=xnew, xp=xold, fp=mywav)
        wpdmatr_sqr.append(mywav_int**2)
        wpdmatr.append(mywav_int)
    return np.array(wpdmatr), np.array(wpdmatr_sqr), myfreq
	

def max_wv_level(x, mother_wv):
	wv = pywt.Wavelet(mother_wv)
	wvlen = wv.dec_len
	return pywt.dwt_max_level(len(x), filter_len=wvlen)
	
def return_best_wv_level_idx(x, fs, sqr, levels, mother_wv, crit, wv_approx, int_points, freq_values=None, freq_range=None, freq_values_2=None, freq_range_2=None):
	n = len(x)
	coeffs = pywt.wavedec(x, mother_wv, level=levels, mode='periodic')
	print('LEN COEF ', len(coeffs))
	for i in range(levels+1):
		print('..', i)
		print('...', len(coeffs[i]))
	
	vec = np.zeros(levels+1)
	for i in range(levels+1):
		print('evaluate level ', i/levels)
		wsignal = coeffs[i]
		
		if int_points != True:
			wsignal = odd_to_even(wsignal)
			new_fs = downsampling_fs(x, wsignal, fs)
		
		# plt.plot(wsignal)
		# plt.show()
		
		if crit == 'mpr':
			fitness = cal_WV_fitness_hilbert_env_Ncomp_mpr(wsignal, sqr, freq_values, freq_range, new_fs)
		elif crit == 'avg_mpr':
			fitness = cal_WV_fitness_hilbert_env_AVG_mpr(wsignal, sqr, freq_values, freq_range, freq_values_2, freq_range_2, new_fs)
		elif crit == 'kurt_sen':			
			kurt = scipy.stats.kurtosis(wsignal, fisher=False)
			sen = shannon_entropy(wsignal)			
			fitness = kurt/sen
		elif crit == 'kurt':			
			kurt = scipy.stats.kurtosis(wsignal, fisher=False)
			fitness = kurt
		else:
			print('fatal error crit wavelet')
			sys.exit()

		if i == 0 and wv_approx != 'ON':
			vec[i] = -9999
		else:
			vec[i] = fitness
	
	best_level_idx = np.argmax(vec)
	print('best fitness = ', np.max(vec))
	

	outsignal = coeffs[best_level_idx]
	if int_points == True:
		xold = np.linspace(0., 1., len(outsignal))
		xnew = np.linspace(0., 1., n)
		outsignal = np.interp(x=xnew, xp=xold, fp=outsignal)
		new_fs = fs
	else:
		outsignal = odd_to_even(outsignal)
		new_fs = downsampling_fs(x, outsignal, fs)

	
	return outsignal, best_level_idx, new_fs

def return_best_inv_wv_level_idx(x, fs, sqr, levels, mother_wv, crit, wv_approx, freq_values=None, freq_range=None, freq_values_2=None, freq_range_2=None):

	coeffs = pywt.wavedec(x, mother_wv, level=levels, mode='periodic')
	print('LEN COEF ', len(coeffs))
	for i in range(levels+1):
		print('..', i)
		print('...', len(coeffs[i]))
	
	vec = np.zeros(levels+1)
	for i in range(levels+1):
		print('evaluate level ', i/levels)
		wsignal = coeffs[i]
		# wsignal = odd_to_even(wsignal)
		# new_fs = downsampling_fs(x, wsignal, fs)
		
		# plt.plot(wsignal)
		# plt.show()
		

		if crit == 'kurt_sen':			
			kurt = scipy.stats.kurtosis(wsignal, fisher=False)
			sen = shannon_entropy(wsignal)			
			fitness = kurt/sen
		else:
			print('fatal error crit wavelet')
			sys.exit()

		if i == 0 and wv_approx != 'ON':
			vec[i] = -9999
		else:
			vec[i] = fitness
	
	best_level_idx = np.argmax(vec)
	print('best fitness = ', np.max(vec))
	
	outsignal = coeffs[best_level_idx]
	
	# outsignal = odd_to_even(outsignal)
	new_fs = downsampling_fs(x, outsignal, fs)
	
	outsignal = pywt.idwt(cA=None, cD=outsignal, wavelet=mother_wv)
	# new_fs = fs
	
	

	
	return outsignal, best_level_idx, new_fs

def return_best_wv_level_idx_PACKET(x, fs, sqr, levels, mother_wv, crit, wv_approx, freq_values=None, freq_range=None, freq_values_2=None, freq_range_2=None):
	n = len(x)
	
	nico = pywt.WaveletPacket(data=x, wavelet=mother_wv, maxlevel=levels)	
	mylevels = [node.path for node in nico.get_level(levels, 'freq')]

	
	vec = np.zeros(len(mylevels))
	count = 0
	for lvl in mylevels:
		print('evaluate level ', lvl)
		wsignal = nico[lvl].data
		
		# print(len(wsignal))
		# a = input('pause----')
		
		xold = np.linspace(0., 1., len(wsignal))
		xnew = np.linspace(0., 1., n)				
		wsignal_int = np.interp(x=xnew, xp=xold, fp=wsignal)	
		
		
		if crit == 'mpr':
			fitness = cal_WV_fitness_hilbert_env_Ncomp_mpr(wsignal_int, sqr, freq_values, freq_range, fs)
		elif crit == 'avg_mpr':
			fitness = cal_WV_fitness_hilbert_env_AVG_mpr(wsignal_int, sqr, freq_values, freq_range, freq_values_2, freq_range_2, new_fs)
		elif crit == 'kurt_sen':			
			kurt = scipy.stats.kurtosis(wsignal_int, fisher=False)
			sen = shannon_entropy(wsignal_int)			
			fitness = kurt/sen
		elif crit == 'kurt':			
			kurt = scipy.stats.kurtosis(wsignal_int, fisher=False)
			fitness = kurt
		else:
			print('fatal error crit wavelet')
			sys.exit()

		vec[count] = fitness
		count += 1
	
	best_level_idx = np.argmax(vec)
	print('best fitness = ', np.max(vec))
	
	outsignal = nico[mylevels[best_level_idx]].data
	
	xold = np.linspace(0., 1., len(outsignal))
	xnew = np.linspace(0., 1., n)				
	outsignal_int = np.interp(x=xnew, xp=xold, fp=outsignal)

	new_fs = fs
	return outsignal_int, best_level_idx, new_fs

def return_iwv_PACKET_fix_levels(x, fs, max_level, mother_wv, idx_levels):
	n = len(x)
	
	nico = pywt.WaveletPacket(data=x, wavelet=mother_wv, maxlevel=max_level)	
	mylevels = [node.path for node in nico.get_level(max_level, 'freq')]
	
	
	print('inverse WV!')
	gato = pywt.WaveletPacket(data=None, wavelet=mother_wv, maxlevel=max_level)
	
	for idx in idx_levels:	
		gato[mylevels[idx]] = nico[mylevels[idx]].data
	outsignal = gato.reconstruct(update=False)

	return outsignal

def return_wv_PACKET_one_level(x, fs, max_level, mother_wv, idx_level):
	idx_level_one = idx_level[0]
	n = len(x)
	
	nico = pywt.WaveletPacket(data=x, wavelet=mother_wv, maxlevel=max_level)	
	mylevels = [node.path for node in nico.get_level(max_level, 'freq')]
	# freq
	
	outsignal = nico[mylevels[idx_level_one]].data
	
	# gato = pywt.WaveletPacket(data=None, wavelet=mother_wv, maxlevel=max_level)
	# gato[mylevels[idx_level_one]] = nico[mylevels[idx_level_one]].data
	# outsignal = gato.reconstruct(update=False)
	
	return outsignal

def odd_to_even(x):
	if len(x) % 2 != 0:			
		x = x[1:]
	return x

def downsampling_fs(x, newsignal, fs):
	red_fact = len(x)/len(newsignal)
	new_fs = fs / red_fact
	return new_fs

def cal_WV_fitness_hilbert_env_Ncomp_mpr(x, sqr, freq_values, freq_range, fs):

	# plt.plot(x)
	# plt.show()
	if sqr == 'ON':
		print('Squared wavelet!!!')
		x = x**2.0
	x_env = hilbert_demodulation(x)
	magENV, f, df = mag_fft(x_env, fs)
	print(len(magENV))
	print(df)
	mag_freq_value = 0.
	for freq_value in freq_values:
		mag_freq_value += amp_component_zone(X=magENV, df=df, freq=freq_value, tol=4.0)
	
	avg_freq_range = avg_in_band(magENV, df, low=freq_range[0], high=freq_range[1])
	fitness = 20*np.log10((mag_freq_value-avg_freq_range)/avg_freq_range)


	return fitness

def cal_WV_fitness_hilbert_env_AVG_mpr(x, sqr, freq_values, freq_range, freq_values_2, freq_range_2, fs):

	# plt.plot(x)
	# plt.show()
	if sqr == 'ON':
		print('Squared wavelet!!!')
		x = x**2.0
	x_env = hilbert_demodulation(x)
	magENV, f, df = mag_fft(x_env, fs)
	print(len(magENV))
	print(df)
	mag_freq_value = 0.
	mag_freq_value_2 = 0.
	for freq_value in freq_values:
		mag_freq_value += amp_component_zone(X=magENV, df=df, freq=freq_value, tol=2.0)
	for freq_value_2 in freq_values_2:
		mag_freq_value_2 += amp_component_zone(X=magENV, df=df, freq=freq_value_2, tol=2.0)
	
	avg_freq_range = avg_in_band(magENV, df, low=freq_range[0], high=freq_range[1])
	
	fitness = 20*np.log10((mag_freq_value-len(freq_values)*avg_freq_range)/avg_freq_range)
	
	avg_freq_range_2 = avg_in_band(magENV, df, low=freq_range_2[0], high=freq_range_2[1])
	
	fitness_2 = 20*np.log10((mag_freq_value_2-len(freq_values_2)*avg_freq_range_2)/avg_freq_range_2)
	
	fitness_avg = (fitness + fitness_2)/2.


	return fitness_avg

def read_parser(argv, Inputs, InputsOpt_Defaults):
	try:
		Inputs_opt = [key for key in InputsOpt_Defaults]
		Defaults = [InputsOpt_Defaults[key] for key in InputsOpt_Defaults]
		parser = ArgumentParser()
		for element in (Inputs + Inputs_opt):
			print(element)
			if element == 'files' or element == 'widths' or element == 'range' or element == 'filter':
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
	
	#Type conversion to float

	
	
	# config['clusters'] = int(config['clusters'])
	
	
	
	config['level_contour'] = int(config['level_contour'])
	config['level_wavelet'] = int(config['level_wavelet'])
	
	config['fs'] = float(config['fs'])
	
	if config['widths'] != None:
		config['widths'][0] = float(config['widths'][0])
		config['widths'][1] = float(config['widths'][1])
	
	
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
