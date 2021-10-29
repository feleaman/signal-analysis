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
from p_fft import *
from p_denois import *

#+++options
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'Times New Roman'
fslabel = 16
fstitle = 16
fstick = 14

#+++inputs
Inputs = ['mode', 'channel', 'fs', 'n_plates']
InputsOpt_Defaults = {'mypath':None, 'range':None, 'output':'plot', 'window':'boxcar', 'filter':['OFF', 5.e3, 3], 'cut_points':300, 'only_burst':'OFF', 'threshold':2., 'widths_wavelet':[1, 31], 'mother_wavelet':'morl', 'level_contour':20, 'cmap':'viridis', 'titanium':'OFF', 'name':'myname'}

#YlGnBu
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
		
		
		#+++names
		filename = basename(filepath)
		dummy = filename.find('_')
		material = filename[2:dummy]		
		if config['n_plates'] == '2':
			dummy2 = find_2nd(filename, '_')
			material2 = filename[dummy+1:dummy2]
			materials = material + ' ' + material2
		else:
			if config['titanium'] != 'ON':
				materials = material
			else:
				materials = 'Ti-Grade2'
		
		#+++load signal
		wfm = load_signal(filepath, config['channel'], extension='tdm')
		if config['range'] != None:
			wfm = wfm[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		time = np.arange(len(wfm))/config['fs']


		#+++filter
		if config['filter'][0] != 'OFF':
			print(config['filter'])
			wfm = butter_filter(wfm, config['fs'], config['filter'])
		
		
		#+++only one burst
		if config['only_burst'] == 'ON':
			for i in range(len(wfm)):
				if wfm[i] >= config['threshold']:
					idx = i
					break
			wfm = wfm[int(i-config['cut_points']/20.) : int(i+config['cut_points'])]
			time = np.arange(len(wfm))/config['fs']
		

		#+++plot
		fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.15, right=0.9, bottom=0.175, top=0.9)
		fig.set_size_inches(6, 3.5)
		ax.plot(time*1000*1000, wfm)
		ax.set_xlim(left=0, right=0.315*1000)
		# ax.set_xlim(left=0, right=int(config['cut_points']+config['cut_points']/20))
		ax.set_ylim(bottom=-6.5, top=6.5)
		# ax.set_ylim(bottom=-2.5, top=2.5)
		ax.set_xlabel(r'Time [$\mu$s]', fontsize=fslabel)
		ax.set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax.set_title('Waveform ' + materials, fontsize=fstitle)
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
			
			
	elif config['mode'] == 'cwt':
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
		
		
		#+++names
		filename = basename(filepath)
		dummy = filename.find('_')
		material = filename[2:dummy]		
		if config['n_plates'] == '2':
			dummy2 = find_2nd(filename, '_')
			material2 = filename[dummy+1:dummy2]
			materials = material + ' ' + material2
		else:
			if config['titanium'] != 'ON':
				materials = material
			else:
				materials = 'Ti-Grade2'
		
		#+++load signal
		wfm = load_signal(filepath, config['channel'], extension='tdm')
		if config['range'] != None:
			wfm = wfm[int(config['range'][0]*config['fs']) : int(config['range'][1]*config['fs'])]
		time = np.arange(len(wfm))/config['fs']


		#+++filter
		if config['filter'][0] != 'OFF':
			print(config['filter'])
			wfm = butter_filter(wfm, config['fs'], config['filter'])
		
		
		#+++only one burst
		if config['only_burst'] == 'ON':
			for i in range(len(wfm)):
				if wfm[i] >= config['threshold']:
					idx = i
					break
			wfm = wfm[int(i-config['cut_points']/20.) : int(i+config['cut_points'])]
			time = np.arange(len(wfm))/config['fs']
		
		
		

		
		#+++config cwt
		mother_wv = config['mother_wavelet']
		min_width = config['widths_wavelet'][0]
		max_width = config['widths_wavelet'][1]
		widths = np.arange(min_width, max_width)
		
		cwtmatr, freq = pywt.cwt(data=wfm, scales=widths, wavelet=mother_wv, sampling_period=1./config['fs'])
		cwtmatr = np.abs(cwtmatr)
		print('Frequencies = ', freq)
		
		#+++contour cwt
		# maxxx = np.round(np.max(cwtmatr))
		# minnn = np.round(np.min(cwtmatr))
		maxxx = np.round(np.max(cwtmatr))
		minnn = 0.
		levels_plot = list(np.linspace(minnn, maxxx, num=config['level_contour']))


		# #+++plot
		fig, ax = plt.subplots()
		plt.subplots_adjust(left=0.15, right=0.9, bottom=0.175, top=0.9)
		fig.set_size_inches(6, 3.5)
		cwt = ax.contourf(time*1000*1000, freq/1000., cwtmatr, levels=levels_plot, cmap=config['cmap'])
		cbar = fig.colorbar(cwt, format='%.1f')
		cbar.ax.tick_params(axis='both', labelsize=fstick)
		cbar.set_label('Abs. Wav. Coeff.', fontsize=fslabel)
		
		ax.set_ylim(bottom=50, top=500.)	
		ax.set_xlabel(r'Time [$\mu$s]', fontsize=fslabel)
		ax.set_ylabel('Frequency [kHz]', fontsize=fslabel)
		ax.set_title('Scalogram ' + materials, fontsize=fstitle)
		ax.tick_params(axis='both', labelsize=fstick)
		ax.grid(axis='both')
		
		
		#+++output
		if config['output'] == 'plot':
			plt.show()
		elif config['output'] == 'save':
			# import datetime
			# stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
			# plt.savefig(stamp+'.png')
			plt.savefig('CWT_' + filename[:-5]+ '.svg')
	
	
	elif config['mode'] == 'max_amplitude':
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
		MAX = []
		count = 0
		for filepath, filename in zip(Filepaths, Filenames):
			wfm = load_signal(filepath, config['channel'], extension='tdm')
			time = [i/config['fs'] for i in range(len(wfm))]
			
			#+++filter
			if config['filter'][0] != 'OFF':
				print('with filter! ', config['filter'])
				wfm = butter_filter(wfm, config['fs'], config['filter'])
			
			MAX.append(np.max(np.abs(wfm)))
				
			
			rownames.append(filename)
		
		mean = np.mean(MAX)
		std = np.std(MAX)
		MAX.append(mean)
		MAX.append(std)
		rownames.append('MEAN')
		rownames.append('STD')
		
		mydict['MAX'] = MAX
		#+++output
		if config['output'] == 'save':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Max_Value')
			writer.close()

	
	elif config['mode'] == 'lamb_amplitude':
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
		MAX_75 = []
		MAX_125 = []
		MAX_225 = []
		count = 0
		for filepath, filename in zip(Filepaths, Filenames):
			wfm = load_signal(filepath, 'Sens_3_75mm', extension='tdm')
			#+++filter
			if config['filter'][0] != 'OFF':
				print('with filter! ', config['filter'])
				wfm = butter_filter(wfm, config['fs'], config['filter'])			
			MAX_75.append(np.max(np.abs(wfm)))
			
			
			wfm = load_signal(filepath, 'Sens_1_125mm', extension='tdm')
			#+++filter
			if config['filter'][0] != 'OFF':
				print('with filter! ', config['filter'])
				wfm = butter_filter(wfm, config['fs'], config['filter'])			
			MAX_125.append(np.max(np.abs(wfm)))
			
			
			wfm = load_signal(filepath, 'Sens_2_225mm', extension='tdm')
			#+++filter
			if config['filter'][0] != 'OFF':
				print('with filter! ', config['filter'])
				wfm = butter_filter(wfm, config['fs'], config['filter'])			
			MAX_225.append(np.max(np.abs(wfm)))
				
			
			rownames.append(filename)
		
		mean_75 = np.mean(MAX_75)
		std_75 = np.std(MAX_75)
		MAX_75.append(mean_75)
		MAX_75.append(std_75)
		
		mean_125 = np.mean(MAX_125)
		std_125 = np.std(MAX_125)
		MAX_125.append(mean_125)
		MAX_125.append(std_125)
		
		mean_225 = np.mean(MAX_225)
		std_225 = np.std(MAX_225)
		MAX_225.append(mean_225)
		MAX_225.append(std_225)		
		
		rownames.append('MEAN')
		rownames.append('STD')
		
		mydict['MAX_75'] = MAX_75
		mydict['MAX_125'] = MAX_125
		mydict['MAX_225'] = MAX_225
		
		#+++output
		if config['output'] == 'save':
			writer = pd.ExcelWriter(config['name'] + '.xlsx')
			DataFr_max = pd.DataFrame(data=mydict, index=rownames)				
			DataFr_max.to_excel(writer, sheet_name='Max_Value_Lamb')
			writer.close()
	
	
		
	elif config['mode'] == 'attenuation_auto':
		x = np.array([0., 75., 125., 225.])
		xc = np.arange(226)
		
		V_all = []
		V_all.append( np.array([6., 2.276, 1.540, 1.102]) ) #almg3
		V_all.append( np.array([6., 2.382, 1.037, 0.863]) ) #dxd51d
		V_all.append( np.array([6., 2.294, 2.070, 1.725]) ) #x5crni18-10
		V_all.append( np.array([6., 1.980, 1.461, 1.619]) ) #tigrade2
		
		Alpha_all = []
		V0_all = []
		Yc_all = []
		Vc_all = []
		Error_all = []
		for v in V_all:		
			y = np.log(v)
			coeff = np.polyfit(x, y, deg=1)		
			
			Yc_all.append( coeff[0]*xc + coeff[1] )
			
			alpha = -coeff[0]
			v0 = np.exp(coeff[1])		
			vc = v0*np.exp(-alpha*xc)
			
			Alpha_all.append(alpha)
			V0_all.append(v0)
			Vc_all.append(vc)
			
			evals = np.array([v0*np.exp(-alpha*x[0]), v0*np.exp(-alpha*x[1]), v0*np.exp(-alpha*x[2]), v0*np.exp(-alpha*x[3])])
			error = np.sum((evals - v)**2.0)/4.
			Error_all.append(error)
			# print(evals)
		
		print('Error = ', Error_all)
		print('Alpha = ', Alpha_all)
		
		fig, ax = plt.subplots(nrows=2, ncols=2, sharey=True)
		fig.set_size_inches(11,6)
		plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.45)
		
		ax[0][0].plot(xc, Vc_all[0], label='Fitted curve')
		ax[0][0].scatter(x, V_all[0], label='Measured')
		# ax[0][0].set_title(r'AlMg3 $\alpha$ = ' + str(round(Alpha_all[0], 4)) + ' mm$^{-1}$' , fontsize=fslabel)
		ax[0][0].set_title(r'AlMg3' , fontsize=fslabel)
		
		ax[0][1].plot(xc, Vc_all[1], label='Fitted curve')
		ax[0][1].scatter(x, V_all[1], label='Measured')
		# ax[0][1].set_title(r'DX51D $\alpha$ = ' + str(round(Alpha_all[1], 4)) + ' mm$^{-1}$' , fontsize=fslabel)
		ax[0][1].set_title('DX51D' , fontsize=fslabel)
		ax[0][1].yaxis.set_tick_params(which='both', labelbottom=True)
		
		ax[1][0].plot(xc, Vc_all[2], label='Fitted curve')
		ax[1][0].scatter(x, V_all[2], label='Measured')
		# ax[1][0].set_title(r'X5CrNi18-10 $\alpha$ = ' + str(round(Alpha_all[2], 4)) + ' mm$^{-1}$' , fontsize=fslabel)
		ax[1][0].set_title('X5CrNi18-10' , fontsize=fslabel)
		
		ax[1][1].plot(xc, Vc_all[3], label='Fitted curve')
		ax[1][1].scatter(x, V_all[3], label='Measured')
		# ax[1][1].set_title(r'Ti-Grade2 $\alpha$ = ' + str(round(Alpha_all[3], 4)) + ' mm$^{-1}$' , fontsize=fslabel)
		ax[1][1].set_title(r'Ti-Grade2' , fontsize=fslabel)
		ax[1][1].yaxis.set_tick_params(which='both', labelbottom=True)
		
		
		ax[0][0].set_xlabel('Distance [mm]', fontsize=fslabel)
		ax[0][0].set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax[0][0].tick_params(axis='both', labelsize=fstick)
		ax[0][0].grid(axis='both')
		ax[0][0].legend(fontsize=fstick-1)
		
		ax[0][1].set_xlabel('Distance [mm]', fontsize=fslabel)
		ax[0][1].set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax[0][1].tick_params(axis='both', labelsize=fstick)
		ax[0][1].grid(axis='both')
		ax[0][1].legend(fontsize=fstick-1)
		
		ax[1][0].set_xlabel('Distance [mm]', fontsize=fslabel)
		ax[1][0].set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax[1][0].tick_params(axis='both', labelsize=fstick)
		ax[1][0].grid(axis='both')
		ax[1][0].legend(fontsize=fstick-1)
		
		ax[1][1].set_xlabel('Distance [mm]', fontsize=fslabel)
		ax[1][1].set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax[1][1].tick_params(axis='both', labelsize=fstick)
		ax[1][1].grid(axis='both')
		ax[1][1].legend(fontsize=fstick-1)
		
		# plt.show()
		plt.savefig('Attenuation_Fit' + '.svg')
	
	elif config['mode'] == 'attenuation_auto_sqrt':
		x = np.array([0., 75., 125., 225.])
		xc = np.arange(226)
		
		V_all = []
		V_all.append( np.array([6., 2.276, 1.540, 1.102]) ) #almg3
		V_all.append( np.array([6., 2.382, 1.037, 0.863]) ) #dxd51d
		V_all.append( np.array([6., 2.294, 2.070, 1.725]) ) #x5crni18-10
		V_all.append( np.array([6., 1.980, 1.461, 1.619]) ) #tigrade2
		
		Alpha_all = []
		V0_all = []
		Yc_all = []
		Vc_all = []
		
		for v in V_all:		
		
			vc = 6/np.sqrt(xc)
			vc[0] = 6
			
			Vc_all.append(vc)
			
		
		
		fig, ax = plt.subplots(nrows=2, ncols=2)
		fig.set_size_inches(12,7)
		plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.35)
		
		ax[0][0].plot(xc, Vc_all[0], label='Curve')
		ax[0][0].scatter(x, V_all[0], label='Measured')
		# ax[0][0].set_title(r'AlMg3 $\alpha$ = ' + str(round(Alpha_all[0], 4)) + ' mm$^{-1}$' , fontsize=fslabel)
		
		ax[0][1].plot(xc, Vc_all[1], label='Curve')
		ax[0][1].scatter(x, V_all[1], label='Measured')
		# ax[0][1].set_title(r'DX51D $\alpha$ = ' + str(round(Alpha_all[1], 4)) + ' mm$^{-1}$' , fontsize=fslabel)
		
		ax[1][0].plot(xc, Vc_all[2], label='Curve')
		ax[1][0].scatter(x, V_all[2], label='Measured')
		# ax[1][0].set_title(r'X5CrNi18-10 $\alpha$ = ' + str(round(Alpha_all[2], 4)) + ' mm$^{-1}$' , fontsize=fslabel)
		
		ax[1][1].plot(xc, Vc_all[3], label='Curve')
		ax[1][1].scatter(x, V_all[3], label='Measured')
		# ax[1][1].set_title(r'Ti-Grade2 $\alpha$ = ' + str(round(Alpha_all[3], 4)) + ' mm$^{-1}$' , fontsize=fslabel)
		
		
		ax[0][0].set_xlabel('Distance [mm]', fontsize=fslabel)
		ax[0][0].set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax[0][0].tick_params(axis='both', labelsize=fstick)
		ax[0][0].grid(axis='both')
		ax[0][0].legend(fontsize=fstick-1)
		
		ax[0][1].set_xlabel('Distance [mm]', fontsize=fslabel)
		ax[0][1].set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax[0][1].tick_params(axis='both', labelsize=fstick)
		ax[0][1].grid(axis='both')
		ax[0][1].legend(fontsize=fstick-1)
		
		ax[1][0].set_xlabel('Distance [mm]', fontsize=fslabel)
		ax[1][0].set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax[1][0].tick_params(axis='both', labelsize=fstick)
		ax[1][0].grid(axis='both')
		ax[1][0].legend(fontsize=fstick-1)
		
		ax[1][1].set_xlabel('Distance [mm]', fontsize=fslabel)
		ax[1][1].set_ylabel('Amplitude [V]', fontsize=fslabel)
		ax[1][1].tick_params(axis='both', labelsize=fstick)
		ax[1][1].grid(axis='both')
		ax[1][1].legend(fontsize=fstick-1)
		
		# plt.show()
		plt.savefig('Attenuation_Fit_Sqrt' + '.svg')
	
	elif config['mode'] == 'attenuation_auto_complex':
		#v0 = 6.
		
		x = np.array([1., 75., 125., 225.])
		#dist_cont = np.arange(226)
		
		v = np.array([6., 2.276, 1.540, 1.102]) #almg3
		# v = np.array([6., 2.382, 1.037, 0.863]) #dxd51d
		# v = np.array([6., 2.294, 2.070, 1.725]) #x5crni18-10
		# v = np.array([6., 1.980, 1.461, 1.619]) #tigrade2
		
		def func_(x, alpha, v0):
			return v0*np.exp(-alpha*x)
		
		def func(x, alpha, v0):
			return v0*np.exp(-alpha*x)/np.sqrt(x)
		
		from scipy.optimize import curve_fit
		
		popt, pcov = curve_fit( func, x, v, bounds=(0, [0.0025, 5]), check_finite=False)
		print('popt = ', popt)
		
		xc = np.arange(226)
		vc = func(xc, *popt)
		
		plt.scatter(x, v)
		plt.plot(xc, vc)
		plt.show()
		
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
	config['threshold'] = float(config['threshold'])
	config['cut_points'] = int(config['cut_points'])

	
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

#+++functions
def find_2nd(string, substring):
	return string.find(substring, string.find(substring) + 1)


if __name__ == '__main__':
	main(sys.argv)

   
#plt.rcParams['agg.path.chunksize'] = 1000
#plt.rcParams['savefig.directory'] = os.chdir(os.path.dirname('D:'))
# plt.rcParams['savefig.dpi'] = 1500
# plt.rcParams['savefig.format'] = 'jpeg'
