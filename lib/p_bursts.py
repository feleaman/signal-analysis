import numpy as np
import matplotlib.pyplot as plt

def thr_burst_detector_stella_lockout(x1, config, count=None, threshold=None):
	dt = 1.0/config['fs']
	n_points = len(x1)
	tr = n_points*dt
	t = np.array([i*dt for i in range(n_points)])
	traw = t
	
	if threshold == None:
		threshold = read_threshold(config['thr_mode'], config['thr_value'], x1)
	
	t_burst_corr1 = []
	t_burst_corr_rev = []	
	
	signal = x1
	ini_point_next = 0
	while (ini_point_next + int(config['window_time']*config['fs'])) < len(signal):
		if signal[ini_point_next] >= threshold:
			t_ini_burst = (ini_point_next)/config['fs']
			t_burst_corr1.append(t_ini_burst)
			
			count = 0
			flag = 'OFF'
			window = x1[(ini_point_next) : (ini_point_next) + int(config['window_time']*config['fs'])]
			accu = 0
			for k in range(len(window)):
				accu += 1
				if window[k] < threshold:
					count += 1
				elif window[k] >= threshold:
					count = 0
				if count >= config['stella']:
					flag = 'ON'
					t_end_burst = t_ini_burst + k/config['fs']
					t_burst_corr_rev.append(t_end_burst)
					ini_point_next = ini_point_next + config['lockout']
					break
			if flag == 'OFF':
				t_end_burst = t_ini_burst + accu/config['fs']
				t_burst_corr_rev.append(t_end_burst)
				ini_point_next = ini_point_next + config['lockout']
			
			ini_point_next = ini_point_next + accu
		else:
			ini_point_next += 1
			
	if len(t_burst_corr_rev) != len(t_burst_corr1):
		print('fatal error 788')
		sys.exit()
	amp_burst_corr1 = [x1[int(time_ini*config['fs'])] for time_ini in t_burst_corr1]
	amp_burst_corr_rev = [x1[int(time_end*config['fs'])] for time_end in t_burst_corr_rev]

	return t_burst_corr1, amp_burst_corr1, t_burst_corr_rev, amp_burst_corr_rev

def plot_burst_rev(fig, ax, nax, t, x1, config, t_burst_corr1, amp_burst_corr1, t_burst_corr1rev, amp_burst_corr1rev):

	# if config['n_files'] == 1:
	ax = [ax]

	ax[nax].plot(t, x1)

	
	thr = True
	if thr == True:
		threshold1 = read_threshold(config['thr_mode'], config['thr_value'], x1)
		ax[nax].axhline(threshold1, color='k')		
	ax[nax].plot(t_burst_corr1, amp_burst_corr1, 'ro')
	ax[nax].plot(t_burst_corr1rev, amp_burst_corr1rev, 'mo') 	

	# if config['save_plot'] == 'ON':
		# if zoom == None:
			# zoom_ini = 'All'
		# else:
			# zoom_ini = str(zoom[0])
		# plt.savefig(config['method'] + plotname + '_' + zoom_ini + '.png')
	return

def signal_rms(x): 
	suma = 0
	for i in range(len(x)):
		suma = suma + x[i]**2.0
	suma = suma/len(x)
	suma = suma**0.5
	return suma

def read_threshold(mode, value, x1=None):
	if mode == 'factor_rms':
		threshold1 = value*signal_rms(x1)
	elif mode == 'fixed_value':
		threshold1 = value
	else:
		print('error threshold mode')
		sys.exit()
	return threshold1