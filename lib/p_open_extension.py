from nptdms import TdmsFile
from scipy.io import loadmat
import pickle
from numpy import ravel
from sys import exit

def load_signal(filename, channel=None, extension=None):
	print('Loaded date: ', filename)
	if extension == None:
		point_index = filename.find('.')
		extension = filename[point_index+1] + filename[point_index+2] + filename[point_index+3]
	
	if extension == 'mat':
		try:
			x = f_open_mat(filename, channel)
			x = ravel(x)
		except:
			import h5py
			with h5py.File(filename, 'r') as f:
				# print(list(f.keys()))
				# print(channel)
				# print(list(f['Ch0']))
				x = list(f[channel])[0]
				# print(type(data))
				# print(data)
	elif extension == 'tdm': #tdms
		x = f_open_tdms(filename, channel)
	elif extension == 'txt':
		x = np.loadtxt(filename)
	elif extension == 'pkl':
		x = read_pickle(filename)
	else:
		print('Error extention')
		exit()
	return x

def f_open_mat(filename, channel):
	if filename == 'Input':
		filename = filedialog.askopenfilename()
	file = loadmat(filename)
	data = file[channel]	
	return data

def save_pickle(pickle_name, pickle_data):
	pik = open(pickle_name, 'wb')
	pickle.dump(pickle_data, pik)
	pik.close()

def read_pickle(pickle_name):
	pik = open(pickle_name, 'rb')
	pickle_data = pickle.load(pik)
	return pickle_data

def f_open_tdms(filename, channel):
	if filename == 'Input':
		filename = filedialog.askopenfilename()
	file = TdmsFile.read(filename)
	all_groups = file.groups()
	group = all_groups[0]
	try:
		data_channel = group[channel]
		data = data_channel[:]
	except:
		print('***error channel, try: ')
		print(group.channels())
	return data

