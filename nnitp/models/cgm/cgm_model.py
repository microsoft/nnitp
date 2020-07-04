import os.path
from math import ceil
import tensorflow as tf
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Input
from keras.layers import concatenate
import wget
import numpy as np
from scipy.io import loadmat
from scipy.linalg import block_diag
from nnitp.keras import Wrapper
from nnitp.datatype import Struct,Numeric,Series
from nnitp.error import LoadError

# Loading the PSO4 dataset and model.
#
# The dataset is available here:
#
#     https://public.jaeb.org/jdrfapp2
#
# The data file is called `PSO4_Public_Dataset.zip`. Unpack it in a subdirectory
# called PSO4_Public_Dataset. 
#
# The model weights should be stored in a MatLab file in this
# directory with the following name:
#
#    40105_poly3scaled_gg_hrSC_meal60_CGM_90_wc.mat
#


def get_data():

    # Download the dataset, if we don't have it. If we fail, we can throw
    # `LoadError` to tell the user what went wrong.

    fname = outfile
    if not os.path.exists(fname):
        try:
            create_csv_file()
        except:
            raise LoadError("Could not generate data file for CGM model")

    # Read the dataset into a numpy array

    try:
        dataset = np.loadtxt(fname, delimiter=',')
    except:
        raise LoadError("Failed to read file '{}'".format(fname))

    # Separate input and label

    X = dataset[:,0:(37+8)]
    Y = dataset[:,(37+8)]
    
    # Separate into 80% training, 20% test sample

    tsize = ceil(len(X) * 0.8)
    x_train = X[:tsize]
    x_test = X[tsize:]
    y_train = Y[:tsize]
    y_test = Y[tsize:]

    # Describe the data so that nnitp can display it.

    # First input: Insulin time series
    # Second input: Glucose data and other fields
    # Second input columns:
    # - CGM coefficient a (3rd order)
    # - CGM coefficient b (2nd order)
    # - CGM coefficient c (1st order)
    # - CGM coefficient d (constant)
    # - Glucose change 30 min
    # - time as sin(2*pi*hr/24)
    # - time as cos(2*pi*hr/24)
    # - carbs last 60 min
    
    description = Struct(
        ('insulin',Series(5.0 * np.array(range(37)))),
        ('glucose',Struct(
            ('a',Numeric()),
            ('b',Numeric()),
            ('c',Numeric()),
            ('d',Numeric()),
            ('CGM change',Numeric()),
            ('sin t',Numeric()),
            ('cos t''age',Numeric()),
            ('carbs 60',Numeric()),
        )),
    )   

    return (x_train, y_train), (x_test, y_test), description

# Fetch the trained model

def get_model():

    # We load the model from weights stored in a MatLab file.

    fname = 'cgm_model.h5'
    if not os.path.exists(fname) or True:
        (x_train, y_train), (x_test, y_test), _ = get_data()
        annots = loadmat('40105_poly3scaled_gg_hrSC_meal60_CGM_90_wc.mat')
        model = Sequential()
        model.add(Dense(16, input_dim=45, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))        
        print ('foo')
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        l1weights = np.transpose(block_diag(annots['W_insulin_1'],annots['W_gluc_1']))
        l1biases = np.concatenate([annots['b_insulin_1'].reshape(8),annots['b_gluc_1'].reshape(8)])
        model.layers[0].set_weights([l1weights,l1biases])
        model.layers[1].set_weights([np.transpose(annots['W_joint_1']),annots['b_joint_1'].reshape(16)])
        model.layers[2].set_weights([np.transpose(annots['W_joint_2']),annots['b_joint_2'].reshape(1)])
        
        # inputs = [Input(shape=(37,)),Input(shape=(8,))]
        # dl = [Dense(8,activation="relu"),Dense(8,activation="relu")]
        # d = [l(inp) for l,inp in zip(dl,inputs)]
        # dl[0].set_weights([np.transpose(annots['W_insulin_1']),annots['b_insulin_1'].reshape(8)])
        # dl[1].set_weights([np.transpose(annots['W_gluc_1']),annots['b_gluc_1'].reshape(8)])
        # nets = [Model(inputs=inp,outputs=out) for inp,out in zip(inputs,d)]
        # conc = concatenate([net.output for net in nets])
        # lj1 = Dense(16,activation="relu")
        # j1 = lj1(conc)
        # lj2 = Dense(1)
        # j2 = lj2(j1)
        # lj1.set_weights([np.transpose(annots['W_joint_1']),annots['b_joint_1'].reshape(16)])
        # lj2.set_weights([np.transpose(annots['W_joint_2']),annots['b_joint_2'].reshape(1)])
        # model = Model(inputs=inputs,outputs=j2)
        
        # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #        model.fit(x_train, y_train, epochs=150, batch_size=10)

        wrap = Wrapper(model)
        print (y_test)
        print (wrap.compute_activation(2,x_test))
        _, accuracy = model.evaluate(x_test, y_test)
        print ('bar')
        print('Accuracy: %.2f' % (accuracy*100))
        model.save(fname)

    # Load the model from disk and wrap it for nnitp
        
    return Wrapper(load_model(fname))

# Interpolation parameters

params = {'size':20000,'alpha':0.98,'gamma':0.6,'mu':0.9,'layers':[1]}
    
# Here we have code to generate the NN input data from the raw dataset.
# To create the network input, we sample random intervals from the
# PSO4 dataset.
#
# The intervals are of duration 180min + t, where t is the prediction
# horizon, say, 90min. Intervals start times are selected uniformly
# over the range from the earliest data record to the latest. An
# interval is rejected if it does not contain have at least one record
# from each file.  So, for example, an interval containing pump
# records but no CGM records would be rejected.
#
# For each interval, the following data are collected:
#
# 1) Insulin time series: 37 data points taken from the BasalRT field
#    at 5min intervals starting at t=0, for a period of 180min. These
#    values are interpolated assuming the rate is piece-wise
#    constant, i.e., it remains at the sampled value until the
#    following sample occurs.
#
# 2) CGM series. The readings of the CGM are fit to a 3rd-order
#    polynomial.  If fewer than four readings are available, the
#    interval is rejected.  The time values of the sample are
#    normalized so that the origin is the start of the interval and
#    are measured in minutes.
#
# 3) CGM change in 30 min. This is cgm(180) - cgm(150), where cgm is the
#    third-order fit described above.
#
# 4) Time. This is the interval onset plus 180min, expressed in the form:
#    sin(2*pi*hr/24), cos(2*pi*hr/24)
#
# 5) Total of the carbs consumed over the interval 120min to 180min (i.e., one
#    hour before the prediction time.
#


import numpy as np
import csv
import datetime as dt
from bisect import bisect_left, bisect_right
import random
from math import pi, sin, cos


files = ['PSO4Pump','PSO4CGM','PSO4SessionFood']
timefields = ['DataDtTm','DataDtTm','MealDtTm']
datadir = 'PSO4_Public_Dataset/DataTables'
delimiter = '|'
suffix = '.txt'
outfile = 'cgm_data.csv'

# Read records from all files into list `records`. Each record is represented
# by a map from field names to string values, with an additional 'file' field
# giving the file name. All records should have a 'DataDtTm' field giving the date and
# time of the record. This is converted from a string to a `datetime` object
# to allow sorting by time.

def rectime(r):
    return r['time']

records = []
record_times = []

def get_records():
    global records
    global record_times
    global start_time
    global end_time
    for file,tf in zip(files,timefields):
        with open(datadir + '/' + file + suffix,'r') as csvfile:
            header = csvfile.readline().strip().split(delimiter)
            for line in csvfile:
                fields = line.strip().split(delimiter)
                record = dict(zip(header,fields))
                record['file'] = file
                record['time'] = dt.datetime.strptime(record[tf],'%Y-%m-%d %H:%M:%S')
                records.append(record)
    # Sort records by time
    records = sorted(records,key=rectime)
    # Store vector of times
    record_times = list(map(rectime,records))

    start_time = record_times[0]
    end_time = record_times[-1]

# Find the earliest record >= a given time.

def earliest(t):
    return bisect_left(record_times,t)

# Find the earliest record after a given time.

def latest(t):
    return bisect_right(record_times,t)

def is_cgm_record(r):
    return r['file'] == 'PSO4CGM'

# Check that we have at least 8 CGM records in the time period
# [t-60min,t].

def enough_cgm_records(t,recs):
    st = t + dt.timedelta(minutes = -60)
    return len(list(x for x in recs if rectime(x) >= st and rectime(x) <= t)) >= 8 

# To get a sample, we choose a random time `t` in the data range.  We
# get all the records in the range [t-360min,t+90min). Within these,
# we select randomly from the patient ids that occur. We search
# backward until we get a pump record the sets the basal rate. We go
# back 6 hours because we want to capture a basal rate record before
# the start of the 180min history period.


def get_sample():
    total_seconds = (end_time - start_time).total_seconds()
    while True:
        sample_seconds = random.randrange(0,total_seconds)
        sample_time = start_time + dt.timedelta(seconds=sample_seconds)
        sample_start = sample_time + dt.timedelta(minutes = -360)
        sample_end = sample_time + dt.timedelta(minutes = 95)
        start_rec = earliest(sample_start)
        end_rec = latest(sample_end)
        sample_records = records[start_rec:end_rec]
        if len(sample_records) == 0:
            continue
        ids = list(sorted(set(x['DeidentID'] for x in sample_records)))
        id = random.sample(ids,1)[0]
        sample_records = [x for x in sample_records if x['DeidentID'] == id]
        if enough_cgm_records(sample_time,sample_records):
            return sample_time,sample_records
       
       

def get_pump_series(t,recs):
#    print ('t = {}'.format(t))
    idx = 0
    basal = -1.0
    basal_t = t + dt.timedelta(minutes= -180)
    hmins = 180
    pmins = 5
    zero = dt.datetime(1901,1,1)
    override_until = zero
    orig_amt = 0.0
    series = []
    for mins in range(-hmins, pmins, pmins):
        ins = 0.0
        pt = t + dt.timedelta(minutes=mins)
        end_t = pt + dt.timedelta(minutes=pmins)
#        print ('pt, end_t = {},{}'.format(pt,end_t))
        while idx < len(recs):
            r = recs[idx]
            if r['file'] == 'PSO4Pump':
                rt = r['time']
                if rt >= end_t:
                    break
#                print (r)
                next_t = min(rt,end_t)
                if override_until > zero and next_t >= override_until:
                    ins += basal * (override_until-basal_t).total_seconds() / 3600
                    basal = orig_amt
                    basal_t = override_until
                    override_until = zero
                if next_t > basal_t and basal < 0.0:
                    print ('no initial basal rate found')
                    return None
                if next_t > basal_t:
#                    print ('dt = {}'.format((next_t-basal_t).total_seconds()))
                    ins += basal * (next_t-basal_t).total_seconds() / 3600
                    basal_t = next_t
                if r['BasalRt'] != '':
                    basal = float(r['BasalRt'])
                if r['TempBasalAmt'] != '':
                    orig_amt = basal
                    fields = list(map(float,r['TempBasalDur'].split(':')))
                    temp_dur = dt.timedelta(hours=fields[0],minutes=fields[1],seconds=fields[2])
                    override_until = rt + temp_dur
                    basal = float(r['TempBasalAmt'])
                if r['BolusDeliv'] != '' and rt >= pt:
                    ins += float(r['BolusDeliv'])
            idx += 1
        if basal_t < end_t:
            if  basal < 0.0:
                print ('no initial basal rate found')
                return None
#            print ('dt = {}'.format((end_t-basal_t).total_seconds()))
            ins += basal * (end_t-basal_t).total_seconds() / 3600
            basal_t = end_t
#        print ('insulin: {}'.format(ins))
        series.append(ins)
    return series

def get_cgm_series(t,recs):
    start_t = t + dt.timedelta(minutes= -180)
    time_vals = []
    cgm_vals = []
    for r in recs:
        if r['file'] == 'PSO4CGM':
            rt = r['time']
            if rt >= start_t and rt <= t and r['CGM'] != '':
                time_vals.append((rt-start_t).total_seconds()/60)
                cgm_vals.append(float(r['CGM']))
    return time_vals,cgm_vals

def get_cgm_coeffs(time_vals,cgm_vals):
    mean = np.mean(time_vals)
    std = np.std(time_vals)
    ntvals = [(x-mean)/std for x in time_vals]
    return list(np.polyfit(ntvals, cgm_vals, 3))
    
# Get the CGM value at a given time, using linear interpolation

def get_cgm_value(t,recs):
    have = False
    old_time = None
    old_val = None
    for r in recs:
        if r['file'] == 'PSO4CGM' and r['CGM'] != '':
            rt = r['time']
            val = float(r['CGM'])
            if have and t >= old_time and t < rt:
                return (old_val + (val - old_val) * ((t-old_time) / (rt-old_time)))
            have = True
            old_time = rt
            old_val = val
    print ('no samples to compute CGM at time {}'.format(t))
    return None


# We get the CGM value 30s ago by linear interpolation

def get_cgm_delta(time_vals,cgm_vals):
    t30 = 30
    for idx in range(len(time_vals)-1):
        t1,t2 = time_vals[idx],time_vals[idx+1]
        if t1 <= t30 and t30 < t2:
            c1,c2 = cgm_vals[idx],cgm_vals[idx+1]
            return [cgm_vals[-1] - (c1 + (c2 - c1) * (t30-t1) / (t2-t1))]
    print ('not enough records to compute cgm at t-30min')
    return None
    
# Get the time of day represented with sin/cos
        
def get_cgm_time(t):
    day = t.replace(hour=0,minute=0,second=0)
    secs = (t-day).total_seconds()
    angle = (secs/(24 * 60 * 60)) * 2 * pi
    return [sin(angle),cos(angle)]

# Get the total grams carbohydrate consumed in last 60 mins

def get_carbs(t,recs):
    start_t = t + dt.timedelta(minutes= -60)
    carbs = 0.0
    for r in recs:
        if r['file'] == 'PSO4SessionFood' and r['GramsCHO'] != '':
            print ('carbs!')
            rt = r['time']
            if rt >= start_t and rt <= t:
                carbs += float(r['GramsCHO'])
    if carbs > 0.0:
        print ('carbs = {}'.format(carbs))
    return [carbs]

def make_sample():
    while True:
        t,recs = get_sample()
        pump = get_pump_series(t,recs)
        if pump is None:
            continue
#        print(pump)
        time_vals,cgm_vals = get_cgm_series(t,recs)
#        print(time_vals,cgm_vals)
        cgm_delta = get_cgm_delta(time_vals,cgm_vals)
        if cgm_delta is None:
            continue
        cgm_time = get_cgm_time(t)
        cgm_coeffs = get_cgm_coeffs(time_vals,cgm_vals)
        carbs = get_carbs(t,recs)
        val = get_cgm_value(t + dt.timedelta(minutes=90),recs)
        if val is None:
            continue
        return pump + cgm_coeffs + cgm_delta + cgm_time + carbs + [val]

#print (get_pump_series(t,recs))

# print (get_cgm_series(t,recs))


# print(make_sample())

def create_csv_file():
    print('Creating CGM data file')
    get_records()
    with open(outfile,'w') as csvfile:
        for idx in range(50000):
            sample = make_sample()
            csvfile.write(','.join(map(str,sample))+'\n')
        
if __name__ == '__main__':
    get_model()
    
