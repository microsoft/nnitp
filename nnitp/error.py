#
# Copyright (c) Microsoft Corporation.
#

# Error exception types for nnitp

class Error(Exception):
    pass
    
class BadParameter(Error):
    def __init__(self,name):
        self.name = name
    def __str__(self):
        return 'Missing or invalid value for parameter "{}"'.format(self.name)
    
#
# Throw this exception if you can't load a model
#

class LoadError(Error):
    def __init__(self,name):
        self.name = name
    def __str__(self):
        return 'Load error: {}'.format(self.name)
    
# Filter parameters, checking types. This is used to get just the
# keyword parameters of a particular module or function.

def filter_params(input_params, params):
    res = {}
    for key,val in iter(input_params.items()):
        if key in params:
            if type(val) is not params[key]:
                raise BadParameter(key)
            res[key] = val
    return res

# Check for undefined parameters

def check_params(input_params,*params_list):
    params = {}
    for p in params_list:
        params.update(params_list)
    for key in input_params:
        if key not in params:
            raise BadParameter(key)
    
